"""
Tool-using agent loop.

  user question --> model --(tool_use)--> deterministic tool --(tool_result)--> model --> ...
                                                                                  |
                                                  final text OR terminal "submit_*" tool

Guarantees:
  * every tool call + result is recorded as an evidence trail for the UI
  * a hard step budget; on the last step the model is told to wrap up
  * a grounding report: which figures in the answer trace back to tool output
  * structured outputs are sanitised (citations to non-existent evidence are dropped)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.llm.tools import describe_call, execute_tool

RESULT_CHAR_CAP = 14000   # what the model sees per tool result
UI_CHAR_CAP = 8000        # what the browser receives per evidence item


@dataclass
class AgentResult:
    answer: str | None = None
    structured: dict | None = None
    evidence: list[dict] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    steps: int = 0
    stopped: str = "end_turn"          # end_turn | terminal_tool | max_steps
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


# --------------------------------------------------------------------------
# grounding: do the numbers in the answer exist in tool output?
# --------------------------------------------------------------------------
_NUM = re.compile(r"(?<![A-Za-z_\d.])(\d[\d,]*(?:\.\d+)?)\s?(%|[kKmMbB](?![A-Za-z]))?")
_MULT = {"k": 1e3, "m": 1e6, "b": 1e9}


def _collect_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        if np.isfinite(obj):
            out.append(abs(float(obj)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    elif isinstance(obj, str):
        # numbers embedded in text fields (e.g. period labels "2024-07") count too
        for m in re.finditer(r"\d+(?:\.\d+)?", obj):
            out.append(float(m.group()))


def _collect_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("evidence_ids",):
                _collect_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_strings(v, out)


def grounding_report(text: str, evidence: list[dict]) -> dict:
    """
    Extract figures from `text` and check each against numbers present in the
    tool inputs/outputs. Untraced figures are not necessarily wrong (the model may
    have derived a difference), but they are surfaced so the user can verify.
    """
    pool: list[float] = []
    for ev in evidence:
        _collect_numbers(ev.get("output"), pool)
        _collect_numbers(ev.get("input"), pool)
    arr = np.array(pool) if pool else np.array([])

    checked, traced, untraced = 0, 0, []
    for m in _NUM.finditer(text or ""):
        raw, suffix = m.group(1), (m.group(2) or "")
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not suffix and decimals == 0 and (val < 10 or 1900 <= val <= 2100):
            continue                      # list numbers, "top 3", years
        # (candidate value, unit scale): rounding tolerance must scale with the unit,
        # e.g. "$1.2M" is 1.2 +/- 0.05 *millions*, i.e. +/- 50,000.
        cands = [(val, 1.0)]
        if suffix == "%":
            cands.append((val / 100, 0.01))
        elif suffix.lower() in _MULT:
            cands.append((val * _MULT[suffix.lower()], _MULT[suffix.lower()]))
        checked += 1
        ok = bool(arr.size) and any(
            np.any(np.abs(arr - c) <= max(0.5 * 10 ** (-decimals) * scale, 0.01 * abs(c)))
            for c, scale in cands
        )
        if ok:
            traced += 1
        else:
            untraced.append(m.group(0).strip())
    return {
        "checked": checked, "traced": traced,
        "score": round(traced / checked, 3) if checked else None,
        "untraced": untraced[:8],
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _block_to_dict(b: Any) -> dict | None:
    t = getattr(b, "type", None)
    if t == "text":
        return {"type": "text", "text": b.text} if (b.text or "").strip() else None
    if t == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return None


def _clip(obj: Any, cap: int) -> Any:
    s = json.dumps(obj, default=str)
    if len(s) <= cap:
        return obj
    return {"truncated": True, "note": "Output too large to display; query narrower.", "preview": s[:cap // 2]}


def sanitize_structured(data: dict, valid_ids: set[int]) -> dict:
    """Drop citations to evidence that does not exist; flag claims with no support."""
    def fix(item: dict) -> None:
        ids = [i for i in item.get("evidence_ids", []) if isinstance(i, int) and i in valid_ids]
        item["evidence_ids"] = ids
        if not ids:
            item["unsupported"] = True

    for key in ("insights", "drivers"):
        for item in data.get(key, []) or []:
            if isinstance(item, dict):
                fix(item)
                if key == "insights" and item.get("unsupported"):
                    item["confidence"] = "low"
    return data


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------
def run_agent(
    client: Any, model: str, system: str, messages: list[dict], df: pd.DataFrame,
    tools: list[dict], terminal_tool: str | None = None, max_steps: int = 8, max_tokens: int = 2048,
) -> AgentResult:
    result = AgentResult()
    messages = list(messages)
    all_tools = list(tools)

    for step in range(max_steps):
        result.steps = step + 1
        last_step = step == max_steps - 1
        kwargs: dict[str, Any] = dict(model=model, max_tokens=max_tokens, system=system,
                                      tools=all_tools, messages=messages)
        if last_step and terminal_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": terminal_tool}

        resp = client.messages.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            result.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            result.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0

        blocks = [d for d in (_block_to_dict(b) for b in resp.content) if d]
        tool_calls = [b for b in blocks if b["type"] == "tool_use"]

        if not tool_calls:
            result.answer = "\n".join(b["text"] for b in blocks if b["type"] == "text").strip() or None
            result.stopped = "end_turn"
            break

        messages.append({"role": "assistant", "content": blocks})
        tool_results, structured = [], None

        for call in tool_calls:
            if terminal_tool and call["name"] == terminal_tool:
                structured = call["input"]
                tool_results.append({"type": "tool_result", "tool_use_id": call["id"],
                                     "content": json.dumps({"status": "received"})})
                continue
            out = execute_tool(call["name"], call["input"], df)
            ev_id = len(result.evidence)
            result.evidence.append({
                "id": ev_id, "tool": call["name"], "label": describe_call(call["name"], call["input"]),
                "input": call["input"], "output": _clip(out, UI_CHAR_CAP), "error": "error" in out,
            })
            payload = json.dumps({"evidence_id": ev_id, "result": out}, default=str)
            if len(payload) > RESULT_CHAR_CAP:
                payload = json.dumps({"evidence_id": ev_id, "warning": "Result truncated; narrow the query "
                                      "(add filters or lower limit).", "partial": payload[:RESULT_CHAR_CAP]})
            tool_results.append({"type": "tool_result", "tool_use_id": call["id"], "content": payload})

        if structured is not None:
            result.structured = sanitize_structured(structured, {e["id"] for e in result.evidence})
            result.stopped = "terminal_tool"
            break

        if step == max_steps - 2:  # the NEXT call is the last: tell the model to wrap up now
            tool_results.append({"type": "text", "text": "Tool budget almost exhausted. Give your final "
                                 "answer now using only the results you already have; state any limitations."})
        messages.append({"role": "user", "content": tool_results})
    else:
        result.stopped = "max_steps"
        result.answer = result.answer or (
            "I ran out of analysis steps before reaching a conclusion. The calculations I did complete "
            "are listed under Evidence; try a narrower question.")

    # grounding over whatever text the user will read
    if result.structured is not None:
        texts: list[str] = []
        _collect_strings(result.structured, texts)
        result.grounding = grounding_report("\n".join(texts), result.evidence)
    elif result.answer:
        result.grounding = grounding_report(result.answer, result.evidence)
    return result
