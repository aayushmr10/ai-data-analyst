"""High-level LLM features built on the agent loop."""
from __future__ import annotations

from app.llm import config
from app.llm.agent import AgentResult, run_agent
from app.llm.prompts import BRIEFING_TOOL, ROOT_CAUSE_TOOL, system_prompt
from app.llm.tools import TOOL_SCHEMAS
from app.store import DatasetSession

MAX_HISTORY = 12
MAX_MESSAGE_CHARS = 2000


def _clean_history(history: list[dict]) -> list[dict]:
    """Keep the last N plain-text turns; the conversation must start with a user turn."""
    turns = [
        {"role": h["role"], "content": str(h["content"])[:8000]}
        for h in history[-MAX_HISTORY:]
        if h.get("role") in ("user", "assistant") and str(h.get("content", "")).strip()
    ]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return turns


def _serialise(r: AgentResult) -> dict:
    return {
        "answer": r.answer, "structured": r.structured, "evidence": r.evidence,
        "grounding": r.grounding, "steps": r.steps, "stopped": r.stopped, "usage": r.usage,
    }


def chat(session: DatasetSession, message: str, history: list[dict]) -> dict:
    message = message.strip()[:MAX_MESSAGE_CHARS]
    client = config.get_client()
    msgs = _clean_history(history) + [{"role": "user", "content": message}]
    res = run_agent(
        client, config.model_name(),
        system_prompt("chat", session.filename, session.df, session.cleaning_log),
        msgs, session.df, TOOL_SCHEMAS, max_steps=config.max_steps(), max_tokens=2048,
    )
    return _serialise(res)


def briefing(session: DatasetSession) -> dict:
    client = config.get_client()
    res = run_agent(
        client, config.model_name(),
        system_prompt("briefing", session.filename, session.df, session.cleaning_log),
        [{"role": "user", "content": "Analyse this dataset and submit the briefing."}],
        session.df, [*TOOL_SCHEMAS, BRIEFING_TOOL], terminal_tool="submit_briefing",
        max_steps=max(config.max_steps(), 10), max_tokens=3000,
    )
    return _serialise(res)


def root_cause(session: DatasetSession, question: str, metric: str | None, date_column: str | None) -> dict:
    client = config.get_client()
    hints = []
    if metric:
        hints.append(f"The user says the metric of interest is '{metric}'.")
    if date_column:
        hints.append(f"The user says the date column is '{date_column}'.")
    prompt = f"Question: {question.strip()[:MAX_MESSAGE_CHARS]}" + ("\n" + " ".join(hints) if hints else "")
    res = run_agent(
        client, config.model_name(),
        system_prompt("root_cause", session.filename, session.df, session.cleaning_log),
        [{"role": "user", "content": prompt}],
        session.df, [*TOOL_SCHEMAS, ROOT_CAUSE_TOOL], terminal_tool="submit_root_cause",
        max_steps=max(config.max_steps(), 10), max_tokens=3000,
    )
    return _serialise(res)
