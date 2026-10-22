"""Small Anthropic-message API compatibility layer backed by Google's Gemini SDK."""
from __future__ import annotations
import base64
import binascii
import json
import os
import re
import threading
import time
from types import SimpleNamespace
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# Gemini 2.5 "thinking" tokens count against max_output_tokens; without headroom the reply can come
# back empty (finish_reason=MAX_TOKENS). Retry transient errors (429 rate limit / 5xx overload).
THINKING_HEADROOM = 4096
RETRY_DELAYS = (5, 15, 30)

# Free-tier Gemini keys allow only a handful of requests per minute, and one briefing makes ~10 calls.
# Space the calls out (seconds between calls; LLM_MIN_INTERVAL_SEC=0 disables) so we stay under the limit.
_throttle_lock = threading.Lock()
_last_call = [0.0]


def _throttle() -> None:
    try:
        gap = float(os.getenv("LLM_MIN_INTERVAL_SEC", "3"))
    except ValueError:
        gap = 3.0
    if gap <= 0:
        return
    with _throttle_lock:
        wait = _last_call[0] + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _retry_wait(err, default: float) -> float:
    """Use the delay Gemini asks for ('retry in 26s' / retryDelay) when it gives one."""
    m = re.search(r"retry(?: in|Delay['\": ]+)\s*([\d.]+)\s*s", str(err), re.I)
    return min(float(m.group(1)) + 1, 60.0) if m else default

class GeminiMessages:
    def __init__(self, api_key: str, timeout: float = 90.0, max_retries: int = 2):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000))
        self.call_names = {}
        self.raw_parts = {}  # call_id -> original Gemini part (keeps the thought_signature Gemini 3 requires)
    @staticmethod
    def _replayable(part):
        """Copy of a function-call part safe to send back. The SDK hands thought_signature over still
        base64-encoded and then base64-encodes it again on the way out, which Gemini 3 rejects as
        'Corrupted thought signature'. Decode once here so the round trip is exact."""
        sig = getattr(part, 'thought_signature', None)
        if isinstance(sig, (bytes, bytearray)) and sig:
            try:
                sig = base64.b64decode(bytes(sig), validate=True)
            except (binascii.Error, ValueError):
                pass
        return types.Part(function_call=part.function_call, thought_signature=sig or None)

    def create(self, *, model, max_tokens, system, tools, messages, tool_choice=None):
        declarations = []
        for t in tools:
            declarations.append(types.FunctionDeclaration(name=t['name'], description=t.get('description',''), parameters_json_schema=t.get('input_schema', {'type':'object','properties':{}})))
        gem_tools = [types.Tool(function_declarations=declarations)] if declarations else None
        contents = []
        for msg in messages:
            role, content = msg.get('role'), msg.get('content', '')
            if role == 'assistant':
                parts=[]
                if isinstance(content, str): parts=[types.Part.from_text(text=content)] if content else []
                else:
                    for block in content:
                        if block.get('type') == 'text': parts.append(types.Part.from_text(text=block.get('text','')))
                        elif block.get('type') == 'tool_use':
                            raw = self.raw_parts.get(block.get('id'))
                            parts.append(raw if raw is not None else types.Part.from_function_call(name=block['name'], args=block.get('input',{})))
                if parts: contents.append(types.Content(role='model', parts=parts))
            elif role == 'user':
                if isinstance(content, str): contents.append(types.Content(role='user', parts=[types.Part.from_text(text=content)]))
                else:
                    # Anthropic tool_result blocks become Gemini function responses.
                    parts=[]
                    for block in content:
                        if block.get('type') == 'tool_result':
                            parts.append(types.Part.from_function_response(name=self.call_names.get(block.get('tool_use_id'), 'unknown_tool'), response={'result': block.get('content','')}))
                        elif block.get('type') == 'text': parts.append(types.Part.from_text(text=block.get('text','')))
                    if parts: contents.append(types.Content(role='user', parts=parts))
        cfg = dict(system_instruction=system, max_output_tokens=max_tokens + THINKING_HEADROOM, tools=gem_tools)
        if not str(model).startswith('gemini-3'):
            cfg['temperature'] = 0.2  # Gemini 3 works best at its default temperature
        low_think = (str(model).startswith('gemini-3') and os.getenv('LLM_THINKING', 'low')
                     and 'thinking_level' in types.ThinkingConfig.model_fields)
        if low_think:  # Gemini 3 thinks a lot by default; 'low' keeps each tool step fast
            cfg['thinking_config'] = types.ThinkingConfig(thinking_level=os.getenv('LLM_THINKING', 'low'))
        config = types.GenerateContentConfig(**cfg)
        if tool_choice and tool_choice.get('type') == 'tool':
            config.tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='ANY', allowed_function_names=[tool_choice['name']]))
        response = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                _throttle()
                response = self.client.models.generate_content(model=model, contents=contents, config=config)
                break
            except genai_errors.APIError as e:
                code = getattr(e, 'code', None)
                if code == 400 and 'thinking' in str(e).lower() and getattr(config, 'thinking_config', None):
                    config.thinking_config = None  # this model doesn't accept it: retry once without
                    continue
                daily = 'perday' in str(e).lower().replace('_', '').replace(' ', '')
                if code in (429, 500, 502, 503, 504) and attempt < len(RETRY_DELAYS) and not daily:
                    time.sleep(_retry_wait(e, RETRY_DELAYS[attempt]))
                    continue
                fb = os.getenv('LLM_FALLBACK_MODEL', '').strip()
                if fb and fb != model and code in (404, 429, 500, 502, 503, 504):
                    return self.create(model=fb, max_tokens=max_tokens, system=system, tools=tools,
                                       messages=messages, tool_choice=tool_choice)
                raise
        blocks=[]
        for cand in (response.candidates or []):
            content = getattr(cand, 'content', None)
            for part in (getattr(content, 'parts', None) or []):
                if getattr(part,'text',None): blocks.append(SimpleNamespace(type='text', text=part.text))
                fc=getattr(part,'function_call',None)
                if fc and fc.name:
                    # Agent expects a unique id, though execution is local and deterministic.
                    call_id=f'gemini-{len(self.call_names)}'
                    self.call_names[call_id]=fc.name
                    self.raw_parts[call_id]=self._replayable(part)
                    blocks.append(SimpleNamespace(type='tool_use', id=call_id, name=fc.name, input=dict(fc.args or {})))
        usage=response.usage_metadata
        return SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=getattr(usage,'prompt_token_count',0) or 0, output_tokens=getattr(usage,'candidates_token_count',0) or 0))

class _Messages:
    def __init__(self, key): self._client=GeminiMessages(key)
    def create(self, **kwargs): return self._client.create(**kwargs)
class GeminiCompatClient:
    def __init__(self, api_key): self.messages=_Messages(api_key)
