"""Small Anthropic-message API compatibility layer backed by Google's Gemini SDK."""
from __future__ import annotations
import json
from types import SimpleNamespace
from google import genai
from google.genai import types

class GeminiMessages:
    def __init__(self, api_key: str, timeout: float = 90.0, max_retries: int = 2):
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000))
        self.call_names = {}
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
                        elif block.get('type') == 'tool_use': parts.append(types.Part.from_function_call(name=block['name'], args=block.get('input',{})))
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
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens, tools=gem_tools, temperature=0.2)
        if tool_choice and tool_choice.get('type') == 'tool':
            config.tool_config = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='ANY', allowed_function_names=[tool_choice['name']]))
        response = self.client.models.generate_content(model=model, contents=contents, config=config)
        blocks=[]
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part,'text',None): blocks.append(SimpleNamespace(type='text', text=part.text))
                fc=getattr(part,'function_call',None)
                if fc and fc.name:
                    # Agent expects a unique id, though execution is local and deterministic.
                    call_id=f'gemini-{len(self.call_names)}'
                    self.call_names[call_id]=fc.name
                    blocks.append(SimpleNamespace(type='tool_use', id=call_id, name=fc.name, input=dict(fc.args or {})))
        usage=response.usage_metadata
        return SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=getattr(usage,'prompt_token_count',0) or 0, output_tokens=getattr(usage,'candidates_token_count',0) or 0))

class _Messages:
    def __init__(self, key): self._client=GeminiMessages(key)
    def create(self, **kwargs): return self._client.create(**kwargs)
class GeminiCompatClient:
    def __init__(self, api_key): self.messages=_Messages(api_key)
