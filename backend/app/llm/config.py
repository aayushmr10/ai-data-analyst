"""LLM settings. Everything is env-driven so no secret ever lives in code."""
from __future__ import annotations

import os

DEFAULT_MODEL = "gemini-2.5-flash"


class LLMNotConfigured(RuntimeError):
    """Raised when no API key is set; routes translate this to HTTP 503."""


_client_override = None  # tests inject a fake client here


def set_client_override(client) -> None:
    global _client_override
    _client_override = client


def model_name() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def max_steps() -> int:
    try:
        return max(2, min(int(os.getenv("LLM_MAX_STEPS", "8")), 15))
    except ValueError:
        return 8


def status() -> dict:
    return {
        "configured": _client_override is not None or bool(os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
        "provider": "gemini" if os.getenv("GEMINI_API_KEY") else "anthropic",
        "model": model_name(),
    }


def get_client():
    if _client_override is not None:
        return _client_override
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        from app.llm.gemini_compat import GeminiCompatClient
        return GeminiCompatClient(api_key=gemini_key)
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise LLMNotConfigured(
            "AI features are disabled: set GEMINI_API_KEY in your .env file (recommended), "
            "or ANTHROPIC_API_KEY, then restart the backend."
        )
    import anthropic
    return anthropic.Anthropic(api_key=key, timeout=90.0, max_retries=2)
