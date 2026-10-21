"""AI endpoints. Kept in a router so the core app works fully without an API key."""
from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import guards
from app.llm import config, service
from app.llm.prompts import suggested_questions
from app.store import store

router = APIRouter(prefix="/api")

_briefing_cache: dict[str, dict] = {}


class ChatTurn(BaseModel):
    role: str
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = []


class RootCauseRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    metric: str | None = None
    date_column: str | None = None


class BriefingRequest(BaseModel):
    force: bool = False


def _session(dataset_id: str):
    s = store.get(dataset_id)
    if s is None:
        raise HTTPException(404, "Dataset not found. Upload it again.")
    return s


def _call(fn, *args):
    """Run an LLM feature and translate failures into clean HTTP errors."""
    try:
        return fn(*args)
    except config.LLMNotConfigured as e:
        raise HTTPException(503, str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        if name == "AuthenticationError":
            raise HTTPException(401, "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.")
        if name == "RateLimitError":
            raise HTTPException(429, "The AI provider is rate-limiting requests. Try again shortly.")
        if name in ("APIConnectionError", "APITimeoutError"):
            raise HTTPException(504, "Could not reach the AI provider. Try again.")
        if name in ("APIStatusError", "BadRequestError", "NotFoundError", "InternalServerError"):
            raise HTTPException(502, f"The AI provider returned an error ({name}). Check LLM_MODEL and your key.")
        raise


def _fingerprint(session) -> str:
    raw = f"{session.id}|{session.df.shape}|{len(session.cleaning_log)}|{config.model_name()}"
    return hashlib.sha1(raw.encode()).hexdigest()


@router.get("/llm/status")
def llm_status():
    return {**config.status(), "requires_code": guards.requires_code()}


@router.get("/datasets/{dataset_id}/chat/suggestions")
def chat_suggestions(dataset_id: str):
    return {"questions": suggested_questions(_session(dataset_id).df)}


@router.post("/datasets/{dataset_id}/chat", dependencies=[Depends(guards.require_ai_access)])
def chat(dataset_id: str, req: ChatRequest):
    session = _session(dataset_id)
    return _call(service.chat, session, req.message, [t.model_dump() for t in req.history])


@router.post("/datasets/{dataset_id}/briefing", dependencies=[Depends(guards.require_ai_access)])
def briefing(dataset_id: str, req: BriefingRequest):
    session = _session(dataset_id)
    key = _fingerprint(session)
    if not req.force and key in _briefing_cache:
        return {**_briefing_cache[key], "cached": True}
    out = _call(service.briefing, session)
    if out.get("structured"):
        _briefing_cache[key] = out
    return out


@router.post("/datasets/{dataset_id}/root-cause", dependencies=[Depends(guards.require_ai_access)])
def root_cause(dataset_id: str, req: RootCauseRequest):
    session = _session(dataset_id)
    for label, col in (("metric", req.metric), ("date_column", req.date_column)):
        if col and col not in session.df.columns:
            raise HTTPException(400, f"{label} '{col}' is not a column in this dataset.")
    return _call(service.root_cause, session, req.question, req.metric, req.date_column)
