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
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        text = str(e)
        low = text.lower()
        # ---- Google Gemini (google.genai.errors.ClientError / ServerError) ----
        if code in (401, 403) or "api key not valid" in low or "api_key_invalid" in low or "permission_denied" in low:
            raise HTTPException(401, "The AI API key was rejected. Check GEMINI_API_KEY (or ANTHROPIC_API_KEY) on your "
                                     "server (Render > Environment) and redeploy.")
        if (code == 429 or "resource_exhausted" in low) and "perday" in low.replace("_", "").replace(" ", ""):
            raise HTTPException(429, "Your Gemini key has used up its DAILY free quota for this model. Try again "
                                     "tomorrow, enable billing in Google AI Studio, or set LLM_MODEL to another model.")
        if code == 429 or "resource_exhausted" in low or name == "RateLimitError":
            raise HTTPException(429, "The Gemini API quota/rate limit was hit (free tier allows only a few requests "
                                     "per minute). Wait a minute and try again, or enable billing on the key.")
        if code == 404 or "not found" in low and "model" in low:
            raise HTTPException(502, f"Gemini model '{config.model_name()}' was not found. Set LLM_MODEL "
                                     "(e.g. gemini-3.8-flash) on your server.")
        if code in (500, 502, 503, 504) or name in ("ServerError", "InternalServerError"):
            raise HTTPException(503, "The Gemini service is busy or overloaded. Try again in a moment.")
        if code == 400 or name in ("ClientError", "BadRequestError", "APIStatusError"):
            raise HTTPException(502, f"The AI provider rejected the request: {text[:300]}")
        if name == "AuthenticationError":
            raise HTTPException(401, "The AI API key was rejected. Check GEMINI_API_KEY (or ANTHROPIC_API_KEY).")
        if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "TimeoutException"):
            raise HTTPException(504, "Could not reach the AI provider. Try again.")
        # Never leak a bare 500: a 500 skips CORS headers and the browser shows only "Something went wrong".
        raise HTTPException(500, f"AI analysis failed ({name}): {text[:300]}")


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
