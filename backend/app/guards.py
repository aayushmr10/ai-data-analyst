"""
Deployment guards for a PUBLIC host: upload cap, optional access code for the AI
features, and rate limits that protect your API key from being drained.

All settings are read at call time (not import time) so they can be changed
without code edits and are easy to test.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request


def _int_env(name: str, default: int, lo: int = 0, hi: int = 10_000_000) -> int:
    try:
        return max(lo, min(int(os.getenv(name, default)), hi))
    except ValueError:
        return default


def max_upload_bytes() -> int:
    return _int_env("MAX_UPLOAD_MB", 25, 1, 500) * 1024 * 1024


def access_code() -> str:
    return os.getenv("APP_ACCESS_CODE", "").strip()


def requires_code() -> bool:
    return bool(access_code())


def client_ip(request: Request) -> str:
    """
    Behind a reverse proxy the right-most X-Forwarded-For entry is the one the proxy
    itself appended (earlier entries can be forged by the client). Best-effort only:
    the global daily cap below is the protection that does not depend on this.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip() or "unknown"
    return request.client.host if request.client else "unknown"


class _Limiter:
    """Sliding-window limits: per-IP/hour, global/day, and failed-code attempts per IP."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.per_ip: dict[str, deque] = defaultdict(deque)
        self.fails: dict[str, deque] = defaultdict(deque)
        self.day: deque = deque()

    @staticmethod
    def _purge(q: deque, window: float, now: float) -> None:
        while q and now - q[0] > window:
            q.popleft()

    def failed_code(self, ip: str, now: float | None = None) -> None:
        now = now or time.time()
        with self._lock:
            q = self.fails[ip]
            self._purge(q, 3600, now)
            if len(q) >= 10:
                raise HTTPException(429, "Too many incorrect access codes. Try again in an hour.")
            q.append(now)

    def check(self, ip: str, now: float | None = None) -> None:
        now = now or time.time()
        per_hour = _int_env("AI_RATE_LIMIT_PER_HOUR", 30)
        per_day = _int_env("AI_DAILY_CAP", 300)
        with self._lock:
            if per_day:
                self._purge(self.day, 86400, now)
                if len(self.day) >= per_day:
                    raise HTTPException(429, "The daily AI usage limit for this deployment has been reached. "
                                             "Try again tomorrow.")
            if per_hour:
                q = self.per_ip[ip]
                self._purge(q, 3600, now)
                if len(q) >= per_hour:
                    raise HTTPException(429, f"AI request limit reached ({per_hour} per hour). Try again later.")
                q.append(now)
            if per_day:
                self.day.append(now)
            if len(self.per_ip) > 5000:  # keep memory bounded
                for k in [k for k, v in self.per_ip.items() if not v]:
                    del self.per_ip[k]


limiter = _Limiter()


def require_ai_access(request: Request, x_access_code: str | None = Header(default=None)) -> None:
    """FastAPI dependency for every endpoint that costs money (calls the LLM)."""
    ip = client_ip(request)
    code = access_code()
    if code:
        supplied = (x_access_code or "").encode()
        if not hmac.compare_digest(supplied, code.encode()):
            if supplied:
                limiter.failed_code(ip)
            raise HTTPException(403, "Access code required or incorrect.")
    limiter.check(ip)
