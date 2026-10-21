"""Serve the built React app from FastAPI so the whole product is ONE container / ONE origin."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_frontend(app: FastAPI, static_dir: str) -> bool:
    index = os.path.join(static_dir, "index.html")
    if not os.path.isfile(index):
        return False
    root = os.path.realpath(static_dir)
    assets = os.path.join(static_dir, "assets")
    if os.path.isdir(assets):
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        candidate = os.path.realpath(os.path.join(root, full_path))
        # serve real files (favicon etc.) but never anything outside the static dir
        if full_path and candidate.startswith(root + os.sep) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(index)  # client-side routing fallback

    return True
