from __future__ import annotations

from pydantic import BaseModel


class CleaningAction(BaseModel):
    action: str
    column: str | None = None


class ApplyCleaningRequest(BaseModel):
    actions: list[CleaningAction]


class ChartRequest(BaseModel):
    column: str | None = None
    x: str | None = None
    y: str | None = None
    date_column: str | None = None
    value_column: str | None = None
    bins: int | None = 20


class MLRequest(BaseModel):
    target: str | None = None
    n_clusters: int | None = None


class ReportRequest(BaseModel):
    target: str | None = None
    include_ml: bool = True
