"""
Deterministic analysis tools exposed to the LLM agent.

Design rule: the LLM never computes anything itself. It decides WHICH tool
to call and with WHAT arguments; every number in an answer must come from a
tool result. Tools validate their inputs and return helpful errors (with the
list of valid columns) so the model can self-correct.

No tool executes arbitrary code or SQL -- only a fixed, parameterised set of
pandas operations.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as sps

from app.ml.engine import run_auto_ml
from app.profiling.profiler import (
    _infer_kind, cleaning_suggestions, column_profile, dataset_overview,
)
from app.statistics.engine import _correlation_strength_label
from app.utils.serialize import safe_json

MAX_GROUPS = 25
GRAIN_FREQ = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
AGG_FUNCS = {"sum": "sum", "mean": "mean", "median": "median", "min": "min", "max": "max", "std": "std"}
ADDITIVE = ("sum", "count")


class ToolError(Exception):
    """Raised for invalid tool input; message is returned to the model."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _need_col(df: pd.DataFrame, col: Any, label: str = "column") -> None:
    if not isinstance(col, str) or col not in df.columns:
        raise ToolError(f"{label} '{col}' not found. Available columns: {list(df.columns)}")


def _need_numeric(df: pd.DataFrame, col: Any) -> None:
    _need_col(df, col)
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ToolError(
            f"column '{col}' is not numeric (dtype {df[col].dtype}); "
            f"use agg='count' or 'nunique', or pick a numeric column."
        )


def _dt(df: pd.DataFrame, col: str) -> pd.Series:
    _need_col(df, col, "date column")
    s = df[col]
    if not pd.api.types.is_datetime64_any_dtype(s):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = pd.to_datetime(s, errors="coerce")
    if s.notna().sum() == 0:
        raise ToolError(f"column '{col}' could not be parsed as dates.")
    return s


def _check_metric(df: pd.DataFrame, metric: Any, agg: str) -> None:
    if agg not in (*AGG_FUNCS, "count", "nunique"):
        raise ToolError(f"unknown agg '{agg}'. Use one of: sum, mean, median, min, max, std, count, nunique.")
    if agg == "count":
        return
    if metric is None:
        raise ToolError(f"agg='{agg}' requires a metric column.")
    if agg == "nunique":
        _need_col(df, metric, "metric")
    else:
        _need_numeric(df, metric)


def _agg_scalar(frame: pd.DataFrame, metric: str | None, agg: str):
    if agg == "count":
        return int(len(frame))
    if agg == "nunique":
        return int(frame[metric].nunique())
    if len(frame) == 0:
        return None
    return float(getattr(frame[metric], AGG_FUNCS[agg])())


def _agg_grouped(frame: pd.DataFrame, by, metric: str | None, agg: str) -> pd.Series:
    g = frame.groupby(by, dropna=False)
    if agg == "count":
        return g.size()
    if agg == "nunique":
        return g[metric].nunique()
    return g[metric].agg(AGG_FUNCS[agg])


def _fill_group(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = d.copy()
    for c in cols:
        d[c] = d[c].astype(object).where(d[c].notna(), "(missing)")
    return d


def _pct(new, old):
    if old is None or new is None or pd.isna(old) or pd.isna(new) or old == 0:
        return None
    return round(float((new - old) / abs(old) * 100), 2)


def _round(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def _as_list(x) -> list:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _partial_flags(dt: pd.Series, grain: str) -> tuple[bool, bool]:
    """
    Is the first / last period incomplete? Matters because a half-finished
    month looks like a crash in a SUM. Skipped when data is already at (or
    coarser than) the requested grain, e.g. one row per month.
    """
    if grain == "day":
        return False, False
    freq = GRAIN_FREQ[grain]
    periods = dt.dt.to_period(freq)
    distinct = dt.dt.normalize().groupby(periods).nunique()
    if len(distinct) == 0 or distinct.median() <= 1:
        return False, False
    first_partial = dt.min().normalize() > periods.min().start_time.normalize()
    last_partial = dt.max().normalize() < periods.max().end_time.normalize()
    return bool(first_partial), bool(last_partial)


def _filter_one(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    if not isinstance(f, dict):
        raise ToolError("each filter must be an object: {column, op, value}.")
    col, op, val = f.get("column"), f.get("op", "eq"), f.get("value")
    _need_col(df, col, "filter column")
    s = df[col]

    if op in ("in", "not_in"):
        vals = val if isinstance(val, list) else [val]
        if pd.api.types.is_numeric_dtype(s):
            try:
                mask = s.isin([float(v) for v in vals])
            except (TypeError, ValueError):
                raise ToolError(f"filter values for numeric column '{col}' must be numbers.")
        else:
            mask = s.astype(str).isin([str(v) for v in vals])
        return df[mask] if op == "in" else df[~mask]

    if op == "contains":
        return df[s.astype(str).str.contains(str(val), case=False, na=False, regex=False)]

    ops: dict[str, Callable] = {
        "eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b, "gte": lambda a, b: a >= b,
        "lt": lambda a, b: a < b, "lte": lambda a, b: a <= b,
    }
    if op not in ops:
        raise ToolError(f"unknown filter op '{op}'. Use: eq, ne, gt, gte, lt, lte, in, not_in, contains.")
    if val is None:
        raise ToolError(f"filter on '{col}' needs a value.")

    if pd.api.types.is_datetime64_any_dtype(s) or _infer_kind(s) == "datetime":
        left, right = _dt(df, col), pd.to_datetime(val)
    elif pd.api.types.is_numeric_dtype(s):
        try:
            left, right = s, float(val)
        except (TypeError, ValueError):
            raise ToolError(f"filter value for numeric column '{col}' must be a number, got {val!r}.")
    else:
        left, right = s.astype(str), str(val)
    return df[ops[op](left, right)]


def apply_filters(df: pd.DataFrame, filters: list | None) -> pd.DataFrame:
    for f in filters or []:
        df = _filter_one(df, f)
    return df


# --------------------------------------------------------------------------
# tools (each takes df first, then keyword args from the model)
# --------------------------------------------------------------------------
def describe_columns(df: pd.DataFrame) -> dict:
    cols = []
    for c in column_profile(df)[:80]:
        item = {k: c[k] for k in ("name", "kind", "missing_pct", "unique_count")}
        if c["kind"] == "numeric" and "min" in c:
            item.update(min=c["min"], max=c["max"], mean=round(c["mean"], 4))
        elif c.get("top_values"):
            item["top_values"] = c["top_values"][:3]
        cols.append(item)
    return {
        "n_rows": len(df),
        "columns": cols,
        "date_columns": [c["name"] for c in cols if c["kind"] == "datetime"],
    }


def get_data_quality(df: pd.DataFrame) -> dict:
    ov = dataset_overview(df)
    ov.pop("column_names", None)
    return {"overview": ov, "issues": cleaning_suggestions(df)[:15]}


def sample_rows(df: pd.DataFrame, n: int = 5, filters: list | None = None) -> dict:
    d = apply_filters(df, filters)
    n = max(1, min(int(n), 10))
    out = d.head(n)
    return {"n_matching_rows": len(d), "rows": safe_json(out.iloc[:, :20].to_dict(orient="records"))}


def aggregate(
    df: pd.DataFrame, metric: str | None = None, agg: str = "sum",
    group_by: list | str | None = None, filters: list | None = None,
    order: str = "desc", limit: int = 10,
) -> dict:
    d = apply_filters(df, filters)
    if len(d) == 0:
        return {"error": "No rows match the filters.", "n_rows": 0}
    _check_metric(d, metric, agg)
    group_by = _as_list(group_by)
    for g in group_by:
        _need_col(d, g, "group_by column")

    overall = _agg_scalar(d, metric, agg)
    if not group_by:
        return {"agg": agg, "metric": metric, "value": _round(overall), "n_rows": len(d)}

    d = _fill_group(d, group_by)
    vals = _agg_grouped(d, group_by, metric, agg)
    counts = d.groupby(group_by, dropna=False).size()
    res = pd.DataFrame({"value": vals, "n_rows": counts}).reset_index()
    n_groups = len(res)
    res = res.sort_values("value", ascending=(order == "asc"), na_position="last")
    res = res.head(max(1, min(int(limit or 10), MAX_GROUPS)))

    rows = []
    for _, r in res.iterrows():
        row = {g: r[g] for g in group_by}
        row.update(value=_round(r["value"]), n_rows=int(r["n_rows"]))
        if agg in ADDITIVE and overall:
            row["share_of_total_pct"] = _round(r["value"] / overall * 100, 2)
        elif agg in ("mean", "median") and overall:
            row["vs_overall_pct"] = _pct(r["value"], overall)
        rows.append(row)
    return {
        "agg": agg, "metric": metric, "group_by": group_by,
        "overall_value": _round(overall), "n_rows": len(d),
        "n_groups_total": n_groups, "rows": safe_json(rows),
    }


def time_trend(
    df: pd.DataFrame, date_column: str, metric: str | None = None, agg: str = "sum",
    grain: str = "month", filters: list | None = None,
) -> dict:
    if grain not in GRAIN_FREQ:
        raise ToolError("grain must be one of: day, week, month, quarter, year.")
    d = apply_filters(df, filters)
    _check_metric(d, metric, agg)
    d = d.assign(_dt=_dt(d, date_column)).dropna(subset=["_dt"])
    if len(d) == 0:
        return {"error": "No dated rows after filtering."}

    freq = GRAIN_FREQ[grain]
    d["_p"] = d["_dt"].dt.to_period(freq)
    series = _agg_grouped(d, "_p", metric, agg).sort_index()
    counts = d.groupby("_p").size().reindex(series.index)
    days = d["_dt"].dt.normalize().groupby(d["_p"]).nunique().reindex(series.index)
    first_partial, last_partial = _partial_flags(d["_dt"], grain)

    points, prev = [], None
    for i, (p, v) in enumerate(series.items()):
        partial = (i == 0 and first_partial) or (i == len(series) - 1 and last_partial)
        points.append({
            "period": str(p), "value": _round(v), "n_rows": int(counts[p]),
            "n_days": int(days[p]), "change_pct": _pct(v, prev), "partial": bool(partial),
        })
        prev = v

    complete = [p for p in points if not p["partial"] and p["value"] is not None]
    summary: dict[str, Any] = {
        "n_periods": len(points), "grain": grain,
        "first_period_partial": first_partial, "last_period_partial": last_partial,
    }
    if first_partial or last_partial:
        summary["note"] = "Partial periods are marked partial=true; ignore them when judging drops/rises."
    summary["caution"] = ("Period totals are affected by n_days (e.g. 30 vs 31-day months) and one-off spikes; "
                          "confirm any drop with a segment breakdown before concluding.")
    if len(complete) >= 2:
        vals = np.array([p["value"] for p in complete], dtype=float)
        changes = [(p["change_pct"], p["period"]) for p in complete[1:] if p["change_pct"] is not None]
        summary["pct_change_first_to_last_complete"] = _pct(vals[-1], vals[0])
        summary["peak"] = max(({"period": p["period"], "value": p["value"]} for p in complete), key=lambda x: x["value"])
        summary["trough"] = min(({"period": p["period"], "value": p["value"]} for p in complete), key=lambda x: x["value"])
        if changes:
            drop, rise = min(changes), max(changes)
            summary["biggest_drop"] = {"period": drop[1], "change_pct": drop[0]}
            summary["biggest_rise"] = {"period": rise[1], "change_pct": rise[0]}
        if len(vals) >= 4:
            lr = sps.linregress(np.arange(len(vals)), vals)
            scale = max(abs(vals.mean()), 1e-9)
            rel = lr.slope * (len(vals) - 1) / scale
            summary["trend"] = {
                "direction": "up" if rel > 0.05 else "down" if rel < -0.05 else "flat",
                "slope_per_period": _round(lr.slope), "r_squared": _round(lr.rvalue ** 2, 3),
                "p_value": _round(lr.pvalue, 4),
            }
    return {"metric": metric, "agg": agg, "date_column": date_column,
            "series": points[-60:], "summary": safe_json(summary)}


def _breakdown(dA: pd.DataFrame, dB: pd.DataFrame, col: str, metric, agg: str) -> dict:
    dA, dB = _fill_group(dA, [col]), _fill_group(dB, [col])
    sa, sb = _agg_grouped(dA, col, metric, agg), _agg_grouped(dB, col, metric, agg)
    idx = sa.index.union(sb.index)
    sa, sb = sa.reindex(idx), sb.reindex(idx)
    additive = agg in ADDITIVE
    if additive:
        sa, sb = sa.fillna(0), sb.fillna(0)
    na = dA.groupby(col).size().reindex(idx).fillna(0)
    nb = dB.groupby(col).size().reindex(idx).fillna(0)
    change = sb - sa
    total_change = float(change.sum()) if additive else None
    gross = float(change.abs().sum()) if additive else None

    rows = []
    for seg in change.abs().sort_values(ascending=False, na_position="last").index[:8]:
        row = {
            "segment": seg, "value_a": _round(sa[seg]), "value_b": _round(sb[seg]),
            "change": _round(change[seg]), "pct_change": _pct(sb[seg], sa[seg]),
            "n_a": int(na[seg]), "n_b": int(nb[seg]),
        }
        if additive and gross:
            # bounded 0-100: this segment's movement as a share of ALL movement (up and down)
            row["share_of_gross_movement_pct"] = _round(abs(change[seg]) / gross * 100, 1)
        if additive and total_change:
            # share of the NET change; can exceed 100% when segments offset each other
            row["share_of_net_change_pct"] = _round(change[seg] / total_change * 100, 1)
        rows.append(row)
    out: dict[str, Any] = {"column": col, "segments": rows, "n_segments": len(idx)}
    if additive:
        shown = sum(r["change"] or 0 for r in rows)
        out["other_segments_change"] = _round(total_change - shown)
        out["net_change"], out["gross_movement"] = _round(total_change), _round(gross)
        if total_change and gross > 1.5 * abs(total_change):
            out["offsetting"] = True
            out["note"] = ("Segments move in opposite directions and partly cancel, so share_of_net_change_pct can exceed "
                           "100%. Report share_of_gross_movement_pct (bounded 0-100) or the segment's own pct_change instead.")
    else:
        out["note"] = f"agg='{agg}' is not additive, so contribution shares are not computed; compare segment values directly."
    return out


def compare_periods(
    df: pd.DataFrame, date_column: str, metric: str | None = None, agg: str = "sum",
    period_a_start: str | None = None, period_a_end: str | None = None,
    period_b_start: str | None = None, period_b_end: str | None = None,
    grain: str = "month", breakdown_by: list | str | None = None, filters: list | None = None,
) -> dict:
    """
    Compare baseline period A against period B and (optionally) decompose the
    change by segment -- the core of root-cause analysis. If no dates are
    given, compares the last two COMPLETE periods of `grain`.
    """
    if grain not in GRAIN_FREQ:
        raise ToolError("grain must be one of: day, week, month, quarter, year.")
    d = apply_filters(df, filters)
    _check_metric(d, metric, agg)
    breakdown_by = _as_list(breakdown_by)[:3]
    for c in breakdown_by:
        _need_col(d, c, "breakdown_by column")
    dt = _dt(d, date_column)
    d = d.assign(_dt=dt).dropna(subset=["_dt"])

    if all([period_a_start, period_a_end, period_b_start, period_b_end]):
        a0, b0 = pd.to_datetime(period_a_start), pd.to_datetime(period_b_start)
        a1 = pd.to_datetime(period_a_end).normalize() + pd.Timedelta(days=1)
        b1 = pd.to_datetime(period_b_end).normalize() + pd.Timedelta(days=1)
        dA = d[(d["_dt"] >= a0) & (d["_dt"] < a1)]
        dB = d[(d["_dt"] >= b0) & (d["_dt"] < b1)]
        label_a, label_b = f"{period_a_start}..{period_a_end}", f"{period_b_start}..{period_b_end}"
    else:
        freq = GRAIN_FREQ[grain]
        periods = d["_dt"].dt.to_period(freq)
        first_partial, last_partial = _partial_flags(d["_dt"], grain)
        uniq = sorted(periods.unique())
        if last_partial:
            uniq = uniq[:-1]
        if first_partial:
            uniq = uniq[1:]
        if len(uniq) < 2:
            raise ToolError(f"fewer than two complete {grain} periods; pass explicit period dates.")
        pa, pb = uniq[-2], uniq[-1]
        dA, dB = d[periods == pa], d[periods == pb]
        label_a, label_b = str(pa), str(pb)

    if len(dA) == 0 or len(dB) == 0:
        return {"error": f"One period has no rows (A: {len(dA)} rows, B: {len(dB)} rows). Check the dates."}

    va, vb = _agg_scalar(dA, metric, agg), _agg_scalar(dB, metric, agg)
    out = {
        "metric": metric, "agg": agg,
        "period_a": {"label": label_a, "value": _round(va), "n_rows": len(dA)},
        "period_b": {"label": label_b, "value": _round(vb), "n_rows": len(dB)},
        "change": _round(vb - va), "pct_change": _pct(vb, va),
        "breakdowns": [_breakdown(dA, dB, c, metric, agg) for c in breakdown_by],
    }
    return safe_json(out)


def correlation(
    df: pd.DataFrame, column_a: str, column_b: str | None = None, filters: list | None = None,
) -> dict:
    d = apply_filters(df, filters)
    _need_numeric(d, column_a)
    caution = "Correlation does not establish causation."

    if column_b:
        _need_numeric(d, column_b)
        pair = d[[column_a, column_b]].dropna()
        if len(pair) < 5:
            raise ToolError(f"only {len(pair)} complete rows; need at least 5.")
        if pair[column_a].nunique() < 2 or pair[column_b].nunique() < 2:
            raise ToolError("one of the columns is constant; correlation is undefined.")
        r, p = sps.pearsonr(pair[column_a], pair[column_b])
        rho, _ = sps.spearmanr(pair[column_a], pair[column_b])
        return {"column_a": column_a, "column_b": column_b, "n": len(pair),
                "pearson_r": _round(r), "spearman_rho": _round(rho), "p_value": _round(p, 6),
                "strength": _correlation_strength_label(float(r)), "caution": caution}

    results = []
    for c in d.columns:
        if c == column_a or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        pair = d[[column_a, c]].dropna()
        if len(pair) < 5 or pair[c].nunique() < 2 or pair[column_a].nunique() < 2:
            continue
        r, _ = sps.pearsonr(pair[column_a], pair[c])
        results.append({"column": c, "pearson_r": _round(r), "n": len(pair),
                        "strength": _correlation_strength_label(float(r))})
    results.sort(key=lambda x: abs(x["pearson_r"] or 0), reverse=True)
    return {"column_a": column_a, "top_correlates": results[:8], "caution": caution}


def detect_anomalies(
    df: pd.DataFrame, column: str, method: str = "iqr", date_column: str | None = None,
    grain: str = "month", agg: str = "sum", limit: int = 8, filters: list | None = None,
) -> dict:
    """
    Row-level outliers (IQR or z-score) when no date_column is given;
    otherwise flags unusual PERIODS (sudden spikes/drops) via a robust z-score.
    """
    d = apply_filters(df, filters)
    _need_numeric(d, column)
    limit = max(1, min(int(limit), 15))

    if date_column:
        if grain not in GRAIN_FREQ:
            raise ToolError("grain must be one of: day, week, month, quarter, year.")
        d = d.assign(_dt=_dt(d, date_column)).dropna(subset=["_dt", column])
        d["_p"] = d["_dt"].dt.to_period(GRAIN_FREQ[grain])
        series = _agg_grouped(d, "_p", column, agg).sort_index()
        fp, lp = _partial_flags(d["_dt"], grain)
        keep = series.copy()
        if lp and len(keep) > 1:
            keep = keep.iloc[:-1]
        if fp and len(keep) > 1:
            keep = keep.iloc[1:]
        if len(keep) < 6:
            raise ToolError(f"need at least 6 complete {grain} periods; have {len(keep)}.")
        med = float(keep.median())
        mad = float((keep - med).abs().median())
        if mad > 0:
            z = 0.6745 * (keep - med) / mad
        else:
            sd = float(keep.std()) or 1.0
            z = (keep - med) / sd
        flagged = z[z.abs() > 3.5].sort_values(key=lambda s: -s.abs()).head(limit)
        return {
            "mode": "period", "column": column, "agg": agg, "grain": grain,
            "typical_value": _round(med), "n_periods_checked": len(keep),
            "anomalies": [{"period": str(p), "value": _round(keep[p]), "robust_z": _round(zv, 2),
                           "direction": "spike" if zv > 0 else "drop"} for p, zv in flagged.items()],
            "excluded_partial_periods": bool(fp or lp),
        }

    s = d[column].dropna()
    if len(s) < 10:
        raise ToolError(f"need at least 10 values; have {len(s)}.")
    if method == "zscore":
        sd = s.std()
        if not sd:
            raise ToolError("column is constant; no anomalies.")
        score = (s - s.mean()) / sd
        mask = score.abs() > 3
        lo, hi = float(s.mean() - 3 * sd), float(s.mean() + 3 * sd)
    elif method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        mask = (s < lo) | (s > hi)
        score = (s - s.median()).abs()
    else:
        raise ToolError("method must be 'iqr' or 'zscore'.")

    hits = s[mask]
    top = score[mask].abs().sort_values(ascending=False).head(limit).index
    ctx_cols = [c for c in d.columns if c != column][:3]
    examples = []
    for idx in top:
        ex = {"row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx), "value": _round(s[idx])}
        for c in ctx_cols:
            ex[c] = safe_json(d.loc[idx, c])
        examples.append(ex)
    return {
        "mode": "row", "column": column, "method": method, "n_values": len(s),
        "n_anomalies": int(mask.sum()), "pct_anomalous": _round(mask.mean() * 100, 2),
        "normal_range": [_round(lo), _round(hi)], "examples": examples,
    }


def run_model(df: pd.DataFrame, target: str | None = None) -> dict:
    if target is not None:
        _need_col(df, target, "target column")
    res = run_auto_ml(df, target)
    if res.get("error"):
        return {"error": res["error"]}
    keep = ("problem_type", "target", "best_model", "best_r2", "best_accuracy", "silhouette_score",
            "n_clusters", "summary", "features_dropped")
    out = {k: res[k] for k in keep if k in res}
    out["models_compared"] = res.get("models_compared")
    out["top_features"] = (res.get("feature_importance") or [])[:8]
    if res.get("cluster_summary"):
        out["clusters"] = [{"cluster": c["cluster"], "size": c["size"], "pct": c["pct_of_data"]}
                           for c in res["cluster_summary"]]
    out["caution"] = "Feature importance shows predictive association, not causal effect."
    return out


# --------------------------------------------------------------------------
# registry + JSON schemas (Anthropic tool format)
# --------------------------------------------------------------------------
_FILTERS = {
    "type": "array",
    "description": "Optional row filters, all must match. Example: [{\"column\":\"Region\",\"op\":\"eq\",\"value\":\"South\"}]",
    "items": {
        "type": "object",
        "properties": {
            "column": {"type": "string"},
            "op": {"type": "string", "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains"]},
            "value": {"description": "Scalar, or a list for in/not_in."},
        },
        "required": ["column", "op", "value"],
    },
}
_AGG = {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "std", "count", "nunique"]}
_GRAIN = {"type": "string", "enum": ["day", "week", "month", "quarter", "year"]}


def _schema(name: str, description: str, props: dict, required: list[str] | None = None) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required or []}}


TOOL_FUNCS: dict[str, Callable[..., dict]] = {
    "describe_columns": describe_columns, "get_data_quality": get_data_quality,
    "sample_rows": sample_rows, "aggregate": aggregate, "time_trend": time_trend,
    "compare_periods": compare_periods, "correlation": correlation,
    "detect_anomalies": detect_anomalies, "run_model": run_model,
}

TOOL_SCHEMAS: list[dict] = [
    _schema("describe_columns", "List every column with its type, missing %, unique count and value range / top values. Call this first if unsure of column names.", {}),
    _schema("get_data_quality", "Dataset-level data quality: missing cells, duplicates, outlier flags and suggested fixes.", {}),
    _schema("sample_rows", "Return a few raw rows (max 10), optionally filtered, to see what the data looks like.",
            {"n": {"type": "integer"}, "filters": _FILTERS}),
    _schema("aggregate",
            "Group-by aggregation. Returns value, row count and share-of-total (sum/count) or difference-vs-overall (mean/median) per group. Use for 'which X has the most Y', rankings, totals, averages.",
            {"metric": {"type": "string", "description": "Numeric column (omit for agg=count)."}, "agg": _AGG,
             "group_by": {"type": "array", "items": {"type": "string"}, "description": "Up to 2 columns."},
             "filters": _FILTERS, "order": {"type": "string", "enum": ["asc", "desc"]}, "limit": {"type": "integer"}},
            ["agg"]),
    _schema("time_trend",
            "Metric over time at a chosen grain, with period-over-period change, biggest drop/rise, trend direction, and partial-period flags. Use for 'how did X change', 'when did it drop'.",
            {"date_column": {"type": "string"}, "metric": {"type": "string"}, "agg": _AGG, "grain": _GRAIN, "filters": _FILTERS},
            ["date_column"]),
    _schema("compare_periods",
            "Compare period A (baseline) vs period B and decompose the change by segment (breakdown_by) with each segment's share of the total change. THE tool for 'why did X change'. Omit the four dates to compare the last two complete periods of `grain`.",
            {"date_column": {"type": "string"}, "metric": {"type": "string"}, "agg": _AGG,
             "period_a_start": {"type": "string"}, "period_a_end": {"type": "string"},
             "period_b_start": {"type": "string"}, "period_b_end": {"type": "string"},
             "grain": _GRAIN, "breakdown_by": {"type": "array", "items": {"type": "string"}, "description": "Up to 3 categorical columns to decompose by."},
             "filters": _FILTERS},
            ["date_column"]),
    _schema("correlation",
            "Pearson/Spearman correlation between two numeric columns, or the top correlates of one column if column_b is omitted. Association only, not causation.",
            {"column_a": {"type": "string"}, "column_b": {"type": "string"}, "filters": _FILTERS}, ["column_a"]),
    _schema("detect_anomalies",
            "Find unusual values. Without date_column: row-level outliers (IQR or z-score). With date_column: unusual periods (sudden spikes/drops) using a robust z-score on the aggregated series.",
            {"column": {"type": "string"}, "method": {"type": "string", "enum": ["iqr", "zscore"]},
             "date_column": {"type": "string"}, "grain": _GRAIN, "agg": _AGG, "limit": {"type": "integer"}, "filters": _FILTERS},
            ["column"]),
    _schema("run_model",
            "Train and compare ML models automatically (regression / classification if target given, else clustering). Returns metrics and top predictive features.",
            {"target": {"type": "string", "description": "Column to predict. Omit to cluster."}}),
]


def execute_tool(name: str, args: dict | None, df: pd.DataFrame) -> dict:
    """Run a tool safely; never raises -- errors are returned to the model."""
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'. Available: {list(TOOL_FUNCS)}"}
    try:
        return safe_json(fn(df, **(args or {})))
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001 - tool failures must not crash the agent
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}


def describe_call(name: str, args: dict | None) -> str:
    """Human-readable label for the evidence trail."""
    a = args or {}
    by = a.get("group_by") or a.get("breakdown_by")
    by = f" by {', '.join(_as_list(by))}" if by else ""
    m = a.get("metric") or a.get("column") or a.get("target") or ""
    labels = {
        "describe_columns": "Read column types and ranges",
        "get_data_quality": "Checked data quality",
        "sample_rows": "Looked at sample rows",
        "aggregate": f"Aggregated {a.get('agg', '')} of {m or 'rows'}{by}",
        "time_trend": f"Tracked {m or 'rows'} over time ({a.get('grain', 'month')})",
        "compare_periods": f"Compared periods for {m or 'rows'}{by}",
        "correlation": f"Correlated {a.get('column_a', '')}" + (f" with {a['column_b']}" if a.get("column_b") else " with other columns"),
        "detect_anomalies": f"Scanned {m} for anomalies",
        "run_model": f"Trained models" + (f" to predict {m}" if m else " (clustering)"),
    }
    return labels.get(name, name)
