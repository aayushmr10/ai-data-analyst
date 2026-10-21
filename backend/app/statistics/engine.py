"""
Descriptive statistics + correlation analysis, with plain-English
explanations generated via simple templates (no LLM needed).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from app.utils.serialize import safe_json


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def descriptive_stats(df: pd.DataFrame) -> list[dict]:
    results = []
    for col in numeric_columns(df):
        s = df[col].dropna()
        if len(s) == 0:
            continue
        skew = float(s.skew()) if len(s) > 2 else 0.0
        kurt = float(s.kurtosis()) if len(s) > 3 else 0.0

        skew_label = (
            "roughly symmetric" if abs(skew) < 0.5 else
            "moderately skewed" if abs(skew) < 1 else
            "highly skewed"
        )
        skew_dir = "right (long tail of high values)" if skew > 0 else "left (long tail of low values)"

        results.append({
            "column": col,
            "count": int(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "mode": float(s.mode().iloc[0]) if len(s.mode()) > 0 else None,
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "variance": float(s.var()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "range": float(s.max() - s.min()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "skewness": skew,
            "kurtosis": kurt,
            "summary": f"{skew_label.capitalize()}" + (f", skewed {skew_dir}" if abs(skew) >= 0.5 else "") + ".",
        })
    return safe_json(results)


def _correlation_strength_label(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak / negligible"
    direction = "positive" if r > 0 else "negative" if r < 0 else "no"
    return f"{strength} {direction} relationship" if r != 0 else "no relationship"


def correlation_analysis(df: pd.DataFrame, method: str = "pearson") -> dict:
    cols = numeric_columns(df)
    if len(cols) < 2:
        return safe_json({"matrix": {}, "columns": cols, "top_pairs": [], "note": "Need at least 2 numeric columns."})

    corr = df[cols].corr(method=method)
    matrix = {c: {c2: (None if pd.isna(corr.loc[c, c2]) else round(float(corr.loc[c, c2]), 4)) for c2 in cols} for c in cols}

    pairs = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = corr.loc[c1, c2]
            if pd.isna(r):
                continue
            pairs.append({
                "column_a": c1,
                "column_b": c2,
                "correlation": round(float(r), 4),
                "strength": _correlation_strength_label(float(r)),
                "explanation": f"{c1} and {c2} show a {_correlation_strength_label(float(r))} (r = {r:.2f}).",
            })

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    return safe_json({
        "matrix": matrix,
        "columns": cols,
        "top_pairs": pairs[:15],
        "method": method,
    })
