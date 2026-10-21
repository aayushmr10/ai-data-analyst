"""
Automatic data profiling.

Given a dataframe, produce:
  - dataset-level overview (rows, cols, memory, duplicates)
  - per-column profile (dtype, missing %, unique count, sample stats)
  - plain-language cleaning suggestions (deterministic, rule-based --
    no LLM required)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from app.utils.serialize import safe_json


def _infer_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # try datetime parse on object columns with low cardinality of formats
    if series.dtype == object:
        non_null = series.dropna()
        if len(non_null) > 0:
            sample = non_null.head(50)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", format=None)
            if parsed.notna().mean() > 0.9:
                return "datetime"
    return "categorical"


def dataset_overview(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    duplicate_rows = int(df.duplicated().sum())
    total_cells = n_rows * n_cols if n_cols else 0
    missing_cells = int(df.isna().sum().sum())

    return safe_json({
        "n_rows": n_rows,
        "n_columns": n_cols,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": n_cols - len(numeric_cols),
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / n_rows * 100, 2) if n_rows else 0,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "column_names": list(df.columns),
    })


def column_profile(df: pd.DataFrame) -> list[dict]:
    profiles = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        kind = _infer_kind(series)
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        entry = {
            "name": col,
            "dtype": str(series.dtype),
            "kind": kind,
            "missing_count": missing,
            "missing_pct": round(missing / n_rows * 100, 2) if n_rows else 0,
            "unique_count": unique,
            "unique_pct": round(unique / n_rows * 100, 2) if n_rows else 0,
        }

        if kind == "numeric":
            non_null = series.dropna()
            if len(non_null) > 0:
                entry.update({
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                })
        elif kind in ("categorical", "boolean"):
            top = series.value_counts(dropna=True).head(5)
            entry["top_values"] = [
                {"value": str(idx), "count": int(cnt)} for idx, cnt in top.items()
            ]

        profiles.append(entry)

    return safe_json(profiles)


def cleaning_suggestions(df: pd.DataFrame) -> list[dict]:
    """
    Deterministic, rule-based cleaning suggestions -- mirrors the
    'Column CustomerAge: 12% missing -> fill using median' style from
    the spec, without needing an LLM.
    """
    suggestions = []
    n_rows = len(df)
    if n_rows == 0:
        return suggestions

    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        missing_pct = missing / n_rows * 100
        kind = _infer_kind(series)

        if missing_pct == 0:
            continue

        if missing_pct > 60:
            suggestions.append({
                "column": col,
                "issue": f"{missing_pct:.1f}% missing values",
                "suggestion": "Consider dropping this column -- too sparse to be reliable.",
                "action": "drop_column",
                "severity": "high",
            })
        elif kind == "numeric":
            skew = series.dropna().skew() if series.dropna().shape[0] > 2 else 0
            method = "median" if abs(skew) > 1 else "mean"
            suggestions.append({
                "column": col,
                "issue": f"{missing_pct:.1f}% missing values",
                "suggestion": f"Fill using {method} (distribution is {'skewed' if abs(skew) > 1 else 'roughly symmetric'}).",
                "action": f"fill_{method}",
                "severity": "medium" if missing_pct > 20 else "low",
            })
        else:
            suggestions.append({
                "column": col,
                "issue": f"{missing_pct:.1f}% missing values",
                "suggestion": "Fill using mode (most frequent value).",
                "action": "fill_mode",
                "severity": "medium" if missing_pct > 20 else "low",
            })

    # duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        suggestions.append({
            "column": None,
            "issue": f"{dup_count} duplicate rows detected",
            "suggestion": "Remove duplicate rows to avoid skewing statistics and model training.",
            "action": "drop_duplicates",
            "severity": "medium",
        })

    # outlier flag for numeric columns (IQR method)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        outlier_pct = outliers / len(s) * 100
        if outlier_pct > 3:
            suggestions.append({
                "column": col,
                "issue": f"{outliers} potential outliers ({outlier_pct:.1f}% of values, IQR method)",
                "suggestion": "Review these values -- cap (winsorize) or investigate as data entry errors.",
                "action": "flag_outliers",
                "severity": "low",
            })

    return safe_json(suggestions)


def run_full_profile(df: pd.DataFrame) -> dict:
    return {
        "overview": dataset_overview(df),
        "columns": column_profile(df),
        "cleaning_suggestions": cleaning_suggestions(df),
    }
