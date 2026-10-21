"""
Instead of rendering static PNGs server-side, we compute chart-ready
JSON (bins, buckets, series) and let the React frontend render
interactive charts with Recharts. This keeps everything fast, avoids a
matplotlib dependency for the live dashboard, and is far nicer UX.

Static image export (for the PDF/DOCX report) is handled separately in
app/reports using matplotlib, since PDFs need rasterized charts.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from app.statistics.engine import numeric_columns
from app.utils.serialize import safe_json


def categorical_columns(df: pd.DataFrame) -> list[str]:
    numeric = set(numeric_columns(df))
    return [c for c in df.columns if c not in numeric]


def histogram_data(df: pd.DataFrame, column: str, bins: int = 20) -> dict:
    s = df[column].dropna()
    if len(s) == 0:
        return {"column": column, "bins": []}
    counts, edges = np.histogram(s, bins=bins)
    data = [
        {"bin": f"{edges[i]:.2f}–{edges[i+1]:.2f}", "count": int(counts[i]), "bin_start": float(edges[i])}
        for i in range(len(counts))
    ]
    return safe_json({"column": column, "bins": data})


def boxplot_data(df: pd.DataFrame, column: str) -> dict:
    s = df[column].dropna()
    if len(s) == 0:
        return {"column": column}
    q1, median, q3 = s.quantile(0.25), s.quantile(0.5), s.quantile(0.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    whisker_low = s[s >= lower_fence].min()
    whisker_high = s[s <= upper_fence].max()
    outliers = s[(s < lower_fence) | (s > upper_fence)].tolist()
    return safe_json({
        "column": column,
        "min": float(s.min()), "q1": float(q1), "median": float(median),
        "q3": float(q3), "max": float(s.max()),
        "whisker_low": float(whisker_low), "whisker_high": float(whisker_high),
        "outliers": outliers[:50],
    })


def bar_chart_data(df: pd.DataFrame, column: str, top_n: int = 15) -> dict:
    counts = df[column].value_counts(dropna=True).head(top_n)
    data = [{"category": str(idx), "count": int(cnt)} for idx, cnt in counts.items()]
    return safe_json({"column": column, "data": data})


def pie_chart_data(df: pd.DataFrame, column: str, top_n: int = 8) -> dict:
    counts = df[column].value_counts(dropna=True)
    top = counts.head(top_n)
    data = [{"name": str(idx), "value": int(cnt)} for idx, cnt in top.items()]
    if len(counts) > top_n:
        data.append({"name": "Other", "value": int(counts.iloc[top_n:].sum())})
    return safe_json({"column": column, "data": data})


def scatter_data(df: pd.DataFrame, x: str, y: str, max_points: int = 1000) -> dict:
    sub = df[[x, y]].dropna()
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)
    data = [{"x": float(row[x]), "y": float(row[y])} for _, row in sub.iterrows()]
    return safe_json({"x": x, "y": y, "data": data})


def timeseries_data(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    sub = df[[date_col, value_col]].dropna().copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)
    # aggregate by day/month depending on span
    span_days = (sub[date_col].max() - sub[date_col].min()).days if len(sub) > 1 else 0
    freq = "M" if span_days > 120 else "D"
    grouped = sub.set_index(date_col)[value_col].resample(freq).mean().dropna()
    data = [{"date": idx.strftime("%Y-%m-%d"), "value": float(val)} for idx, val in grouped.items()]
    return safe_json({"date_column": date_col, "value_column": value_col, "frequency": freq, "data": data})


def auto_chart_suggestions(df: pd.DataFrame) -> dict:
    """Suggests which charts are worth generating given this dataset's shape."""
    num_cols = numeric_columns(df)
    cat_cols = [c for c in categorical_columns(df) if df[c].nunique(dropna=True) <= 30]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not date_cols:
        for c in df.columns:
            if c in num_cols:
                continue
            sample = df[c].dropna().head(30)
            if len(sample) and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9:
                date_cols.append(c)

    return safe_json({
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "date_columns": date_cols,
        "suggested": {
            "histograms": num_cols[:6],
            "boxplots": num_cols[:6],
            "bar_charts": cat_cols[:4],
            "pie_charts": [c for c in cat_cols if df[c].nunique(dropna=True) <= 8][:3],
            "scatter_pairs": [[num_cols[i], num_cols[j]] for i in range(len(num_cols)) for j in range(i+1, len(num_cols))][:5],
            "timeseries": [{"date": date_cols[0], "value": num_cols[0]}] if date_cols and num_cols else [],
        },
    })
