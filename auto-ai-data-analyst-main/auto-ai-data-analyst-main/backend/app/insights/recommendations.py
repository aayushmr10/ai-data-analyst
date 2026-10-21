"""
Deterministic, rule-based recommendation generator. Looks at profiling,
correlation, and ML results and produces plain-English action items --
covers the "Recommendations" feature from the spec without needing an
LLM API key.
"""
from __future__ import annotations

import pandas as pd

from app.profiling.profiler import run_full_profile
from app.statistics.engine import correlation_analysis


def generate_recommendations(df: pd.DataFrame, ml_result: dict | None = None) -> list[str]:
    recs: list[str] = []
    profile = run_full_profile(df)
    ov = profile["overview"]

    if ov["missing_pct"] > 10:
        recs.append(
            f"{ov['missing_pct']}% of cells are missing across the dataset -- "
            "apply the suggested cleaning actions before drawing firm conclusions from statistics or models."
        )

    if ov["duplicate_rows"] > 0:
        recs.append(
            f"{ov['duplicate_rows']} duplicate rows were found -- removing them will prevent "
            "double-counting in aggregate statistics and biased model training."
        )

    high_sev = [s for s in profile["cleaning_suggestions"] if s.get("severity") == "high"]
    for s in high_sev[:3]:
        recs.append(f"Column '{s['column']}' is {s['issue']} -- {s['suggestion'].lower()}")

    corr = correlation_analysis(df)
    for pair in corr.get("top_pairs", [])[:3]:
        if abs(pair["correlation"]) >= 0.6:
            direction = "increasing" if pair["correlation"] > 0 else "decreasing"
            recs.append(
                f"'{pair['column_a']}' and '{pair['column_b']}' are strongly related (r = {pair['correlation']}) -- "
                f"{direction} one tends to move with the other, worth investigating for a causal or business link."
            )

    if ml_result and not ml_result.get("error"):
        ptype = ml_result.get("problem_type")
        if ptype == "regression":
            r2 = ml_result.get("best_r2", 0)
            if r2 < 0.4:
                recs.append(
                    f"The best regression model only explains {r2*100:.0f}% of variance in "
                    f"'{ml_result.get('target')}' -- consider collecting additional features, as current columns "
                    "may not capture the main drivers."
                )
            top_feat = (ml_result.get("feature_importance") or [{}])[0].get("feature")
            if top_feat:
                recs.append(f"'{top_feat}' is the strongest driver of '{ml_result.get('target')}' -- prioritize it in planning and forecasting.")
        elif ptype == "classification":
            acc = ml_result.get("best_accuracy", 0)
            if acc < 0.7:
                recs.append(
                    f"Classification accuracy is {acc*100:.0f}% -- treat predictions as directional, "
                    "not definitive, until more data or features are added."
                )
            top_feat = (ml_result.get("feature_importance") or [{}])[0].get("feature")
            if top_feat:
                recs.append(f"'{top_feat}' most strongly predicts the outcome -- consider monitoring or acting on it directly.")
        elif ptype == "clustering":
            n = ml_result.get("n_clusters")
            recs.append(
                f"Data naturally separates into {n} distinct groups -- consider tailoring strategy "
                "(e.g. marketing, pricing, outreach) separately per group rather than treating the dataset as one segment."
            )

    if not recs:
        recs.append("No major data quality or statistical issues detected -- dataset looks analysis-ready.")

    return recs
