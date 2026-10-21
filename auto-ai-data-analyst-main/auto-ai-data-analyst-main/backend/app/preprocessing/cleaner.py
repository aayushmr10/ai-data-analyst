"""
Applies cleaning actions to a dataframe. Actions are the same vocabulary
produced by profiling.cleaning_suggestions(), so the frontend can show
suggestions and let the user click "Apply" per-column or "Apply All".
"""
from __future__ import annotations

import pandas as pd

VALID_ACTIONS = {
    "fill_mean", "fill_median", "fill_mode", "drop_column",
    "drop_duplicates", "drop_rows_with_missing", "flag_outliers",
}


def apply_action(df: pd.DataFrame, action: str, column: str | None) -> tuple[pd.DataFrame, str]:
    """Returns (new_df, log_message). flag_outliers is a no-op (informational only)."""
    df = df.copy()

    if action == "drop_duplicates":
        before = len(df)
        df = df.drop_duplicates()
        return df, f"Dropped {before - len(df)} duplicate rows."

    if column is None or column not in df.columns:
        return df, f"Skipped action '{action}': column not found."

    if action == "drop_column":
        df = df.drop(columns=[column])
        return df, f"Dropped column '{column}'."

    if action == "fill_mean":
        val = df[column].mean()
        df[column] = df[column].fillna(val)
        return df, f"Filled missing values in '{column}' with mean ({val:.3f})."

    if action == "fill_median":
        val = df[column].median()
        df[column] = df[column].fillna(val)
        return df, f"Filled missing values in '{column}' with median ({val:.3f})."

    if action == "fill_mode":
        mode_vals = df[column].mode(dropna=True)
        if len(mode_vals) > 0:
            val = mode_vals.iloc[0]
            df[column] = df[column].fillna(val)
            return df, f"Filled missing values in '{column}' with mode ('{val}')."
        return df, f"No mode found for '{column}'; skipped."

    if action == "drop_rows_with_missing":
        before = len(df)
        df = df.dropna(subset=[column])
        return df, f"Dropped {before - len(df)} rows with missing '{column}'."

    if action == "flag_outliers":
        return df, f"Outliers in '{column}' flagged for review (no rows changed)."

    return df, f"Unknown action '{action}'; skipped."


def apply_actions(df: pd.DataFrame, actions: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    log = []
    for item in actions:
        df, msg = apply_action(df, item.get("action"), item.get("column"))
        log.append(msg)
    return df, log
