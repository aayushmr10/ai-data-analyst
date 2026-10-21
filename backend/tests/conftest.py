"""
Shared fixture: a synthetic sales dataset with a KNOWN story baked in, so tests
can assert the tools find the truth rather than just "return something".

Planted facts:
  * Daily data 2024-01-01 .. 2024-09-27  (September is PARTIAL)
  * From 2024-07-01, South revenue is cut by 40%  -> the true root cause
  * 2024-05-15: one North row is 8x normal      -> a planted spike
  * AdSpend drives Revenue (strong positive correlation)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def sales_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", "2024-09-27", freq="D")
    rows = []
    for d in dates:
        for region in ["North", "South", "East", "West"]:
            ad = rng.uniform(100, 1000)
            rev = 500 + 2.0 * ad + rng.normal(0, 60)
            if region == "South" and d >= pd.Timestamp("2024-07-01"):
                rev *= 0.6
            rows.append({"Date": d, "Region": region, "Channel": rng.choice(["Web", "Store"]),
                         "AdSpend": round(ad, 2), "Revenue": round(rev, 2)})
    df = pd.DataFrame(rows)
    spike = df[(df["Date"] == "2024-05-15") & (df["Region"] == "North")].index[0]
    df.loc[spike, "Revenue"] *= 8
    return df


@pytest.fixture(autouse=True)
def _fresh_limiter():
    """Rate-limit state is process-global; isolate every test."""
    from app import guards
    guards.limiter.reset()
    yield
    guards.limiter.reset()
