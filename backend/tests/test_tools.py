import json

import pandas as pd
import pytest

from app.llm.tools import TOOL_SCHEMAS, TOOL_FUNCS, describe_call, execute_tool


def run(name, df, **kw):
    return execute_tool(name, kw, df)


# ---------- aggregate ----------
def test_aggregate_shares_sum_to_100(sales_df):
    out = run("aggregate", sales_df, metric="Revenue", agg="sum", group_by=["Region"])
    assert out["n_groups_total"] == 4
    assert sum(r["share_of_total_pct"] for r in out["rows"]) == pytest.approx(100, abs=0.1)


def test_aggregate_filters_and_total(sales_df):
    out = run("aggregate", sales_df, metric="Revenue", agg="sum",
              filters=[{"column": "Region", "op": "eq", "value": "South"}])
    expected = sales_df.loc[sales_df.Region == "South", "Revenue"].sum()
    assert out["value"] == pytest.approx(expected, rel=1e-6)


def test_aggregate_errors_are_actionable(sales_df):
    out = run("aggregate", sales_df, metric="Nope", agg="sum")
    assert "not found" in out["error"] and "Revenue" in out["error"]  # lists valid columns
    out = run("aggregate", sales_df, metric="Region", agg="sum")
    assert "not numeric" in out["error"]


# ---------- time_trend ----------
def test_time_trend_flags_partial_september(sales_df):
    out = run("time_trend", sales_df, date_column="Date", metric="Revenue", agg="sum", grain="month")
    assert out["summary"]["last_period_partial"] is True
    assert out["series"][-1]["period"] == "2024-09" and out["series"][-1]["partial"] is True
    # the partial month must NOT be reported as the biggest drop
    assert out["summary"]["biggest_drop"]["period"] != "2024-09"


def test_time_trend_matches_independent_pandas_oracle(sales_df):
    """Tool output must equal a from-scratch pandas calculation (excluding partial Sept)."""
    out = run("time_trend", sales_df, date_column="Date", metric="Revenue", agg="sum", grain="month")
    monthly = sales_df.groupby(sales_df.Date.dt.to_period("M"))["Revenue"].sum()
    complete = monthly.iloc[:-1]                      # September is partial
    pct = complete.pct_change().dropna() * 100
    assert out["summary"]["biggest_drop"]["period"] == str(pct.idxmin())
    assert out["summary"]["biggest_drop"]["change_pct"] == pytest.approx(pct.min(), abs=0.01)
    assert out["summary"]["peak"]["period"] == str(complete.idxmax())


def test_headline_trend_can_mislead_but_segment_view_does_not(sales_df):
    """The planted South collapse is NOT the biggest monthly drop in the totals
    (a May spike + month length dominate) -- which is exactly why the agent must
    decompose by segment instead of trusting the headline series."""
    trend = run("time_trend", sales_df, date_column="Date", metric="Revenue", agg="sum", grain="month")
    assert trend["summary"]["biggest_drop"]["period"] != "2024-07"
    assert "n_days" in trend["series"][0] and "caution" in trend["summary"]
    seg = run("compare_periods", sales_df, date_column="Date", metric="Revenue", agg="sum",
              period_a_start="2024-06-01", period_a_end="2024-06-30",
              period_b_start="2024-07-01", period_b_end="2024-07-31", breakdown_by=["Region"])
    assert seg["breakdowns"][0]["segments"][0]["segment"] == "South"


def test_time_trend_no_false_partial_for_monthly_rows():
    df = pd.DataFrame({"d": pd.date_range("2023-01-01", periods=12, freq="MS"), "v": range(12)})
    out = run("time_trend", df, date_column="d", metric="v", agg="sum", grain="month")
    assert out["summary"]["last_period_partial"] is False


# ---------- compare_periods (root cause) ----------
def test_compare_periods_finds_south_as_root_cause(sales_df):
    out = run("compare_periods", sales_df, date_column="Date", metric="Revenue", agg="sum",
              period_a_start="2024-06-01", period_a_end="2024-06-30",
              period_b_start="2024-07-01", period_b_end="2024-07-31",
              breakdown_by=["Region", "Channel"])
    region = next(b for b in out["breakdowns"] if b["column"] == "Region")
    top = region["segments"][0]
    assert top["segment"] == "South"
    assert top["pct_change"] == pytest.approx(-40, abs=6)

    # oracle: recompute the decomposition independently in pandas
    jun = sales_df[(sales_df.Date >= "2024-06-01") & (sales_df.Date <= "2024-06-30")].groupby("Region").Revenue.sum()
    jul = sales_df[(sales_df.Date >= "2024-07-01") & (sales_df.Date <= "2024-07-31")].groupby("Region").Revenue.sum()
    chg = jul - jun
    assert top["change"] == pytest.approx(chg["South"], abs=0.01)
    assert top["share_of_gross_movement_pct"] == pytest.approx(abs(chg["South"]) / chg.abs().sum() * 100, abs=0.1)
    assert top["share_of_net_change_pct"] == pytest.approx(chg["South"] / chg.sum() * 100, abs=0.1)


def test_gross_share_is_bounded_and_offsetting_is_flagged(sales_df):
    """Regression test: net-share hit 514% when other regions rose. Gross share must stay in [0,100]."""
    out = run("compare_periods", sales_df, date_column="Date", metric="Revenue", agg="sum",
              period_a_start="2024-06-01", period_a_end="2024-06-30",
              period_b_start="2024-07-01", period_b_end="2024-07-31", breakdown_by=["Region"])
    region = out["breakdowns"][0]
    shares = [s["share_of_gross_movement_pct"] for s in region["segments"]]
    assert all(0 <= v <= 100 for v in shares) and sum(shares) == pytest.approx(100, abs=0.5)
    assert region["segments"][0]["share_of_net_change_pct"] > 100          # the misleading number...
    assert region["offsetting"] is True and "share_of_gross_movement_pct" in region["note"]   # ...is flagged


def test_compare_periods_auto_skips_partial_month(sales_df):
    out = run("compare_periods", sales_df, date_column="Date", metric="Revenue", agg="sum", grain="month")
    # last COMPLETE month is August, not the partial September
    assert out["period_b"]["label"] == "2024-08" and out["period_a"]["label"] == "2024-07"


def test_compare_periods_non_additive_agg_has_note(sales_df):
    out = run("compare_periods", sales_df, date_column="Date", metric="Revenue", agg="mean",
              period_a_start="2024-06-01", period_a_end="2024-06-30",
              period_b_start="2024-07-01", period_b_end="2024-07-31", breakdown_by=["Region"])
    assert "not additive" in out["breakdowns"][0]["note"]


# ---------- correlation ----------
def test_correlation_adspend_revenue_strong(sales_df):
    out = run("correlation", sales_df, column_a="AdSpend", column_b="Revenue")
    import numpy as np
    expected = np.corrcoef(sales_df["AdSpend"], sales_df["Revenue"])[0, 1]
    assert out["pearson_r"] == pytest.approx(expected, abs=1e-4)      # oracle, not a magic threshold
    assert "strong" in out["strength"] and "causation" in out["caution"]


def test_correlation_top_correlates(sales_df):
    out = run("correlation", sales_df, column_a="Revenue")
    assert out["top_correlates"][0]["column"] == "AdSpend"


# ---------- anomalies ----------
def test_row_anomalies_find_planted_spike(sales_df):
    out = run("detect_anomalies", sales_df, column="Revenue", method="iqr")
    assert out["n_anomalies"] >= 1
    assert out["examples"][0]["value"] == pytest.approx(sales_df["Revenue"].max(), rel=1e-6)


def test_period_anomalies_runs_and_excludes_partial(sales_df):
    out = run("detect_anomalies", sales_df, column="Revenue", date_column="Date", grain="week", agg="sum")
    assert out["mode"] == "period" and out["excluded_partial_periods"] is True


# ---------- model / quality / describe ----------
def test_run_model_regression(sales_df):
    out = run("run_model", sales_df[["AdSpend", "Revenue"]].head(800), target="Revenue")
    assert out["problem_type"] == "regression" and out["top_features"][0]["feature"] == "AdSpend"


def test_describe_and_quality(sales_df):
    d = run("describe_columns", sales_df)
    assert "Date" in d["date_columns"] and d["n_rows"] == len(sales_df)
    assert "overview" in run("get_data_quality", sales_df)


# ---------- safety / contract ----------
def test_unknown_tool_and_bad_args_never_raise(sales_df):
    assert "Unknown tool" in execute_tool("drop_table", {}, sales_df)["error"]
    assert "Invalid arguments" in execute_tool("aggregate", {"bogus": 1, "agg": "sum"}, sales_df)["error"]


def test_all_outputs_are_json_serialisable(sales_df):
    calls = [("aggregate", dict(metric="Revenue", group_by=["Region"], agg="mean")),
             ("time_trend", dict(date_column="Date", metric="Revenue")),
             ("compare_periods", dict(date_column="Date", metric="Revenue", breakdown_by=["Region"])),
             ("detect_anomalies", dict(column="Revenue")), ("sample_rows", dict(n=3))]
    for name, kw in calls:
        json.dumps(execute_tool(name, kw, sales_df))  # raises on NaN-objects / numpy types / Periods


def test_every_schema_has_a_function_and_valid_shape():
    assert {s["name"] for s in TOOL_SCHEMAS} == set(TOOL_FUNCS)
    for s in TOOL_SCHEMAS:
        assert s["input_schema"]["type"] == "object" and s["description"]


def test_describe_call_labels():
    assert "South" not in describe_call("aggregate", {"metric": "Revenue", "agg": "sum", "group_by": ["Region"]})
    assert "by Region" in describe_call("aggregate", {"metric": "Revenue", "agg": "sum", "group_by": ["Region"]})
