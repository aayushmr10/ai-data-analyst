"""
Generates a professional PDF report: Executive Summary, Dataset Overview,
Data Quality, Charts, Statistics, Correlations, Model Performance,
Recommendations. Pulls from the same profiling/statistics/ml modules
used by the live dashboard so numbers always match what the user saw.
"""
from __future__ import annotations

import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from app.profiling.profiler import run_full_profile
from app.statistics.engine import descriptive_stats, correlation_analysis
from app.visualization.charts import numeric_columns, categorical_columns

BRAND_COLOR = colors.HexColor("#4F46E5")
LIGHT_BG = colors.HexColor("#F3F4F6")


def _fig_to_image(fig, width=6.2 * inch):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=width * 0.55)


def _make_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(6, 3.3))
    df[col].dropna().hist(bins=20, ax=ax, color="#4F46E5", edgecolor="white")
    ax.set_title(f"Distribution of {col}", fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    return _fig_to_image(fig)


def _make_corr_heatmap(df: pd.DataFrame, cols: list[str]):
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[cols].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Heatmap", fontsize=11)
    fig.tight_layout()
    return _fig_to_image(fig)


def _make_bar(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(6, 3.3))
    counts = df[col].value_counts(dropna=True).head(10)
    counts.plot(kind="bar", ax=ax, color="#4F46E5")
    ax.set_title(f"Top values in {col}", fontsize=11)
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return _fig_to_image(fig)


def generate_pdf_report(
    df: pd.DataFrame,
    filename: str,
    output_path: str,
    ml_result: dict | None = None,
    recommendations: list[str] | None = None,
) -> str:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleBig", parent=styles["Title"], textColor=BRAND_COLOR, fontSize=24)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=BRAND_COLOR, spaceBefore=16, spaceAfter=8)
    body = styles["BodyText"]

    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    # --- Cover / Executive summary ---
    story.append(Paragraph("Autonomous Data Analyst Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Dataset: {filename}", body))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", body))
    story.append(Spacer(1, 12))

    profile = run_full_profile(df)
    ov = profile["overview"]
    story.append(Paragraph("Executive Summary", h2))
    summary_text = (
        f"This dataset contains {ov['n_rows']:,} rows and {ov['n_columns']} columns "
        f"({ov['numeric_columns']} numeric, {ov['categorical_columns']} categorical). "
        f"{ov['missing_pct']}% of all cells are missing and {ov['duplicate_rows']} duplicate rows were detected. "
    )
    if ml_result and ml_result.get("summary"):
        summary_text += ml_result["summary"]
    story.append(Paragraph(summary_text, body))

    # --- Dataset overview table ---
    story.append(Paragraph("Dataset Overview", h2))
    overview_rows = [["Metric", "Value"]] + [
        ["Rows", f"{ov['n_rows']:,}"],
        ["Columns", ov["n_columns"]],
        ["Numeric Columns", ov["numeric_columns"]],
        ["Categorical Columns", ov["categorical_columns"]],
        ["Duplicate Rows", f"{ov['duplicate_rows']} ({ov['duplicate_pct']}%)"],
        ["Missing Cells", f"{ov['missing_cells']} ({ov['missing_pct']}%)"],
        ["Memory Usage", f"{ov['memory_usage_kb']} KB"],
    ]
    t = Table(overview_rows, colWidths=[2.5 * inch, 3.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    # --- Data quality / cleaning suggestions ---
    if profile["cleaning_suggestions"]:
        story.append(Paragraph("Data Quality Report", h2))
        rows = [["Column", "Issue", "Suggestion"]]
        for s in profile["cleaning_suggestions"][:12]:
            rows.append([s["column"] or "(dataset)", s["issue"], s["suggestion"]])
        t = Table(rows, colWidths=[1.3 * inch, 2.2 * inch, 2.5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)

    story.append(PageBreak())

    # --- Charts ---
    story.append(Paragraph("Visualizations", h2))
    num_cols = numeric_columns(df)[:3]
    for col in num_cols:
        story.append(_make_histogram(df, col))
        story.append(Spacer(1, 10))

    cat_cols = [c for c in categorical_columns(df) if df[c].nunique(dropna=True) <= 30][:2]
    for col in cat_cols:
        story.append(_make_bar(df, col))
        story.append(Spacer(1, 10))

    # --- Statistics ---
    story.append(PageBreak())
    story.append(Paragraph("Statistical Analysis", h2))
    stats = descriptive_stats(df)
    if stats:
        rows = [["Column", "Mean", "Median", "Std Dev", "Min", "Max"]]
        for s in stats[:12]:
            rows.append([
                s["column"], f"{s['mean']:.2f}", f"{s['median']:.2f}",
                f"{s['std']:.2f}", f"{s['min']:.2f}", f"{s['max']:.2f}",
            ])
        t = Table(rows, colWidths=[1.3 * inch] * 6)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)

    # --- Correlations ---
    corr = correlation_analysis(df)
    if corr.get("columns") and len(corr["columns"]) >= 2:
        story.append(Spacer(1, 14))
        story.append(Paragraph("Correlation Analysis", h2))
        story.append(_make_corr_heatmap(df, corr["columns"][:10]))
        story.append(Spacer(1, 8))
        for p in corr["top_pairs"][:5]:
            story.append(Paragraph(f"• {p['explanation']}", body))

    # --- Model performance ---
    if ml_result and not ml_result.get("error"):
        story.append(PageBreak())
        story.append(Paragraph("Model Performance", h2))
        story.append(Paragraph(ml_result.get("summary", ""), body))
        story.append(Spacer(1, 8))
        if ml_result.get("models_compared"):
            first = ml_result["models_compared"][0]
            headers = list(first.keys())
            rows = [[h.replace("_", " ").title() for h in headers]] + [
                [str(m[h]) for h in headers] for m in ml_result["models_compared"]
            ]
            t = Table(rows, colWidths=[1.6 * inch] * len(headers))
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t)

        if ml_result.get("feature_importance"):
            story.append(Spacer(1, 12))
            story.append(Paragraph("Top Features", h2))
            for f in ml_result["feature_importance"][:8]:
                story.append(Paragraph(f"• {f['feature']}: {f['importance']}", body))

    # --- Recommendations ---
    if recommendations:
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", h2))
        for r in recommendations:
            story.append(Paragraph(f"• {r}", body))

    doc.build(story)
    return output_path
