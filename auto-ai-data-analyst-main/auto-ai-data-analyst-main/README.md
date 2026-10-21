---
title: Autonomous Data Analyst
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Upload a dataset, get a full analysis, charts, ML and an AI analyst.
---

# Autonomous Data Analyst

An AI-powered autonomous data analyst that accepts a dataset (CSV / Excel / JSON), profiles it,
suggests and applies cleaning, computes statistics and correlations, generates interactive charts,
auto-selects and trains ML models (regression, classification, or clustering), produces plain-English
recommendations, and exports a professional PDF report — no SQL or notebook required.

All numerical analysis is deterministic (pandas / scikit-learn / scipy). No LLM API key is required to
run this project; recommendations and explanations are generated with rule-based templates driven by
the actual computed statistics.

## Stack

- **Backend:** Python, FastAPI, pandas, scikit-learn, scipy, reportlab, matplotlib
- **Frontend:** React (Vite), Recharts, Axios
- **Infra:** Docker, Docker Compose, Nginx (frontend reverse proxy to backend)

## Project structure

```
autonomous-data-analyst/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + all routes
│   │   ├── store.py             # in-memory dataset session store
│   │   ├── api/schemas.py       # request models
│   │   ├── profiling/           # dataset profiling + cleaning suggestions
│   │   ├── preprocessing/       # cleaning action executor
│   │   ├── statistics/          # descriptive stats + correlation
│   │   ├── visualization/       # chart-ready JSON generators
│   │   ├── ml/                  # auto regression / classification / clustering
│   │   ├── insights/            # rule-based recommendation engine
│   │   ├── reports/             # PDF report generator
│   │   └── utils/               # JSON-safe serialization helpers
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # state + tab routing
│   │   ├── api/client.js        # backend API client
│   │   ├── components/
│   │   │   ├── UploadScreen.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   ├── Primitives.jsx   # design system components
│   │   │   └── tabs/            # Overview, Cleaning, Stats, Charts, Correlation, ML, Recommendations, Report
│   │   └── styles/tokens.css    # design tokens
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Running with Docker (recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000 (docs at http://localhost:8000/docs)

## Running locally without Docker

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (in a second terminal):
```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 — Vite's dev server proxies `/api` calls to `localhost:8000`
(see `frontend/vite.config.js`).

> **Running or deploying?** See [DEPLOY.md](DEPLOY.md) for step-by-step local, Hugging Face Spaces and Vercel + Render instructions.

## AI analyst (optional)

Add an Gemini API key and three tabs light up: **AI Insights** (a briefing plus a root-cause investigator) and **Ask the Analyst** (chat).

```bash
cp .env.example .env      # then set GEMINI_API_KEY
docker compose up --build
```

**The model never does the maths.** It cannot see the raw data; it can only call nine deterministic tools
(`aggregate`, `time_trend`, `compare_periods`, `correlation`, `detect_anomalies`, `run_model`, ...) implemented in
`backend/app/llm/tools.py`. Every figure in an answer must come from a tool result.

| Safeguard | What it does |
|---|---|
| **Evidence trail** | Every answer lists the calculations behind it (tool, arguments, result table). |
| **Grounding check** | Figures in the answer are matched against tool output; untraced ones are flagged. A heuristic, not a proof. |
| **Citation sanitising** | Briefing/root-cause claims must cite real evidence ids; claims without support are shown as *no supporting calculation* and forced to low confidence. |
| **Partial-period + offsetting guards** | Tools flag half-finished months and segments that cancel each other, two classic ways to misread a trend. |
| **Injection hardening** | Cell values/column names are treated as data in the prompt; tools are a fixed allow-list (no code or SQL execution). |
| **Step + size budgets** | Hard cap on tool calls, input length and result size per request. |

Data sent to the provider: the schema and small summaries/samples the model requests through tools (not the full file).
Use a local/approved deployment if your data cannot leave your environment.

### Tests
```bash
cd backend && pip install -r requirements-dev.txt && pytest
```
70 tests: tools are verified against a synthetic dataset with a *planted* ground truth and independent pandas oracles;
the agent loop (error recovery, step budget, sanitising) runs against a scripted client and through the real
`anthropic` SDK with a mocked HTTP layer. They do **not** call the live API, so answer quality is not covered.

## How it works

1. **Upload** a CSV/Excel/JSON file. The backend reads it into pandas and creates an in-memory
   session (`dataset_id`) that every subsequent call references.
2. **Overview** shows row/column counts, per-column types, missing %, and a data preview.
3. **Data Quality** surfaces rule-based cleaning suggestions (fill mean/median/mode, drop duplicates,
   flag outliers via IQR) which you can apply individually or all at once — this mutates the working
   copy of the dataframe (the original stays available via "Reset to original").
4. **Statistics** computes mean/median/std/skew/kurtosis etc. per numeric column with plain-language
   distribution notes.
5. **Visualizations** auto-generates histograms, box plots, bar/pie charts, scatter plots, and time
   series based on what column types are present — rendered live with Recharts.
6. **Correlations** shows a full correlation matrix heatmap plus the strongest relationships explained
   in plain English.
7. **ML Models** — pick a target column (or leave blank for clustering) and the backend auto-detects
   whether the problem is regression or classification, trains two models, picks the best one, and
   shows metrics, a confusion matrix (classification) or feature importance (regression), or cluster
   breakdowns (clustering).
8. **Recommendations** combines profiling, correlation, and ML results into a numbered list of
   plain-English action items.
9. **Report** generates a downloadable PDF containing all of the above sections, built from the current
   (possibly cleaned) state of the dataset.

## Notes on design choices

- **No XGBoost** — scikit-learn's `RandomForestRegressor`/`RandomForestClassifier` fill the
  "robust non-linear default" role without the platform-specific build issues XGBoost sometimes
  brings to a lightweight setup. Swap it in under `backend/app/ml/engine.py` if you want it.
- **Charts render client-side from JSON**, not server-side images, so the live dashboard is fast and
  interactive. The PDF report renders its own matplotlib charts separately since PDFs need rasterized
  images.
- **In-memory dataset store** (`backend/app/store.py`) is intentionally simple — no database. It's
  correct for a single-process demo/local deployment; swap in Redis or Postgres-backed storage if you
  need to survive restarts or scale to multiple backend workers.
- **No LLM dependency** — every explanation and recommendation is generated from actual computed
  statistics via templates, per your requirements. The architecture (see `insights/recommendations.py`)
  is structured so swapping in an LLM call for richer prose later is a contained change.

## Extending this project

Natural next steps, roughly in order of effort:
- Forecasting (Prophet/ARIMA) for datasets with a date column and numeric target
- Anomaly detection (Isolation Forest / LOF) as its own tab
- SHAP-based explainability for ML feature importance
- Chat-with-your-dataset (would reintroduce an LLM dependency, by design not included here)
- Multi-user auth + saved analysis history (would need to replace the in-memory store with Postgres)
