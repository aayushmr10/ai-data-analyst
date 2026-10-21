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
