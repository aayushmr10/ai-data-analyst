# Run & deploy — step by step

## 1. Run it locally

### Option A — Docker Compose (two containers, closest to a normal production layout)
```bash
cp .env.example .env          # optional: add GEMINI_API_KEY to enable the AI analyst
docker compose up --build
```
Open http://localhost:3000.

### Option B — without Docker (best for development)
```bash
# terminal 1 — backend
cd backend
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_gemini_api_key                        # optional; Windows: set GEMINI_API_KEY=...
uvicorn app.main:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 (Vite proxies `/api` to port 8000).
Note: only Docker Compose reads `.env`; in this mode, set variables in your shell as shown.

### Option C — test the exact image you will deploy (single container)
```bash
docker build -t data-analyst .
docker run --rm -p 7860:7860 --env-file .env data-analyst
```
Open http://localhost:7860 — run this once before deploying, so any build problem shows up on your machine
rather than in the hosting build log.

### Run the tests
```bash
cd backend && pip install -r requirements-dev.txt && pytest
```

---

## 2. Deploy free on Hugging Face Spaces (recommended)

Why: the free tier has 2 CPU cores and 16 GB RAM, which suits pandas + scikit-learn, and one container serves
both the frontend and the API on one URL (no CORS setup). Free Spaces go to sleep after a long idle period
and wake on the next visit; uploaded datasets live in memory only, so they are lost when the Space restarts.

1. **Create an account** at huggingface.co.
2. **Create the Space:** *New → Space*. Pick a name, choose **Docker** as the SDK (Blank template),
   hardware **CPU basic (free)**, then *Create Space*.
3. **Add secrets** (Space → *Settings → Variables and secrets*). Use **Secrets**, not Variables, for the first two,
   because variables on a public Space are visible to everyone:

   | Name | Type | Value |
   |---|---|---|
   | `GEMINI_API_KEY` | Secret | your key (skip to run without AI) |
   | `APP_ACCESS_CODE` | Secret | a code you share only with intended users |
   | `AI_DAILY_CAP` | Variable | e.g. `100` — hard ceiling on AI requests per day |
   | `AI_RATE_LIMIT_PER_HOUR` | Variable | e.g. `20` |
   | `MAX_UPLOAD_MB` | Variable | e.g. `25` |

4. **Create a write token:** your profile → *Settings → Access Tokens → New token* with **Write** permission.
   Git will ask for it instead of a password.
5. **Push the code** (the Space is a git repository; `README.md` already contains the Space config header):
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   cp -r /path/to/autonomous-data-analyst/. .      # copy the project in, including the hidden files
   git add . && git commit -m "Deploy Autonomous Data Analyst"
   git push        # username = your HF username, password = the write token
   ```
6. **Watch the build:** open the Space → *Logs*. The first build takes several minutes (it compiles the frontend,
   then installs the Python packages). When the status turns *Running*, the app appears on the Space page.
7. **Verify:** add `/api/health` to the app's direct URL — it should return `{"status":"ok"}`. Upload a CSV,
   then open *AI Insights*: it should ask for your access code.

**Before you share the link**
- In the Anthropic console, set a monthly **spend limit** on the key. `AI_DAILY_CAP` is your in-app ceiling,
  the console limit is the backstop.
- Add a line telling users not to upload sensitive data: files are held in server memory, and the AI features send
  summaries and samples to the model provider.
- To pause AI entirely, delete the `GEMINI_API_KEY` secret — everything else keeps working.

### Updating
Commit and `git push` again; the Space rebuilds automatically.

---

## 3. Alternative: Vercel (frontend) + Render (backend)

Works, with trade-offs: Render's free web service has 0.1 CPU and 512 MB RAM, spins down after 15 idle minutes
(about a minute to wake up), and loses all in-memory datasets whenever it spins down.
Vercel is only suitable for the frontend (its serverless functions are stateless and cap request/response bodies at 4.5 MB).

1. **Backend on Render:** *New → Web Service* → connect your repo → **Root Directory** `backend`,
   runtime **Docker**, instance type **Free**, health check path `/api/health`. Add environment variables:
   `GEMINI_API_KEY`, `APP_ACCESS_CODE`, `MAX_UPLOAD_MB=5` (512 MB RAM is tight), and — after step 2 —
   `CORS_ORIGINS=https://<your-app>.vercel.app`. Note the service URL (`https://<name>.onrender.com`).
2. **Frontend on Vercel:** *Add New → Project* → import the repo → **Root Directory** `frontend` (Vite is
   auto-detected) → add environment variable `VITE_API_URL=https://<name>.onrender.com` → *Deploy*.
3. Go back to Render, set `CORS_ORIGINS` to your Vercel URL, and let it redeploy.
4. Expect the first request after idle to take about a minute. The app shows a "session expired — upload again"
   message if Render restarted in between.

---

## 4. Environment variables

See `.env.example` for the full list with defaults. The important ones for a public site:
`APP_ACCESS_CODE`, `AI_DAILY_CAP`, `AI_RATE_LIMIT_PER_HOUR`, `MAX_UPLOAD_MB`, `SESSION_TTL_MIN`, `MAX_SESSIONS`.

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Space stuck on *Building* or *Build error* | Open *Logs → Build*. A failure in the `npm ci` or `pip install` step names the package. |
| App loads, AI tabs say "Connect the AI analyst" | `GEMINI_API_KEY` missing or the Space wasn't restarted after adding it. |
| AI tabs keep asking for the access code | The code doesn't match `APP_ACCESS_CODE` exactly (case-sensitive). |
| "AI request limit reached" / "daily AI usage limit" | Working as designed — raise `AI_RATE_LIMIT_PER_HOUR` / `AI_DAILY_CAP` if appropriate. |
| "Your session expired" | The server restarted or the dataset was idle past `SESSION_TTL_MIN`; upload the file again. |
| "File is larger than the N MB limit" | Raise `MAX_UPLOAD_MB` (mind the host's RAM). |
| Browser shows CORS errors (split deployment only) | `CORS_ORIGINS` must exactly match the frontend's origin, with no trailing slash. |
