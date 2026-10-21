# Single-container build: React frontend + FastAPI backend on ONE port (7860).
# Used by Hugging Face Spaces (Docker SDK). For local multi-container dev use docker-compose.yml instead.

# ---------- Stage 1: build the React app ----------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python runtime serving the API and the built frontend ----------
FROM python:3.11-slim

# libs needed by matplotlib / reportlab
RUN apt-get update && apt-get install -y --no-install-recommends libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STATIC_DIR=/home/user/app/static \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user backend/app ./app
COPY --chown=user --from=web /web/dist ./static
RUN mkdir -p reports_output

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health')" || exit 1
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
