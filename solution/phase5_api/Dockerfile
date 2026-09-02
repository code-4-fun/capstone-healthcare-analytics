# Phase 5 API - serves the persisted Phase 3 models behind FastAPI.
#
# Build context is the solution/ root (so src/capstone and the model artefacts
# are in scope):
#   docker build -f phase5_api/Dockerfile -t capstone-api .
# or just `docker compose -f phase5_api/docker-compose.yml up --build`.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# uv for reproducible, locked installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 1. dependencies (cached until the lock changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. the package + the artefacts the API loads at startup
COPY src/ ./src/
COPY phase3_models/models/ ./phase3_models/models/
COPY phase2_eda/output/feature_frame.parquet ./phase2_eda/output/feature_frame.parquet
COPY phase5_api/ ./phase5_api/
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Postgres detail comes from the environment (compose sets PGHOST=postgres).
ENV API_MODELS_DIR=/app/phase3_models/models \
    API_SERVING_CONFIG=/app/phase5_api/serving_config.json

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "uvicorn", "app.main:app", "--app-dir", "phase5_api", "--host", "0.0.0.0", "--port", "8000"]
