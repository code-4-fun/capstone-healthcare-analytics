# Phase 6 scheduler sidecar - seeds the demo traffic, then runs the drift job
# on an interval, writing capstone_solution.drift_report.
#
# Build context is the solution/ root (src/capstone + the model artefacts):
#   docker compose -f phase6_monitoring/docker-compose.yml up --build

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 1. dependencies (cached until the lock changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. the package + the artefacts the drift job loads
COPY src/ ./src/
COPY phase3_models/models/ ./phase3_models/models/
COPY phase2_eda/output/feature_frame.parquet ./phase2_eda/output/feature_frame.parquet
COPY phase5_api/serving_config.json ./phase5_api/serving_config.json
COPY phase5_api/sql/prediction_log.sql ./phase5_api/sql/prediction_log.sql
COPY phase6_monitoring/ ./phase6_monitoring/
RUN uv sync --frozen --no-dev

# Postgres detail + cadence come from the environment (compose sets them).
ENV DRIFT_INTERVAL_SECONDS=21600 \
    DRIFT_WINDOW=drift \
    SEED_ON_START=true

CMD ["python", "-m", "phase6_monitoring", "scheduler"]
