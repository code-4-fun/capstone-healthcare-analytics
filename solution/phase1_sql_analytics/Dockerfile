# Phase 1 DB bootstrap - a one-shot container that applies the Phase 1 DDL
# (schema, tables, typed load, indexes, views, data-quality checks) to a fresh
# Postgres so Phase 2/3 can run against it immediately.
#
# Build context is the solution/ root (so src/capstone is in scope):
#   docker compose up bootstrap --build
# or just `docker compose up --build`, which brings up postgres + bootstrap.

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

# 2. the package + the Phase 1 SQL/load scripts (the raw CSVs are bind-mounted
#    at runtime via docker-compose.yml, not baked into the image)
COPY src/ ./src/
COPY phase1_sql_analytics/sql/ ./phase1_sql_analytics/sql/
COPY phase1_sql_analytics/load_data.py phase1_sql_analytics/bootstrap_db.py ./phase1_sql_analytics/
RUN uv sync --frozen --no-dev

# Postgres detail + data location come from the environment (compose sets them).
CMD ["python", "phase1_sql_analytics/bootstrap_db.py"]
