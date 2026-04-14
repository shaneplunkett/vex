# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps in production)
RUN uv sync --no-dev --frozen

# Copy application code
COPY app/ app/
COPY migrations/ migrations/

# --- Production image ---
FROM python:3.12-slim

WORKDIR /app

# Copy installed venv and app from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=builder /app/migrations /app/migrations
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Non-root user for security
RUN adduser --disabled-password --no-create-home --uid 1000 vex
USER vex

# Use the venv's Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default config — overridden by compose/env
ENV VEX_BRAIN_HOST=0.0.0.0
ENV VEX_BRAIN_PORT=8000
ENV VEX_BRAIN_LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); exit(0 if r.status_code == 200 else 1)"

ENTRYPOINT ["python", "-m", "app.main"]
