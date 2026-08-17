# Multi-stage production Dockerfile for Banana Ripeness FastAPI Serving API
FROM python:3.10-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and core serving dependencies
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    torchvision==0.19.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.8.2 \
    pydantic-settings==2.4.0 \
    python-multipart==0.0.9 \
    pillow==10.4.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    httpx==0.27.0

# Production Runtime Stage
FROM python:3.10-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy site-packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Copy application source code
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser data/production_logs/ /app/data/production_logs/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.banana_mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
