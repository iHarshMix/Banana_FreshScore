# Production Dockerfile for FreshScore Streamlit Frontend Dashboard
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8501

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit==1.38.0 \
    plotly==5.24.0 \
    httpx==0.27.0 \
    pillow==10.4.0 \
    pandas==2.2.2 \
    numpy==1.26.4

# Create unprivileged user
RUN useradd -m -u 1000 -s /bin/bash appuser

# Copy dashboard and necessary code
COPY --chown=appuser:appuser frontend/ /app/frontend/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser reports/ /app/reports/

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
