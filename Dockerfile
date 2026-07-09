# Deploys the playground. CPU-only, no GPU, no API keys.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY vecstore ./vecstore
COPY playground ./playground
RUN pip install --no-cache-dir -e ".[demo]"

EXPOSE 8000
# Railway/Fly set $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn playground.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
