# Production Dockerfile for FastAPI MLOps Inference Service
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ml/ ./ml/
COPY data/ ./data/
COPY ecommerce.db .
COPY mlflow.db* ./

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI microservice
CMD ["uvicorn", "ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
