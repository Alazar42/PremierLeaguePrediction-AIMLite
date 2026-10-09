# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies (curl for healthchecks & network tooling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install uv (blazing fast backend) and AIMLite framework
RUN pip install uv aimlite

# 2. Set project working directory
WORKDIR /app

# 3. Copy project manifest
COPY aimlite.json .

# 4. Install all project dependencies into managed .venv using aimlite CLI
RUN aimlite install

# 5. Copy the remaining application files, data, and frontend assets
COPY . .

# 6. Train and calibrate model weights for serving
RUN aimlite train

# 7. Expose serving port
EXPOSE 8000

# 8. Health check verifying inference server status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 9. Host multi-model inference server with custom frontend
CMD ["aimlite", "serve", "--host", "0.0.0.0", "--port", "8000", "--frontend", "frontend"]
