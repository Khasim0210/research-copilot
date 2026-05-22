# Use slim Python image (small + fast)
FROM python:3.11-slim

# System deps needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker layer caching: speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sentencepiece python-multipart

# Copy the rest of the application
COPY src/ ./src/
COPY data/ ./data/
COPY main.py ./

# Expose the FastAPI port
EXPOSE 8000

# Healthcheck (used by docker compose / orchestrators)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]