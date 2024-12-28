FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/embedding/requirements-embedding.txt .
RUN pip install --no-cache-dir -r requirements-embedding.txt

# Copy source code
COPY src/embedding/ .

CMD ["python", "chunk_embedder.py"]