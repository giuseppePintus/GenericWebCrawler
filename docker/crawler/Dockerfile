FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/config /app/data /app/logs /app/src/crawler/spiders

# Copy requirements and install dependencies
COPY docker/crawler/requirements-crawler.txt /app/
RUN pip install --no-cache-dir -r requirements-crawler.txt \
    && pip install python-magic

# Copy source code
COPY src/crawler/spiders/* /app/src/crawler/spiders/
COPY config/* /app/config/

# Set permissions
RUN chmod -R 755 /app/src

WORKDIR /app/src/crawler
ENTRYPOINT ["python", "spiders/general_spider.py"]