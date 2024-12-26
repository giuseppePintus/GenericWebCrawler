FROM python:3.9

WORKDIR /crawler

# Copy requirements first for better caching
COPY dockerImage/requirements.txt .
#scrapy configuration
RUN pip install -r requirements.txt

# Create necessary directories
RUN mkdir -p /crawler/logs /crawler/output /crawler/crawler

# Copy the Scrapy project structure
COPY crawler /crawler/crawler/
COPY config /crawler/config/

# Generate scrapy.cfg
RUN echo "[settings]\ndefault = crawler.settings" > scrapy.cfg

# Set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/crawler"