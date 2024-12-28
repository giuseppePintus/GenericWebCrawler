BOT_NAME = 'general_crawler'
SPIDER_MODULES = ['crawler.spiders']
NEWSPIDER_MODULE = 'crawler.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure logging
LOG_LEVEL = "INFO"
LOG_FILE = "/crawler/logs/crawler.log"

# Configure maximum concurrent requests
CONCURRENT_REQUESTS = 16
DOWNLOAD_DELAY = 1

# Enable or disable HTTP compression
COMPRESSION_ENABLED = True