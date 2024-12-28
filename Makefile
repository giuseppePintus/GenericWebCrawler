# Configuration
CRAWLER_IMAGE=crawler-service
EMBEDDER_IMAGE=embedder-service
DOCKER_COMPOSE=docker-compose
CONFIG_DIR=config
DATA_DIR=data
LOGS_DIR=logs
MILVUS_PORT=19530
RESET=0

# Ensure required directories exist
$(shell mkdir -p ${CONFIG_DIR} ${DATA_DIR} ${LOGS_DIR})

.PHONY: build build-all crawl embed clean logs validate-site milvus-up milvus-down check-milvus milvus-clean milvus-reset-collection milvus-status search

# Build individual services
build-crawler:
	${DOCKER_COMPOSE} build crawler

build-embedder:
	${DOCKER_COMPOSE} build embedding

# Build all services
build-all: build-crawler build-embedder

# Infrastructure commands
milvus-clean:
	@echo "Stopping existing Milvus containers..."
	-docker stop milvus-etcd milvus-minio milvus || true
	@echo "Removing existing Milvus containers..."
	-docker rm -f milvus-etcd milvus-minio milvus || true
	@echo "Cleaning up Docker system..."
	-docker system prune -f
	@echo "Cleanup complete"

milvus-reset-collection:
	@echo "Resetting Milvus collection..."
	@mkdir -p scripts
	${DOCKER_COMPOSE} run --rm \
		-e PYTHONPATH=/app \
		embedding python /app/scripts/reset_collection.py

milvus-status:
	@echo "Checking Milvus status..."
	@docker ps --filter name=milvus
	@echo "\nChecking collection info..."
	${DOCKER_COMPOSE} run --rm \
		-e PYTHONPATH=/app \
		embedding python -c '\
		from pymilvus import connections, utility; \
		connections.connect("default", host="milvus", port="19530"); \
		print("Collections:", utility.list_collections()); \
		connections.disconnect("default");'

milvus-up: milvus-clean
	@echo "Starting Milvus infrastructure..."
	${DOCKER_COMPOSE} up -d etcd minio milvus
	@echo "Waiting for services to start..."
	@sleep 10

milvus-down:
	@echo "Stopping Milvus stack..."
	-${DOCKER_COMPOSE} stop milvus minio etcd
	-${DOCKER_COMPOSE} rm -f milvus minio etcd
	@echo "Milvus stack stopped and removed"

# Check if Milvus is running
check-milvus:
	@nc -z localhost ${MILVUS_PORT} || (echo "Error: Milvus is not running. Please run 'make milvus-up' first" && exit 1)

# Run crawler with site parameter
crawl: validate-site
	${DOCKER_COMPOSE} run --rm \
		-v ${PWD}/${CONFIG_DIR}:/app/config:ro \
		-v ${PWD}/${DATA_DIR}:/app/data \
		-v ${PWD}/${LOGS_DIR}:/app/logs \
		crawler \
				--site $(site) \
				--config /app/config/crawler_config.json \
				--output /app/data \
				--log /app/logs/crawler.log

# Create and store embeddings in Milvus
embed: check-milvus
	${DOCKER_COMPOSE} run --rm \
		-v ${PWD}/${CONFIG_DIR}:/config:ro \
		-v ${PWD}/${DATA_DIR}:/app/data \
		embedding \
		python chunk_embedder.py $(if $(filter 1,${RESET}),--reset,)

# Search in Milvus vector database
search: check-milvus
	@if [ -z "$(query)" ]; then \
		echo "Error: query parameter is required. Usage: make search query='your search query' [top_k=10]"; \
		exit 1; \
	fi
	${DOCKER_COMPOSE} run --rm \
		-v ${PWD}/${CONFIG_DIR}:/config:ro \
		-v ${PWD}/${DATA_DIR}:/app/data \
		-v ${PWD}/src/scripts:/app/scripts \
		embedding \
		python /app/scripts/milvus_search.py \
			--query "$(query)" \
			--top_k $(or $(top_k),10)

# Full pipeline
process-site: validate-site milvus-up crawl embed

# Clean up
clean:
	${DOCKER_COMPOSE} down -v
	rm -rf ${DATA_DIR}/* ${LOGS_DIR}/*

# View logs
logs:
	tail -f ${LOGS_DIR}/crawler.log

# Validate site parameter
validate-site:
	@if [ -z "$(site)" ]; then \
		echo "Error: site parameter is required. Usage: make crawl site=example.com"; \
		exit 1; \
	fi

# Help
help:
	@echo "Available commands:"
	@echo "  build-crawler    - Build crawler service"
	@echo "  build-embedder   - Build embedding service"
	@echo "  build-all       - Build all services"
	@echo "  milvus-up       - Start Milvus infrastructure"
	@echo "  milvus-down     - Stop Milvus infrastructure"
	@echo "  crawl           - Run crawler (requires site=<domain>)"
	@echo "  embed           - Create and store embeddings"
	@echo "  process-site    - Run full pipeline (requires site=<domain>)"
	@echo "  clean           - Clean up all containers and data"
	@echo "  logs            - View crawler logs"
	@echo "  milvus-reset-collection - Reset Milvus collection"
	@echo "  milvus-status   - Check Milvus status"
	@echo "  search          - Search in Milvus vector database"