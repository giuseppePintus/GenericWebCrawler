# Configuration
CRAWLER_IMAGE=crawler-service
EMBEDDER_IMAGE=embedder-service
TESTING_IMAGE=testing-service  # <-- Nome dell'immagine Docker per il testing
DOCKER_COMPOSE=docker-compose
CONFIG_DIR=config
DATA_DIR=data
LOGS_DIR=logs
MILVUS_PORT=19530
RESET=0
BACKEND_PORT=5000
FRONTEND_PORT=3000
BACKEND_IMAGE=chatbot-backend
FRONTEND_IMAGE=chatbot-frontend
CHATBOT_DIR=src/chatbot-ui
CHATBOT_COMPOSE=docker-compose -f ${CHATBOT_DIR}/docker-compose.yml

# Ensure required directories exist
$(shell mkdir -p ${CONFIG_DIR} ${DATA_DIR} ${LOGS_DIR})

.PHONY: build build-all build-crawler build-embedder build-testing crawl embed clean logs validate-site milvus-up milvus-down check-milvus milvus-clean milvus-reset-collection milvus-status search process-site help test-hallucination chatbot-build chatbot-up chatbot-down chatbot-status

# Build individual services
build-crawler:
	${DOCKER_COMPOSE} build crawler

build-embedder:
	${DOCKER_COMPOSE} build embedding

# Nuovo: build dell'immagine di testing
build-testing:
	${DOCKER_COMPOSE} build testing

build-chatbot:
	@if [ ! -f "${CHATBOT_DIR}/docker-compose.yml" ]; then \
		echo "Error: Chatbot docker-compose.yml not found in ${CHATBOT_DIR}"; \
		exit 1; \
	fi
	@echo "Building chatbot services..."
	${CHATBOT_COMPOSE} build backend frontend

# Build all services
build-all: build-crawler build-embedder build-testing

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

# Run crawler with site parametermake mi
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

# Nuovo: test-hallucination => esegue /src/notebook/testing_hallucination_detection.py dentro container "testing"
test-hallucination: build-testing
	@echo "Running hallucination detection tests..."
	${DOCKER_COMPOSE} run --rm \
		-v ${PWD}/${CONFIG_DIR}:/app/config:ro \
		-v ${PWD}/${DATA_DIR}:/app/data \
		testing \
		python /app/src/notebook/testing_hallucination_detection.py

chatbot-up: check-milvus
	@if [ ! -f "${CHATBOT_DIR}/docker-compose.yml" ]; then \
		echo "Error: Chatbot docker-compose.yml not found in ${CHATBOT_DIR}"; \
		exit 1; \
	fi
	@echo "Ensuring Milvus network exists..."
	@docker network inspect milvus-net >/dev/null 2>&1 || docker network create milvus-net
	@echo "Starting chatbot services..."
	${CHATBOT_COMPOSE} up -d backend frontend

chatbot-down:
	@if [ -f "${CHATBOT_DIR}/docker-compose.yml" ]; then \
		echo "Stopping chatbot services..."; \
		${CHATBOT_COMPOSE} stop frontend backend; \
		${CHATBOT_COMPOSE} rm -f frontend backend; \
	fi

chatbot-status:
	@if [ ! -f "${CHATBOT_DIR}/docker-compose.yml" ]; then \
		echo "Error: Chatbot docker-compose.yml not found in ${CHATBOT_DIR}"; \
		exit 1; \
	fi
	@echo "Checking chatbot services status..."
	@echo "\nFrontend:"
	@${CHATBOT_COMPOSE} ps frontend
	@echo "\nBackend:"
	@${CHATBOT_COMPOSE} ps backend
	@echo "\nAPI Status:"
	@curl -s http://localhost:${BACKEND_PORT}/health || echo "Backend not responding"

# Full stack command
start-all: milvus-up chatbot-up
	@echo "All services started successfully"

stop-all: chatbot-down milvus-down
	@echo "All services stopped successfully"

# Help
help:
	@echo "Available commands:"
	@echo "  build-crawler           - Build crawler service"
	@echo "  build-embedder          - Build embedding service"
	@echo "  build-testing           - Build testing service"
	@echo "  build-chatbot           - Build chatbot frontend/backend"
	@echo "  build-all               - Build all services"
	@echo "  milvus-up               - Start Milvus infrastructure"
	@echo "  milvus-down             - Stop Milvus infrastructure"
	@echo "  milvus-reset-collection - Reset Milvus collection"
	@echo "  milvus-status           - Check Milvus status"
	@echo "  chatbot-up              - Start chatbot services"
	@echo "  chatbot-down            - Stop chatbot services"
	@echo "  chatbot-status          - Check chatbot services status"
	@echo "  start-all               - Start full stack (Milvus + Chatbot)"
	@echo "  stop-all                - Stop all services"
	@echo "  crawl                   - Run crawler (requires site=...)"
	@echo "  embed                   - Create and store embeddings"
	@echo "  search                  - Search in Milvus vector database"
	@echo "  process-site            - Full pipeline (site=...)"
	@echo "  check-milvus            - Verify Milvus is running"
	@echo "  clean                   - Clean all containers/data"
	@echo "  logs                    - View crawler logs"
	@echo "  test-hallucination      - Run hallucination detection tests (Docker-based)"
	@echo "  help                    - Show this help"
