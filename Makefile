CRAWLER_IMAGE=general_crawler
DOCKER_DIR=dockerImage
CONFIG_DIR=config
GPU_FLAG=--gpus all  # For GPU support

.PHONY: build crawl process-chunks index clean all

build:
	docker build -t ${CRAWLER_IMAGE} .

crawl:
	docker run --rm ${GPU_FLAG} \
		-v ${PWD}/config:/crawler/config:ro \
		-v ${PWD}/output:/crawler/output \
		-v ${PWD}/logs:/crawler/logs \
		${CRAWLER_IMAGE} \
		scrapy crawl general_spider \
		-a config_file=/crawler/config/crawler_config.json \
		-a site=${site} \
		--logfile=/crawler/logs/crawler.log \
		-L DEBUG

process-chunks:
	docker run --rm ${GPU_FLAG} \
		-v ${PWD}/output:/crawler/output \
		${CRAWLER_IMAGE} \
		python scripts/process_chunks.py \
		--site ${site} \
		--use-gpu

index:
	docker run --rm \
		-v ${PWD}/output:/crawler/output \
		${CRAWLER_IMAGE} \
		python scripts/create_index.py \
		--site ${site}

clean:
	rm -rf output/* logs/*

all: crawl process-chunks index