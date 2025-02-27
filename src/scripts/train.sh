#!/bin/bash

# Navigate to your project directory (if needed)
# cd /path/to/your/project
make build-testing
# Run the testing container and execute the Python script
docker-compose run --rm \
    -v ${PWD}/config:/app/config:ro \
    -v ${PWD}/data:/app/data \
    -v ${PWD}/output:/app/output \
    testing \
    python /app/src/notebook/testing_hallucination_detection.py

    