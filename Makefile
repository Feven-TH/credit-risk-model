.PHONY: test build up down clean

# Run all tests using the Mocking strategy we built
test:
	PYTHONPATH=. pytest tests/ -v

# Build the Docker image from your final Dockerfile
build:
	docker compose build

# Start the API container
up:
	docker compose up

# Stop the container and clean up
down:
	docker compose down

# Wipe out python cache to keep things clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache