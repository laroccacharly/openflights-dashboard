#!/usr/bin/env python3
"""
Configuration settings for Docker container management.
This file centralizes all configuration parameters used by build.py and run.py.
"""

import os

# Docker image configuration
IMAGE_NAME = "openflights-app"
IMAGE_TAG = "latest"
FULL_IMAGE_NAME = f"{IMAGE_NAME}:{IMAGE_TAG}"

# Docker container configuration
CONTAINER_NAME = "openflights-container"

# Docker platform configuration
PLATFORM = "linux/amd64"  # Use amd64 for compatibility with fireducks package

# Docker build configuration
DOCKERFILE_PATH = "."
BUILD_ARGS = {}  # Add any build arguments as key-value pairs if needed

# Docker run configuration
PORT_MAPPINGS = []  # Example: ["8080:8080"] to map host port 8080 to container port 8080
ENVIRONMENT_VARS = {}  # Example: {"DEBUG": "true"} for environment variables
VOLUMES = [f"{os.path.abspath('src')}:/app/src"]  # Mount only src directory to /app/src in container

# Application configuration
APP_COMMAND = ["uv", "run", "src/app.py"]  # Updated command to run app.py from src directory 