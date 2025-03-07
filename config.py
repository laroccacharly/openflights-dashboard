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
BUILD_ARGS = {}  # Add any build arguments as needed

# Docker run configuration
PORT_MAPPINGS = ["8501:8501"]  # Map host port 8501 to container port 8501 for Streamlit
ENVIRONMENT_VARS = {
    "PYTHONPATH": "/app"  # Add /app to Python path to ensure modules can be found
}
VOLUMES = [
    f"{os.path.abspath('src')}:/app/src",  # Mount src directory to /app/src in container
    f"{os.path.abspath('data')}:/app/data"  # Mount data directory to /app/data in container
]

# Application configuration
APP_COMMAND = ["uv", "run", "src/app.py"]  # Command to run app.py from src directory

# UI configuration
UI_COMMAND = ["uv", "run", "streamlit", "run", "src/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]  # Command to run Streamlit UI 