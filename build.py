#!/usr/bin/env python3
import subprocess
import sys
from config import DOCKERFILE_PATH, BUILD_ARGS, FULL_IMAGE_NAME, PLATFORM

def build_docker_image():
    """Build the Docker image from the Dockerfile."""
    print(f"Building Docker image '{FULL_IMAGE_NAME}' for platform '{PLATFORM}'...")
    
    # Prepare build command
    build_cmd = ["docker", "build", f"--platform={PLATFORM}", "-t", FULL_IMAGE_NAME, DOCKERFILE_PATH]
    
    # Add build arguments if any
    for arg_name, arg_value in BUILD_ARGS.items():
        build_cmd.extend(["--build-arg", f"{arg_name}={arg_value}"])
    
    # Run the build command with real-time output
    print("\n--- Docker Build Logs ---")
    result = subprocess.run(
        build_cmd,
        check=False,
        text=True
    )
    print("--- End of Docker Build Logs ---\n")
    
    if result.returncode != 0:
        print(f"Error building Docker image. Build failed with exit code {result.returncode}")
        sys.exit(1)
    
    print("Docker image built successfully!")
    return FULL_IMAGE_NAME

if __name__ == "__main__":
    # Build the Docker image
    image_name = build_docker_image()
    
    print(f"\nBuild completed successfully!")
    print(f"Image name: {image_name}")
