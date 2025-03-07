#!/usr/bin/env python3
"""
Script to stop and remove the Docker container for the OpenFlights dashboard.
This script will:
1. Check if the container exists
2. Stop the container if it's running
3. Remove the container
"""

import subprocess
import json
from config import CONTAINER_NAME

def stop_and_remove_container():
    """Stop and remove the Docker container."""
    # Check if container exists
    result = subprocess.run(
        ["docker", "container", "inspect", CONTAINER_NAME],
        check=False,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Container '{CONTAINER_NAME}' does not exist. Nothing to stop.")
        return
    
    # Parse the JSON output to check if the container is running
    container_running = False
    try:
        container_info = json.loads(result.stdout)
        if container_info and container_info[0]["State"]["Running"]:
            container_running = True
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing container info: {e}")
        return
    
    # Stop the container if it's running
    if container_running:
        print(f"Stopping container '{CONTAINER_NAME}'...")
        stop_result = subprocess.run(
            ["docker", "stop", CONTAINER_NAME],
            check=False,
            capture_output=True,
            text=True
        )
        
        if stop_result.returncode != 0:
            print(f"Error stopping container: {stop_result.stderr}")
            return
        
        print(f"Container '{CONTAINER_NAME}' stopped successfully.")
    else:
        print(f"Container '{CONTAINER_NAME}' is not running.")
    
    # Remove the container
    print(f"Removing container '{CONTAINER_NAME}'...")
    remove_result = subprocess.run(
        ["docker", "rm", CONTAINER_NAME],
        check=False,
        capture_output=True,
        text=True
    )
    
    if remove_result.returncode != 0:
        print(f"Error removing container: {remove_result.stderr}")
        return
    
    print(f"Container '{CONTAINER_NAME}' removed successfully.")

if __name__ == "__main__":
    stop_and_remove_container() 