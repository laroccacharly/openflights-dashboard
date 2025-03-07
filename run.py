#!/usr/bin/env python3
import subprocess
import sys
import os
import json
from config import CONTAINER_NAME, PORT_MAPPINGS, ENVIRONMENT_VARS, VOLUMES, PLATFORM, APP_COMMAND, FULL_IMAGE_NAME

def run_docker_container(run_app=True):
    """Run the Docker container.
    
    Args:
        run_app (bool): Whether to run the application command after starting the container.
    """
    # Check if container exists and is running
    container_exists = False
    container_running = False
    
    # Check if container exists
    result = subprocess.run(
        ["docker", "container", "inspect", CONTAINER_NAME],
        check=False,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        container_exists = True
        # Parse the JSON output to check if the container is running
        try:
            container_info = json.loads(result.stdout)
            if container_info and container_info[0]["State"]["Running"]:
                container_running = True
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing container info: {e}")
    
    # If container doesn't exist, create it
    if not container_exists:
        print(f"Container '{CONTAINER_NAME}' does not exist. Creating it...")
        create_container_cmd = ["docker", "create", f"--platform={PLATFORM}", "--name", CONTAINER_NAME]
        
        # Add volume mappings
        for volume in VOLUMES:
            create_container_cmd.extend(["-v", volume])
            
        # Add port mappings
        for port_mapping in PORT_MAPPINGS:
            create_container_cmd.extend(["-p", port_mapping])
            
        # Add environment variables
        for env_name, env_value in ENVIRONMENT_VARS.items():
            create_container_cmd.extend(["-e", f"{env_name}={env_value}"])
            
        # Add image name
        create_container_cmd.append(FULL_IMAGE_NAME)
        
        result = subprocess.run(
            create_container_cmd,
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error creating Docker container:\n{result.stderr}")
            sys.exit(1)
            
        print("Container created successfully.")
        container_exists = True
    
    # If container exists but is not running, start it
    if container_exists and not container_running:
        print(f"Starting container '{CONTAINER_NAME}'...")
        result = subprocess.run(
            ["docker", "start", CONTAINER_NAME],
            check=False,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error starting Docker container: {result.stderr}")
            sys.exit(1)
        
        print("Container started successfully.")
    elif container_running:
        print(f"Container '{CONTAINER_NAME}' is already running.")
    
    # Run the application using docker exec if run_app is True
    if run_app:
        print(f"Running application with command: {' '.join(APP_COMMAND)}")
        # Use bash to ensure proper environment setup
        exec_cmd = ["docker", "exec", "-it", CONTAINER_NAME, "bash", "-c", f"cd /app && {' '.join(APP_COMMAND)}"]
        
        result = subprocess.run(
            exec_cmd,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Application exited with error code: {result.returncode}")
            sys.exit(1)
    else:
        print("Container is ready. Not running application command.")

if __name__ == "__main__":
    run_docker_container() 