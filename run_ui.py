#!/usr/bin/env python3
import subprocess
import sys
import os
from config import CONTAINER_NAME, UI_COMMAND

def run_ui_in_container():
    """Run the Streamlit UI in the Docker container."""
    print("Starting OpenFlights Dashboard UI in container...")
    
    # Check if container is running
    result = subprocess.run(
        ["docker", "container", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        check=False,
        capture_output=True,
        text=True
    )
    
    container_running = result.returncode == 0 and result.stdout.strip() == "true"
    
    if not container_running:
        print(f"Container '{CONTAINER_NAME}' is not running.")
        print("Starting container...")
        
        # Try to start the container if it exists
        start_result = subprocess.run(
            ["docker", "start", CONTAINER_NAME],
            check=False,
            capture_output=True,
            text=True
        )
        
        # If container doesn't exist or can't be started, run the run.py script
        if start_result.returncode != 0:
            print("Container doesn't exist or couldn't be started. Running build and setup...")
            
            # Check if build.py exists and run it
            if os.path.exists("build.py"):
                print("Building Docker image...")
                build_result = subprocess.run(["python", "build.py"], check=False)
                if build_result.returncode != 0:
                    print("Failed to build Docker image.")
                    sys.exit(1)
            
            # Run the container setup from run.py but don't execute the app command
            if os.path.exists("run.py"):
                # Import the run_docker_container function from run.py
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from run import run_docker_container
                
                # Run the container setup
                print("Setting up container...")
                run_docker_container(run_app=False)
            else:
                print("Error: run.py not found.")
                sys.exit(1)
    
    # Run the UI command in the container
    print(f"Running UI with command: {' '.join(UI_COMMAND)}")
    print("URL: http://localhost:8501")
    exec_cmd = ["docker", "exec", "-it", CONTAINER_NAME, "bash", "-c", f"cd /app && {' '.join(UI_COMMAND)}"]
    
    try:
        subprocess.run(exec_cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down OpenFlights Dashboard UI...")
    except subprocess.CalledProcessError as e:
        print(f"UI exited with error code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ui_in_container()
