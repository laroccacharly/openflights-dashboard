#!/usr/bin/env python3
"""
Sample application using fireducks.
This file can be modified without rebuilding the Docker image.
Running with 'uv run' instead of 'python'.
"""

def main():
    print("Hello from the fireducks application!")
    print("This application is running inside a Docker container using uv run.")
    print("You can modify this file and run it again without rebuilding the image.")
    
    # Try to import fireducks to verify it's installed
    try:
        import fireducks
        print(f"Successfully imported fireducks package (version: {getattr(fireducks, '__version__', 'unknown')})!")
    except ImportError as e:
        print(f"Error importing fireducks: {e}")

if __name__ == "__main__":
    main()