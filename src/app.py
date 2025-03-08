#!/usr/bin/env python3
"""
Main application for OpenFlights Dashboard.
This file can be modified without rebuilding the Docker image.
Running with 'uv run' instead of 'python'.
"""

import fireducks.pandas as pd
from .data import get_data

def main():
    print("Starting OpenFlights Dashboard application...")
    print("This application is running inside a Docker container using uv run.")
    
    # Try to import fireducks to verify it's installed
    try:
        import fireducks
        print(f"Successfully imported fireducks package (version: {getattr(fireducks, '__version__', 'unknown')})!")
    except ImportError as e:
        print(f"Error importing fireducks: {e}")
    
    # Load the OpenFlights data
    try:
        airports_df, routes_df = get_data()
        print(f"Successfully loaded OpenFlights data:")
        print(f"  - {len(airports_df)} airports")
        print(f"  - {len(routes_df)} routes")
    except Exception as e:
        print(f"Error loading OpenFlights data: {e}")

if __name__ == "__main__":
    main()