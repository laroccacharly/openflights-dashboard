#!/usr/bin/env python3
"""
Data management module for OpenFlights dashboard.
This module handles downloading, caching, and loading the OpenFlights datasets.
"""

import requests
from . import pandas as pd_module
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('openflights-data')

# Data source URLs
AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
ROUTES_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

# Data directory
DATA_DIR = Path("/app/data")  # Path inside Docker container
LOCAL_DATA_DIR = Path("data")  # Path for local development

# Column names for the datasets
AIRPORTS_COLUMNS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "latitude", "longitude", "altitude", "timezone", "dst",
    "tz_database_timezone", "type", "source"
]

ROUTES_COLUMNS = [
    "airline", "airline_id", "source_airport", "source_airport_id",
    "destination_airport", "destination_airport_id", "codeshare",
    "stops", "equipment"
]

def ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    # Use the Docker container path if it exists, otherwise use local path
    if DATA_DIR.exists():
        data_dir = DATA_DIR
    else:
        data_dir = LOCAL_DATA_DIR
        
    # Create the directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    return data_dir

def download_file(url: str, filename: str) -> Path:
    """Download a file from a URL and save it to the data directory."""
    data_dir = ensure_data_dir()
    raw_file_path = data_dir / filename
    
    # If the file already exists, return its path
    if raw_file_path.exists():
        print(f"File {filename} already exists, skipping download")
        return raw_file_path
    
    # Download the file
    print(f"Downloading {url} to {raw_file_path}")
    response = requests.get(url)
    response.raise_for_status()
    
    # Save the file
    with open(raw_file_path, 'wb') as f:
        f.write(response.content)
    
    return raw_file_path

def load_airports() -> Any:
    """
    Load the airports dataset.
    Returns a DataFrame with airport information.
    """
    data_dir = ensure_data_dir()
    parquet_file_path = data_dir / "airports.parquet"
    
    # If the parquet file exists, load it directly
    if parquet_file_path.exists():
        print(f"Loading airports from parquet file")
        return pd_module.read_parquet(parquet_file_path)
    
    # If not, download and process the raw file
    file_path = download_file(AIRPORTS_URL, "airports.dat")
    
    # Load the data with proper column names
    df = pd_module.read_csv(
        file_path,
        header=None,
        names=AIRPORTS_COLUMNS,
        na_values=["\\N", ""],
        keep_default_na=True,
        encoding='utf-8'
    )
    
    # Save as parquet for faster loading next time
    df.to_parquet(parquet_file_path, index=False)
    
    return df

def load_routes() -> Any:
    """
    Load the routes dataset.
    Returns a DataFrame with route information.
    """
    data_dir = ensure_data_dir()
    parquet_file_path = data_dir / "routes.parquet"
    
    # If the parquet file exists, load it directly
    if parquet_file_path.exists():
        print(f"Loading routes from parquet file")
        return pd_module.read_parquet(parquet_file_path)
    
    # If not, download and process the raw file
    file_path = download_file(ROUTES_URL, "routes.dat")
    
    # Load the data with proper column names
    df = pd_module.read_csv(
        file_path,
        header=None,
        names=ROUTES_COLUMNS,
        na_values=["\\N", ""],
        keep_default_na=True,
        encoding='utf-8'
    )
    
    # Save as parquet for faster loading next time
    df.to_parquet(parquet_file_path, index=False)
    
    return df

def get_data() -> Tuple[Any, Any]:
    """
    Get both airports and routes datasets.
    Returns a tuple of (airports_df, routes_df).
    """
    start_time = time.time()
    
    airports_df = load_airports()
    routes_df = load_routes()
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"Airports: {len(airports_df)} rows, Routes: {len(routes_df)} rows")
    
    return airports_df, routes_df

if __name__ == "__main__":
    # Test the data loading functions
    airports, routes = get_data()
    print(f"Airports sample:\n{airports.head()}")
    print(f"Routes sample:\n{routes.head()}") 