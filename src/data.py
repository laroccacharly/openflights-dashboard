#!/usr/bin/env python3
"""
Data management module for OpenFlights dashboard.
This module handles downloading, caching, and loading the OpenFlights datasets.
"""

import os
import requests
# import fireducks.pandas as pd
from .pandas import pd 
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

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
    data_dir = DATA_DIR if DATA_DIR.exists() else LOCAL_DATA_DIR
    data_dir.mkdir(exist_ok=True)
    return data_dir

def download_file(url: str, filename: str) -> Path:
    """
    Download a file from a URL and save it to the data directory.
    
    Args:
        url: The URL to download from
        filename: The name to save the file as
        
    Returns:
        Path to the downloaded file
    """
    data_dir = ensure_data_dir()
    raw_file_path = data_dir / filename
    
    # Generate the parquet filename by replacing the extension
    parquet_filename = filename.rsplit('.', 1)[0] + '.parquet'
    parquet_file_path = data_dir / parquet_filename
    
    # Check if parquet file already exists
    if parquet_file_path.exists():
        print(f"Using cached {parquet_filename}")
        return parquet_file_path
    
    # Download the raw file if it doesn't exist
    if not raw_file_path.exists():
        print(f"Downloading {url} to {raw_file_path}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(raw_file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} successfully")
    else:
        print(f"Using cached {filename}")
    
    # Return the raw file path - the loading functions will handle conversion to parquet
    return raw_file_path

def load_airports() -> pd.DataFrame:
    """
    Load the airports dataset.
    
    Returns:
        DataFrame containing airport data
    """
    data_dir = ensure_data_dir()
    parquet_file_path = data_dir / "airports.parquet"
    
    # Check if parquet file exists
    if parquet_file_path.exists():
        print(f"Loading airports from parquet file")
        return pd.read_parquet(parquet_file_path)
    
    # If not, download and process the raw file
    file_path = download_file(AIRPORTS_URL, "airports.dat")
    
    # Load the data with proper column names
    df = pd.read_csv(
        file_path,
        header=None,
        names=AIRPORTS_COLUMNS,
        na_values=["\\N", ""],
        keep_default_na=True,
        encoding='utf-8',
        quotechar='"',
        escapechar='\\'
    )
    
    # Save as parquet for future use
    df.to_parquet(parquet_file_path, index=False)
    print(f"Saved airports data to {parquet_file_path}")
    
    return df

def load_routes() -> pd.DataFrame:
    """
    Load the routes dataset.
    
    Returns:
        DataFrame containing route data
    """
    data_dir = ensure_data_dir()
    parquet_file_path = data_dir / "routes.parquet"
    
    # Check if parquet file exists
    if parquet_file_path.exists():
        print(f"Loading routes from parquet file")
        return pd.read_parquet(parquet_file_path)
    
    # If not, download and process the raw file
    file_path = download_file(ROUTES_URL, "routes.dat")
    
    # Load the data with proper column names
    df = pd.read_csv(
        file_path,
        header=None,
        names=ROUTES_COLUMNS,
        na_values=["\\N", ""],
        keep_default_na=True,
        encoding='utf-8'
    )
    
    # Save as parquet for future use
    df.to_parquet(parquet_file_path, index=False)
    print(f"Saved routes data to {parquet_file_path}")
    
    return df

def get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get both airports and routes datasets.
    
    Returns:
        Tuple of (airports_df, routes_df)
    """
    airports_df = load_airports()
    routes_df = load_routes()
    
    print(f"Loaded {len(airports_df)} airports and {len(routes_df)} routes")
    
    return airports_df, routes_df

if __name__ == "__main__":
    # Test the data loading functions
    airports, routes = get_data()
    print(f"Airports sample:\n{airports.head()}")
    print(f"Routes sample:\n{routes.head()}") 