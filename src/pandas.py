# import fireducks.pandas as pd
import os

# Global variable to track which backend is currently selected
# Default to regular pandas
CURRENT_BACKEND = "pandas"

def set_backend(backend_name):
    """
    Set the pandas backend to use (either 'pandas' or 'fireducks')
    """
    global CURRENT_BACKEND
    CURRENT_BACKEND = backend_name
    
    # Update the module's attributes with the new backend
    pd_module = get_pandas_module()
    
    # Update global attributes to match the selected pandas module
    global pd, DataFrame, Series, read_csv, read_parquet, merge
    pd = pd_module
    DataFrame = pd_module.DataFrame
    Series = pd_module.Series
    read_csv = pd_module.read_csv
    read_parquet = pd_module.read_parquet
    merge = pd_module.merge
    
    return pd_module

def get_pandas_module():
    """
    Returns the pandas module based on the selected backend.
    """
    global CURRENT_BACKEND
    
    if CURRENT_BACKEND == 'fireducks':
        try:
            import fireducks.pandas as pd
            return pd
        except ImportError:
            import pandas as pd
            print("Warning: Fireducks not installed. Falling back to pandas.")
            return pd
    else:
        import pandas as pd
        return pd

# Initialize the pandas module
pd = get_pandas_module()

# Expose common pandas classes and functions directly
DataFrame = pd.DataFrame
Series = pd.Series
read_csv = pd.read_csv
read_parquet = pd.read_parquet
merge = pd.merge

