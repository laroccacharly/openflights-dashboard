import streamlit as st
import fireducks.pandas as pd
import os
import numpy as np
import plotly.express as px
from data import get_data

# Set page configuration
st.set_page_config(
    page_title="OpenFlights Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("✈️ OpenFlights Dashboard")

# Load data
with st.spinner("Loading OpenFlights data..."):
    try:
        airports_df, routes_df = get_data()
        st.success(f"Successfully loaded {len(airports_df)} airports and {len(routes_df)} routes!")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Sidebar for filtering
st.sidebar.title("Filters")

# Sampling ratio slider for performance
sampling_ratio = st.sidebar.slider(
    "Map Sampling Ratio",
    min_value=0.01,
    max_value=1.0,
    value=0.1,
    step=0.01,
    help="Percentage of airports to display on the map for better performance. Lower values = faster loading."
)

# Country filter
countries = sorted(airports_df['country'].dropna().unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=[]
)

# Apply filters
filtered_airports = airports_df
if selected_countries:
    filtered_airports = airports_df[airports_df['country'].isin(selected_countries)]

# Main content
st.header("Airport Data")

# Display basic statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Airports", len(filtered_airports))
with col2:
    st.metric("Total Countries", filtered_airports['country'].nunique())
with col3:
    st.metric("Total Routes", len(routes_df))

# Map of airports - Make it larger
st.subheader("Airport Locations")
# Filter out invalid coordinates
map_data = filtered_airports[
    (filtered_airports['latitude'].notna()) & 
    (filtered_airports['longitude'].notna()) &
    (filtered_airports['latitude'] >= -90) & 
    (filtered_airports['latitude'] <= 90) &
    (filtered_airports['longitude'] >= -180) & 
    (filtered_airports['longitude'] <= 180)
].copy()

# Apply sampling for better performance
if not map_data.empty:
    # Only sample if we have more than a few airports
    if len(map_data) > 10:
        # Calculate sample size based on the ratio
        sample_size = max(10, int(len(map_data) * sampling_ratio))
        # Sample the data
        map_data = map_data.sample(n=sample_size, random_state=42)
        st.info(f"Displaying {sample_size} airports ({sampling_ratio:.0%} of {len(filtered_airports)} filtered airports) for better performance.")
    
    # Create a larger map with double the height
    fig = px.scatter_geo(
        map_data,
        lat='latitude',
        lon='longitude',
        hover_name='name',
        hover_data=['city', 'country', 'iata'],
        color='country' if len(selected_countries) <= 10 else None,
        title="Airport Locations Worldwide",
        projection="natural earth"
    )
    
    # Update layout to make the map twice as big
    fig.update_layout(
        height=800,  # Double the default height
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No airports to display on the map with the current filters.")

# Display data tables
st.subheader("Airport Data Table")
st.dataframe(filtered_airports, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Data source: [OpenFlights](https://openflights.org/data.html)") 