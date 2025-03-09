import streamlit as st
from .pandas import pd 
import os
import numpy as np
import plotly.express as px
import time
import logging
from .data import get_data
import math

# Configure logging for UI
logger = logging.getLogger('openflights-ui')

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def run_ui():
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
    if 'airports_df' not in st.session_state or 'routes_df' not in st.session_state:
        with st.spinner("Loading OpenFlights data..."):
            try:
                st.session_state.airports_df, st.session_state.routes_df = get_data()
                st.success(f"Successfully loaded {len(st.session_state.airports_df)} airports and {len(st.session_state.routes_df)} routes!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.stop()
    
    # Use the data from session state
    airports_df = st.session_state.airports_df
    routes_df = st.session_state.routes_df

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
    # Airport Connectivity Analysis
    st.header("Airport Routes Analysis")
    st.write("This analysis joins the airports and routes datasets to analyze airport connectivity.")
    
    # Add a button to run the analysis (to avoid running it automatically every time)
    if st.button("Run Routes Analysis"):
        with st.spinner("Analyzing airport routes..."):
            # Benchmark the join operation
            start_time = time.time()
            
            # Outbound routes (departures)
            outbound_routes = routes_df.groupby('source_airport_id').size().reset_index(name='outbound_routes')
            
            # Inbound routes (arrivals)
            inbound_routes = routes_df.groupby('destination_airport_id').size().reset_index(name='inbound_routes')
            
            # Join with airports data
            # First, join with outbound routes
            connectivity_df = pd.merge(
                airports_df,
                outbound_routes,
                left_on='airport_id',
                right_on='source_airport_id',
                how='left'
            )
            
            # Then, join with inbound routes
            connectivity_df = pd.merge(
                connectivity_df,
                inbound_routes,
                left_on='airport_id',
                right_on='destination_airport_id',
                how='left'
            )
            
            # Fill NaN values with 0 (airports with no routes)
            connectivity_df['outbound_routes'] = connectivity_df['outbound_routes'].fillna(0).astype(int)
            connectivity_df['inbound_routes'] = connectivity_df['inbound_routes'].fillna(0).astype(int)
            
            # Calculate total connectivity (sum of inbound and outbound routes)
            connectivity_df['total_connectivity'] = connectivity_df['outbound_routes'] + connectivity_df['inbound_routes']
            
            # Calculate connectivity ratio (outbound/inbound)
            connectivity_df['connectivity_ratio'] = np.where(
                connectivity_df['inbound_routes'] > 0,
                connectivity_df['outbound_routes'] / connectivity_df['inbound_routes'],
                np.nan
            )
            
            # End benchmark
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Display benchmark results
            st.subheader("Benchmark Results")
            st.info(f"Join operation completed in {execution_time:.2f} seconds")
            
            # Filter for the selected countries if any
            if selected_countries:
                connectivity_df = connectivity_df[connectivity_df['country'].isin(selected_countries)]
            
            # Display the top 20 most connected airports
            st.subheader("Top 20 Most Connected Airports")
            top_connected = connectivity_df.sort_values('total_connectivity', ascending=False).head(20)
            
            # Create a bar chart for the top connected airports
            fig = px.bar(
                top_connected,
                x='name',
                y='total_connectivity',
                hover_data=['city', 'country', 'outbound_routes', 'inbound_routes'],
                color='country',
                title="Top 20 Most Connected Airports",
                labels={'name': 'Airport', 'total_connectivity': 'Total Routes (In + Out)'}
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate route distances
            st.subheader("Route Distance Distribution")
            
            # Create a progress bar for the distance calculation
            progress_bar = st.progress(0)
            
            # Create a dataframe with source and destination airport information
            route_distances_df = pd.merge(
                routes_df,
                airports_df[['airport_id', 'latitude', 'longitude']],
                left_on='source_airport_id',
                right_on='airport_id',
                how='inner'
            ).rename(columns={'latitude': 'source_lat', 'longitude': 'source_lon'})
            
            route_distances_df = pd.merge(
                route_distances_df,
                airports_df[['airport_id', 'latitude', 'longitude']],
                left_on='destination_airport_id',
                right_on='airport_id',
                how='inner'
            ).rename(columns={'latitude': 'dest_lat', 'longitude': 'dest_lon'})
            
            # Update progress
            progress_bar.progress(0.3)
            
            # Filter out routes with missing coordinates
            valid_routes = route_distances_df.dropna(subset=['source_lat', 'source_lon', 'dest_lat', 'dest_lon'])
            
            # Update progress
            progress_bar.progress(0.5)
            
            # Calculate distances for each route
            valid_routes['distance_km'] = valid_routes.apply(
                lambda row: haversine_distance(
                    row['source_lat'], row['source_lon'], 
                    row['dest_lat'], row['dest_lon']
                ),
                axis=1
            )
            
            # Update progress
            progress_bar.progress(0.8)
            
            # Create a histogram of route distances
            fig_dist = px.histogram(
                valid_routes,
                x='distance_km',
                nbins=50,
                title="Distribution of Flight Route Distances",
                labels={'distance_km': 'Distance (km)', 'count': 'Number of Routes'},
                marginal="box"  # Add a box plot on the marginal axis
            )
            
            # Add some statistics as annotations
            avg_distance = valid_routes['distance_km'].mean()
            median_distance = valid_routes['distance_km'].median()
            max_distance = valid_routes['distance_km'].max()
            
            fig_dist.add_annotation(
                x=0.95, y=0.95,
                xref="paper", yref="paper",
                text=f"Average: {avg_distance:.0f} km<br>Median: {median_distance:.0f} km<br>Max: {max_distance:.0f} km",
                showarrow=False,
                font=dict(size=12),
                bgcolor="black",
                bordercolor="white",
                borderwidth=1
            )
            
            # Complete progress
            progress_bar.progress(1.0)
            
            # Display the histogram
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Add a table with distance statistics by continent or region
            if 'country' in valid_routes.columns:
                st.subheader("Distance Statistics by Country")
                
                # Group by country and calculate statistics
                country_stats = valid_routes.groupby('country').agg(
                    avg_distance=('distance_km', 'mean'),
                    median_distance=('distance_km', 'median'),
                    max_distance=('distance_km', 'max'),
                    min_distance=('distance_km', 'min'),
                    route_count=('distance_km', 'count')
                ).reset_index().sort_values('route_count', ascending=False)
                
                # Format the distances
                country_stats['avg_distance'] = country_stats['avg_distance'].round(0).astype(int)
                country_stats['median_distance'] = country_stats['median_distance'].round(0).astype(int)
                country_stats['max_distance'] = country_stats['max_distance'].round(0).astype(int)
                country_stats['min_distance'] = country_stats['min_distance'].round(0).astype(int)
                
                # Display the table
                st.dataframe(country_stats.head(20), use_container_width=True)
            
            # Display connectivity statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg. Outbound Routes", f"{connectivity_df['outbound_routes'].mean():.1f}")
            with col2:
                st.metric("Avg. Inbound Routes", f"{connectivity_df['inbound_routes'].mean():.1f}")
            with col3:
                st.metric("Max Total Connectivity", int(connectivity_df['total_connectivity'].max()))
            
            # Display the connectivity data table
            st.subheader("Airport Connectivity Data")
            st.dataframe(
                connectivity_df[[
                    'name', 'city', 'country', 'iata', 
                    'outbound_routes', 'inbound_routes', 'total_connectivity', 'connectivity_ratio'
                ]].sort_values('total_connectivity', ascending=False),
                use_container_width=True
            )
            
            # Calculate and display network statistics
            st.subheader("Network Statistics")
            
            # Number of airports with at least one route
            connected_airports = len(connectivity_df[connectivity_df['total_connectivity'] > 0])
            
            # Percentage of airports that are connected
            connected_percentage = (connected_airports / len(connectivity_df)) * 100
            
            # Average degree (average number of connections per airport)
            avg_degree = connectivity_df['total_connectivity'].mean()
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Connected Airports", connected_airports)
            with col2:
                st.metric("Connected Percentage", f"{connected_percentage:.1f}%")
            with col3:
                st.metric("Average Degree", f"{avg_degree:.1f}")
    
    # Footer
    st.markdown("---")
    st.markdown("Data source: [OpenFlights](https://openflights.org)") 