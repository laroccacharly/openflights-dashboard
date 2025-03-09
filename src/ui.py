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

# ===== Utility Functions =====

def haversine_distance(lat1, lon1, lat2, lon2):

    
    # Calculate the distance if not in cache
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    distance = c * r
    return distance

# ===== Data Loading Functions =====

def load_data(st):
    """Load the OpenFlights data if not already in session state"""
    if 'airports_df' not in st.session_state or 'routes_df' not in st.session_state:
        with st.spinner("Loading OpenFlights data..."):
            try:
                st.session_state.airports_df, st.session_state.routes_df = get_data()
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.stop()
    
    return st.session_state.airports_df, st.session_state.routes_df

# ===== Filter Functions =====

def apply_country_filter(airports_df, selected_countries):
    """Filter airports dataframe by selected countries"""
    if not selected_countries:
        return airports_df
    return airports_df[airports_df['country'].isin(selected_countries)]

def get_valid_map_data(airports_df, sampling_ratio=1.0):
    """Filter out invalid coordinates and apply sampling for better map performance"""
    # Filter out invalid coordinates
    map_data = airports_df[
        (airports_df['latitude'].notna()) & 
        (airports_df['longitude'].notna()) &
        (airports_df['latitude'] >= -90) & 
        (airports_df['latitude'] <= 90) &
        (airports_df['longitude'] >= -180) & 
        (airports_df['longitude'] <= 180)
    ].copy()

    # Apply sampling for better performance
    if not map_data.empty and len(map_data) > 10:
        # Calculate sample size based on the ratio
        sample_size = max(10, int(len(map_data) * sampling_ratio))
        # Sample the data
        map_data = map_data.sample(n=sample_size, random_state=42)
    
    return map_data

# ===== Computation Functions =====

def calculate_connectivity_stats(airports_df, routes_df):
    """Calculate airport connectivity statistics"""
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
    
    return connectivity_df

def prepare_route_distances(routes_df, airports_df):
    """Calculate distances for each route"""
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
    
    # Filter out routes with missing coordinates
    valid_routes = route_distances_df.dropna(subset=['source_lat', 'source_lon', 'dest_lat', 'dest_lon'])
    
    # Calculate distances for each route
    valid_routes['distance_km'] = valid_routes.apply(
        lambda row: haversine_distance(
            row['source_lat'], row['source_lon'], 
            row['dest_lat'], row['dest_lon']
        ),
        axis=1
    )
    
    return valid_routes

def calculate_country_distance_stats(valid_routes):
    """Calculate distance statistics by country"""
    # Group by country and calculate statistics
    if 'country' not in valid_routes.columns:
        return None
        
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
    
    return country_stats

def calculate_network_stats(connectivity_df):
    """Calculate network statistics"""
    # Number of airports with at least one route
    connected_airports = len(connectivity_df[connectivity_df['total_connectivity'] > 0])
    
    # Percentage of airports that are connected
    connected_percentage = (connected_airports / len(connectivity_df)) * 100
    
    # Average degree (average number of connections per airport)
    avg_degree = connectivity_df['total_connectivity'].mean()
    
    return {
        'connected_airports': connected_airports,
        'connected_percentage': connected_percentage,
        'avg_degree': avg_degree
    }

def time_operation(operation_name, func, *args, **kwargs):
    """Time an operation and store the result in session state"""
    # Initialize timing dictionary if not exists
    if 'execution_times' not in st.session_state:
        st.session_state.execution_times = {}
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    # Store the execution time
    st.session_state.execution_times[operation_name] = execution_time
    logger.info(f"Operation '{operation_name}' completed in {execution_time:.2f} seconds")
    
    return result, execution_time

# ===== UI Components =====

def setup_page_config():
    """Set up the Streamlit page configuration"""
    st.set_page_config(
        page_title="OpenFlights Dashboard",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def display_header():
    """Display the main title"""
    st.title("✈️ OpenFlights Dashboard")

def setup_sidebar_filters(st, airports_df):
    """Set up the sidebar filters and return the filter values"""
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
    
    return sampling_ratio, selected_countries

def display_basic_metrics(filtered_airports, routes_df):
    """Display basic metrics in three columns"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Airports", len(filtered_airports))
    with col2:
        st.metric("Total Countries", filtered_airports['country'].nunique())
    with col3:
        st.metric("Total Routes", len(routes_df))

def create_airports_map(map_data, selected_countries, sampling_ratio, filtered_airports_count):
    """Create and display the airports map"""
    st.subheader("Airport Locations")
    
    if map_data.empty:
        st.info("No airports to display on the map with the current filters.")
        return
    
    # Show sampling info if applicable
    if len(map_data) < filtered_airports_count:
        st.info(f"Displaying {len(map_data)} airports ({sampling_ratio:.0%} of {filtered_airports_count} filtered airports) for better performance.")
    
    # Create a larger map
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
    
    # Update layout to make the map larger
    fig.update_layout(
        height=800,
        margin={"r":0,"t":30,"l":0,"b":0}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_connectivity_chart(connectivity_df, selected_countries):
    """Display the top connected airports chart"""
    st.subheader("Top 20 Most Connected Airports")
    
    # Filter for selected countries if any
    if selected_countries:
        connectivity_df = connectivity_df[connectivity_df['country'].isin(selected_countries)]
    
    # Get top 20 most connected airports
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

def display_route_distance_histogram(valid_routes):
    """Display histogram of route distances"""
    
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
    
    # Display the histogram
    st.plotly_chart(fig_dist, use_container_width=True)

def display_degree_histogram(connectivity_df):
    """Display histogram of airport connectivity degrees (in + out)"""
    
    # Create a histogram of degree distribution
    fig_degree = px.histogram(
        connectivity_df,
        x='total_connectivity',
        nbins=50,
        title="Distribution of Airport Connectivity Degrees (In + Out)",
        labels={'total_connectivity': 'Degree (Total Routes)', 'count': 'Number of Airports'},
        marginal="box"  # Add a box plot on the marginal axis
    )
    
    # Add some statistics as annotations
    avg_degree = connectivity_df['total_connectivity'].mean()
    median_degree = connectivity_df['total_connectivity'].median()
    max_degree = connectivity_df['total_connectivity'].max()
    
    fig_degree.add_annotation(
        x=0.95, y=0.95,
        xref="paper", yref="paper",
        text=f"Average: {avg_degree:.1f}<br>Median: {median_degree:.0f}<br>Max: {max_degree:.0f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="black",
        bordercolor="white",
        borderwidth=1
    )
    
    # Display the histogram
    st.plotly_chart(fig_degree, use_container_width=True)

def display_connectivity_metrics(connectivity_df):
    """Display connectivity metrics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg. Outbound Routes", f"{connectivity_df['outbound_routes'].mean():.1f}")
    with col2:
        st.metric("Avg. Inbound Routes", f"{connectivity_df['inbound_routes'].mean():.1f}")
    with col3:
        st.metric("Max Total Connectivity", int(connectivity_df['total_connectivity'].max()))

def display_connectivity_table(connectivity_df):
    """Display the connectivity data table"""
    st.subheader("Airport Connectivity Data")
    st.dataframe(
        connectivity_df[[
            'name', 'city', 'country', 'iata', 
            'outbound_routes', 'inbound_routes', 'total_connectivity', 'connectivity_ratio'
        ]].sort_values('total_connectivity', ascending=False),
        use_container_width=True
    )

def display_network_stats(network_stats):
    """Display network statistics"""
    st.subheader("Network Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Connected Airports", network_stats['connected_airports'])
    with col2:
        st.metric("Connected Percentage", f"{network_stats['connected_percentage']:.1f}%")
    with col3:
        st.metric("Average Degree", f"{network_stats['avg_degree']:.1f}")

def display_country_stats_table(country_stats):
    """Display the country statistics table"""
    if country_stats is not None:
        st.subheader("Distance Statistics by Country")
        st.dataframe(country_stats.head(20), use_container_width=True)

def display_footer():
    """Display the footer with data source information"""
    st.markdown("---")
    st.markdown("Data source: [OpenFlights](https://openflights.org)")

def display_execution_times():
    """Display execution times for all tracked computations"""
    st.subheader("Operation Execution Times")
    
    if 'execution_times' not in st.session_state or not st.session_state.execution_times:
        st.info("No operations have been timed yet.")
        return
    
    # Create a dataframe from the execution times
    times_df = pd.DataFrame({
        'Operation': list(st.session_state.execution_times.keys()),
        'Execution Time (seconds)': list(st.session_state.execution_times.values())
    })
    
    # Sort by execution time (descending)
    times_df = times_df.sort_values('Execution Time (seconds)', ascending=False)
    
    # Also display as a table
    st.dataframe(times_df, use_container_width=True)
    
    # Add option to clear timing data
    if st.button("Clear Timing Data"):
        st.session_state.execution_times = {}
        st.experimental_rerun()

# ===== Main UI Function =====

def run_routes_analysis(st, airports_df, routes_df, selected_countries):
    """Run the routes analysis and display the results"""
    with st.spinner("Analyzing airport routes..."):
        # Calculate connectivity statistics
        connectivity_df, conn_time = time_operation(
            "Calculate Connectivity Statistics",
            calculate_connectivity_stats,
            airports_df, routes_df
        )
        
        # Filter for the selected countries if any
        filtered_connectivity_df = connectivity_df
        if selected_countries:
            filtered_connectivity_df, filter_time = time_operation(
                "Filter Connectivity by Countries",
                lambda df, countries: df[df['country'].isin(countries)],
                connectivity_df, selected_countries
            )
        

        # Calculate route distances
        progress_bar = st.progress(0)

        # Display the top 20 most connected airports chart
        display_connectivity_chart(filtered_connectivity_df, selected_countries)
        

        
        # Prepare route distances data
        progress_bar.progress(0.3)
        valid_routes, dist_time = time_operation(
            "Prepare Route Distances",
            prepare_route_distances,
            routes_df, airports_df
        )
        progress_bar.progress(0.8)
        
        # Display the route distance histogram
        display_route_distance_histogram(valid_routes)
        
        # Display the degree histogram
        display_degree_histogram(filtered_connectivity_df)
        
        progress_bar.progress(0.9)
        
        # Calculate and display country statistics
        country_stats, country_time = time_operation(
            "Calculate Country Distance Statistics",
            calculate_country_distance_stats,
            valid_routes
        )
        display_country_stats_table(country_stats)

        # Calculate and display network statistics
        network_stats, network_time = time_operation(
            "Calculate Network Statistics",
            calculate_network_stats,
            filtered_connectivity_df
        )
        display_network_stats(network_stats)

        # Display the connectivity data table
        display_connectivity_table(filtered_connectivity_df)
        
        progress_bar.progress(1.0)
        
        # Display execution times
        display_execution_times()

def run_ui():
    """Main function to run the Streamlit UI"""
    # Set up page configuration
    setup_page_config()
    
    # Display header
    display_header()
    
    # Load data
    airports_df, routes_df = load_data(st)
    
    # Set up sidebar filters
    sampling_ratio, selected_countries = setup_sidebar_filters(st, airports_df)
    
    # Apply filters
    filtered_airports = apply_country_filter(airports_df, selected_countries)
        
    # Display basic metrics
    display_basic_metrics(filtered_airports, routes_df)
    
    # Get valid map data with sampling
    map_data = get_valid_map_data(filtered_airports, sampling_ratio)
    
    # Create and display airports map
    create_airports_map(map_data, selected_countries, sampling_ratio, len(filtered_airports))
    
    # Airport Routes Analysis
    st.header("Airport Routes Analysis")
    st.write("This analysis joins the airports and routes datasets to analyze airport connectivity.")
    
    # Add a button to run the analysis (to avoid running it automatically every time)
    if st.button("Run Routes Analysis"):
        run_routes_analysis(st, filtered_airports, routes_df, selected_countries)
    
    # Display footer
    display_footer() 