import streamlit as st
from .pandas import pd 
import os
import numpy as np
import plotly.express as px
import time
import logging
from .data import get_data

# Configure logging for UI
logger = logging.getLogger('openflights-ui')

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
            
            # Create a scatter plot of outbound vs inbound routes
           
            
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