import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
import requests
import json
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌠 Meteorite Impact Analysis",
    page_icon="🌠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_meteorite_data():
    """Load meteorite data from NASA's Open Data Portal"""
    try:
        # NASA Meteorite Landings Dataset
        url = "https://data.nasa.gov/resource/y77d-th95.json"
        response = requests.get(url, params={"$limit": 50000})
        data = response.json()
        df = pd.DataFrame(data)
        
        # Clean and process the data
        df = df.dropna(subset=['reclat', 'reclong'])
        df['reclat'] = pd.to_numeric(df['reclat'], errors='coerce')
        df['reclong'] = pd.to_numeric(df['reclong'], errors='coerce')
        df['mass'] = pd.to_numeric(df['mass'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Remove invalid coordinates
        df = df[(df['reclat'].between(-90, 90)) & (df['reclong'].between(-180, 180))]
        
        # Create geometry column for GeoPandas
        geometry = [Point(xy) for xy in zip(df['reclong'], df['reclat'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_world_data():
    """Load world countries data for context"""
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        return world
    except:
        return None

def perform_clustering_analysis(gdf, eps_km=200, min_samples=5):
    """Perform DBSCAN clustering on meteorite locations"""
    # Convert to Web Mercator for distance calculations
    gdf_proj = gdf.to_crs('EPSG:3857')
    
    # Extract coordinates
    coords = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
    
    # Perform DBSCAN clustering (eps in meters, converted from km)
    eps_meters = eps_km * 1000
    clustering = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(coords)
    
    # Add cluster labels to original dataframe
    gdf_clustered = gdf.copy()
    gdf_clustered['cluster'] = clustering.labels_
    
    return gdf_clustered

def calculate_density_statistics(gdf):
    """Calculate meteorite density statistics by continent/region"""
    # Simple continent assignment based on coordinates
    def assign_continent(lat, lon):
        if lat > 70:
            return "Arctic"
        elif lat < -60:
            return "Antarctica"
        elif -35 < lat < 70 and -10 < lon < 60:
            return "Europe/Asia"
        elif -35 < lat < 40 and -20 < lon < 50:
            return "Africa"
        elif 10 < lat < 70 and -170 < lon < -50:
            return "North America"
        elif -60 < lat < 15 and -85 < lon < -35:
            return "South America"
        elif -50 < lat < -10 and 110 < lon < 180:
            return "Australia/Oceania"
        else:
            return "Other"
    
    gdf['continent'] = gdf.apply(lambda row: assign_continent(row['reclat'], row['reclong']), axis=1)
    
    continent_stats = gdf.groupby('continent').agg({
        'name': 'count',
        'mass': ['mean', 'median', 'sum'],
        'year': ['min', 'max']
    }).round(2)
    
    return continent_stats

def create_main_map(gdf, world_gdf=None):
    """Create main meteorite distribution map using Folium"""
    # Create base map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
    
    # Add world boundaries if available
    if world_gdf is not None:
        folium.GeoJson(
            world_gdf.to_json(),
            style_function=lambda x: {
                'fillColor': '#f0f0f0',
                'color': '#999999',
                'weight': 1,
                'fillOpacity': 0.3
            }
        ).add_to(m)
    
    # Add meteorite points
    for idx, row in gdf.iterrows():
        if pd.notna(row['mass']):
            # Size based on mass (log scale)
            size = max(3, min(15, np.log10(float(row['mass']) + 1) * 2))
            color = 'red' if float(row['mass']) > 10000 else 'orange' if float(row['mass']) > 1000 else 'yellow'
        else:
            size = 3
            color = 'blue'
        
        popup_text = f"""
        <b>{row['name']}</b><br>
        Year: {row.get('year', 'Unknown')}<br>
        Mass: {row.get('mass', 'Unknown')} g<br>
        Class: {row.get('recclass', 'Unknown')}<br>
        Location: ({row['reclat']:.2f}, {row['reclong']:.2f})
        """
        
        folium.CircleMarker(
            location=[row['reclat'], row['reclong']],
            radius=size,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def create_cluster_map(gdf_clustered):
    """Create cluster visualization map"""
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Color palette for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Add clustered points
    for idx, row in gdf_clustered.iterrows():
        if row['cluster'] == -1:
            color = 'gray'  # Noise points
            popup_text = f"<b>{row['name']}</b><br>Cluster: Noise<br>Year: {row.get('year', 'Unknown')}"
        else:
            color = colors[row['cluster'] % len(colors)]
            popup_text = f"<b>{row['name']}</b><br>Cluster: {row['cluster']}<br>Year: {row.get('year', 'Unknown')}"
        
        folium.CircleMarker(
            location=[row['reclat'], row['reclong']],
            radius=5,
            popup=folium.Popup(popup_text, max_width=200),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def create_temporal_analysis(gdf):
    """Create temporal analysis visualizations"""
    # Filter for valid years
    gdf_temporal = gdf[gdf['year'].notna() & (gdf['year'] > 1800) & (gdf['year'] <= 2024)]
    
    if len(gdf_temporal) == 0:
        return None, None
    
    # Discoveries over time
    yearly_counts = gdf_temporal.groupby('year').size().reset_index(name='count')
    yearly_counts['cumulative'] = yearly_counts['count'].cumsum()
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Meteorite Discoveries Over Time', 'Cumulative Discoveries', 
                       'Mass Distribution by Decade', 'Discovery Locations by Era'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Annual discoveries
    fig.add_trace(
        go.Scatter(x=yearly_counts['year'], y=yearly_counts['count'], 
                  mode='lines+markers', name='Annual Discoveries'),
        row=1, col=1
    )
    
    # Cumulative discoveries
    fig.add_trace(
        go.Scatter(x=yearly_counts['year'], y=yearly_counts['cumulative'], 
                  mode='lines', name='Cumulative Discoveries', line=dict(color='red')),
        row=1, col=2
    )
    
    # Mass distribution by decade
    gdf_temporal['decade'] = (gdf_temporal['year'] // 10) * 10
    decade_mass = gdf_temporal[gdf_temporal['mass'].notna()]
    
    if len(decade_mass) > 0:
        fig.add_trace(
            go.Box(x=decade_mass['decade'], y=np.log10(decade_mass['mass'] + 1), 
                   name='Log Mass by Decade'),
            row=2, col=1
        )
    
    # Geographic distribution by era
    modern = gdf_temporal[gdf_temporal['year'] >= 1950]
    historical = gdf_temporal[gdf_temporal['year'] < 1950]
    
    fig.add_trace(
        go.Scattergeo(
            lat=historical['reclat'], lon=historical['reclong'],
            mode='markers', name='Pre-1950',
            marker=dict(size=3, color='blue', opacity=0.6)
        ), row=2, col=2
    )
    
    fig.add_trace(
        go.Scattergeo(
            lat=modern['reclat'], lon=modern['reclong'],
            mode='markers', name='1950+',
            marker=dict(size=3, color='red', opacity=0.6)
        ), row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Temporal Analysis of Meteorite Discoveries")
    fig.update_geos(projection_type="natural earth", row=2, col=2)
    
    return fig, yearly_counts

def main():
    st.markdown('<div class="main-header">🌠 Global Meteorite Impact Analysis</div>', unsafe_allow_html=True)
    st.markdown("### Uncovering spatial patterns and stories from meteorite landings across Earth")
    
    # Sidebar controls
    st.sidebar.title("Analysis Controls")
    
    # Load data
    with st.spinner("Loading meteorite data..."):
        gdf = load_meteorite_data()
        world_gdf = load_world_data()
    
    if gdf is None:
        st.error("Failed to load meteorite data. Please check your internet connection.")
        return
    
    # Basic statistics
    st.markdown('<div class="sub-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Meteorites", f"{len(gdf):,}")
    with col2:
        valid_mass = gdf[gdf['mass'].notna()]
        if len(valid_mass) > 0:
            st.metric("Avg Mass", f"{valid_mass['mass'].mean():.1f}g")
        else:
            st.metric("Avg Mass", "N/A")
    with col3:
        valid_years = gdf[gdf['year'].notna()]
        if len(valid_years) > 0:
            st.metric("Year Range", f"{int(valid_years['year'].min())}-{int(valid_years['year'].max())}")
        else:
            st.metric("Year Range", "N/A")
    with col4:
        countries_covered = len(gdf['reclong'].unique())  # Rough estimate
        st.metric("Global Coverage", f"{countries_covered} regions")
    
    # Main visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Global Distribution", "🎯 Cluster Analysis", "📈 Temporal Patterns", "🌍 Continental Analysis"])
    
    with tab1:
        st.markdown('<div class="sub-header">Global Meteorite Distribution</div>', unsafe_allow_html=True)
        
        # Mass filter
        mass_filter = st.slider("Minimum Mass (grams)", 0, 10000, 0, step=100)
        filtered_gdf = gdf[gdf['mass'].fillna(0) >= mass_filter] if mass_filter > 0 else gdf
        
        st.markdown(f"Showing {len(filtered_gdf):,} meteorites")
        
        # Create and display map
        main_map = create_main_map(filtered_gdf.head(1000), world_gdf)  # Limit for performance
        folium_static(main_map, width=1200, height=600)
        
        # Key insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Key Insights:**")
        st.markdown("""
        - **Desert Bias**: Many meteorites are found in arid regions (Sahara, Antarctica, Australia) where preservation is better
        - **Population Correlation**: Increased discoveries in populated areas reflect reporting bias rather than impact density
        - **Coastal Patterns**: Fewer discoveries over oceans due to collection challenges
        - **Size Distribution**: Larger meteorites (red dots) are relatively rare and often found in remote areas
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Meteorite Clustering Analysis</div>', unsafe_allow_html=True)
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        with col1:
            eps_km = st.slider("Cluster Radius (km)", 50, 1000, 200, step=50)
        with col2:
            min_samples = st.slider("Minimum Cluster Size", 2, 20, 5)
        
        # Perform clustering
        with st.spinner("Performing clustering analysis..."):
            gdf_clustered = perform_clustering_analysis(gdf, eps_km, min_samples)
        
        # Cluster statistics
        n_clusters = len(set(gdf_clustered['cluster'])) - (1 if -1 in gdf_clustered['cluster'].values else 0)
        n_noise = sum(gdf_clustered['cluster'] == -1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters Found", n_clusters)
        with col2:
            st.metric("Clustered Points", len(gdf_clustered) - n_noise)
        with col3:
            st.metric("Noise Points", n_noise)
        
        # Display cluster map
        cluster_map = create_cluster_map(gdf_clustered.head(1000))
        folium_static(cluster_map, width=1200, height=600)
        
        # Cluster insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Clustering Insights:**")
        st.markdown("""
        - **Strewn Fields**: Clusters often represent meteorite showers from single events
        - **Collection Hotspots**: Some clusters reflect intensive collection efforts in specific regions
        - **Geological Correlation**: Dense clusters in places like Antarctica's blue ice areas
        - **Historical Events**: Large clusters may represent well-documented historical falls
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Temporal Discovery Patterns</div>', unsafe_allow_html=True)
        
        # Create temporal analysis
        fig, yearly_data = create_temporal_analysis(gdf)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Temporal insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**Temporal Insights:**")
            st.markdown("""
            - **Modern Discovery Boom**: Exponential increase in discoveries since 1970s due to systematic searches
            - **Antarctica Expeditions**: Major spikes correspond to Antarctic meteorite collection programs
            - **Technology Impact**: Satellite imagery and GPS have revolutionized meteorite hunting
            - **Preservation Bias**: Older meteorites are underrepresented due to weathering and loss
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Insufficient temporal data for analysis")
    
    with tab4:
        st.markdown('<div class="sub-header">Continental Distribution Analysis</div>', unsafe_allow_html=True)
        
        # Calculate continental statistics
        continent_stats = calculate_density_statistics(gdf)
        
        # Display continental breakdown
        if not continent_stats.empty:
            st.dataframe(continent_stats, use_container_width=True)
            
            # Continental distribution pie chart
            continent_counts = gdf['continent'].value_counts()
            
            fig_pie = px.pie(
                values=continent_counts.values,
                names=continent_counts.index,
                title="Meteorite Distribution by Continent"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Continental insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**Continental Insights:**")
        st.markdown("""
        - **Antarctica Dominance**: Highest density due to ideal preservation conditions and systematic collection
        - **Desert Advantage**: Africa and Australia show high discovery rates in arid regions
        - **Ocean Gaps**: Vast majority of ocean impacts go unrecorded
        - **Population Effect**: Europe and North America show discovery patterns influenced by population density
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Final story section
    st.markdown('<div class="sub-header">🌟 The Story Meteorites Tell</div>', unsafe_allow_html=True)
    st.markdown("""
    The global distribution of meteorite discoveries reveals a fascinating interplay between cosmic bombardment, 
    terrestrial processes, and human activity. While Earth receives meteorites uniformly across its surface, 
    our ability to find and preserve them depends heavily on:
    
    **Environmental Factors:**
    - Arid climates that preserve meteorites for millennia
    - Ice fields that concentrate and preserve specimens
    - Geological stability that prevents burial
    
    **Human Factors:**
    - Population density affecting discovery likelihood
    - Scientific expeditions to remote but promising areas
    - Technological advances in detection and collection
    
    **Cosmic Insights:**
    - Clustering patterns revealing meteorite shower events
    - Size distributions showing asteroid breakup processes
    - Temporal patterns reflecting both cosmic events and human discovery efforts
    
    This analysis demonstrates that meteorite science is as much about understanding Earth's surface processes 
    and human exploration patterns as it is about cosmic visitors themselves.
    """)

if __name__ == "__main__":
    main()
