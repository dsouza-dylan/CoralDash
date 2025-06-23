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
    """Load meteorite data from NASA's Open Data Portal with fallback options"""

    # Try multiple data sources
    urls = [
        "https://data.nasa.gov/resource/y77d-th95.json?$limit=3000",
        "https://data.nasa.gov/api/views/y77d-th95/rows.json?accessType=DOWNLOAD"
    ]

    for i, url in enumerate(urls):
        try:
            st.info(f"Trying data source {i+1}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            if i == 0:  # JSON format
                data = response.json()
                df = pd.DataFrame(data)
            else:  # SODA format
                json_data = response.json()
                df = pd.DataFrame(json_data['data'], columns=[col['name'] for col in json_data['meta']['view']['columns']])

            if len(df) == 0:
                continue

            # Clean and process the data
            required_columns = ['reclat', 'reclong']
            if not all(col in df.columns for col in required_columns):
                # Try alternative column names
                lat_cols = [col for col in df.columns if 'lat' in col.lower()]
                lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower()]
                if lat_cols and lon_cols:
                    df = df.rename(columns={lat_cols[0]: 'reclat', lon_cols[0]: 'reclong'})
                else:
                    continue

            df = df.dropna(subset=['reclat', 'reclong'])
            df['reclat'] = pd.to_numeric(df['reclat'], errors='coerce')
            df['reclong'] = pd.to_numeric(df['reclong'], errors='coerce')

            # Handle mass and year columns
            if 'mass' in df.columns:
                df['mass'] = pd.to_numeric(df['mass'], errors='coerce')
            else:
                df['mass'] = np.nan

            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
            else:
                df['year'] = np.nan

            # Add name column if missing
            if 'name' not in df.columns:
                df['name'] = 'Meteorite_' + df.index.astype(str)

            # Remove invalid coordinates
            df = df[(df['reclat'].between(-90, 90)) & (df['reclong'].between(-180, 180))]

            if len(df) == 0:
                continue

            # Create geometry column for GeoPandas
            geometry = [Point(xy) for xy in zip(df['reclong'], df['reclat'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

            st.success(f"Successfully loaded {len(gdf)} meteorite records!")
            return gdf

        except requests.exceptions.RequestException as e:
            st.warning(f"Network error with source {i+1}: {e}")
            continue
        except ValueError as e:
            st.warning(f"Data parsing error with source {i+1}: {e}")
            continue
        except Exception as e:
            st.warning(f"Unexpected error with source {i+1}: {e}")
            continue

    # If all sources fail, create sample data
    st.warning("All data sources failed. Creating sample data for demonstration...")
    return create_sample_data()

def create_sample_data():
    """Create sample meteorite data for demonstration"""
    np.random.seed(42)
    n_samples = 500

    # Create realistic distribution with some clustering
    lats = []
    lons = []
    masses = []
    names = []
    years = []

    # Add some clusters (representing strewn fields)
    cluster_centers = [
        (34.0, -111.0),  # Arizona
        (-80.0, 155.0),  # Antarctica
        (20.0, 10.0),    # Sahara
        (-25.0, 135.0),  # Australia
    ]

    for i, (center_lat, center_lon) in enumerate(cluster_centers):
        n_cluster = n_samples // len(cluster_centers)
        cluster_lats = np.random.normal(center_lat, 2, n_cluster)
        cluster_lons = np.random.normal(center_lon, 2, n_cluster)
        cluster_masses = np.random.lognormal(3, 2, n_cluster)
        cluster_years = np.random.randint(1950, 2023, n_cluster)

        lats.extend(cluster_lats)
        lons.extend(cluster_lons)
        masses.extend(cluster_masses)
        years.extend(cluster_years)
        names.extend([f"Sample_{i}_{j}" for j in range(n_cluster)])

    # Clip coordinates to valid ranges
    lats = np.clip(lats, -89, 89)
    lons = np.clip(lons, -179, 179)

    df = pd.DataFrame({
        'name': names,
        'reclat': lats,
        'reclong': lons,
        'mass': masses,
        'year': years,
        'recclass': ['L6'] * len(names)  # Most common type
    })

    geometry = [Point(xy) for xy in zip(df['reclong'], df['reclat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    return gdf

@st.cache_data
def load_world_data():
    """Load world countries data for context"""
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        return world
    except:
        return None

@st.cache_data
def perform_clustering_analysis(gdf, eps_km=200, min_samples=5):
    """Perform DBSCAN clustering on meteorite locations"""
    # Sample data for performance if too large
    if len(gdf) > 2000:
        gdf_sample = gdf.sample(n=2000, random_state=42)
    else:
        gdf_sample = gdf

    # Convert to Web Mercator for distance calculations
    gdf_proj = gdf_sample.to_crs('EPSG:3857')

    # Extract coordinates
    coords = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])

    # Perform DBSCAN clustering (eps in meters, converted from km)
    eps_meters = eps_km * 1000
    clustering = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(coords)

    # Add cluster labels to sampled dataframe
    gdf_clustered = gdf_sample.copy()
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

def create_main_map(gdf, world_gdf=None, max_points=500):
    """Create main meteorite distribution map using Folium"""
    # Sample data for performance
    if len(gdf) > max_points:
        # Prioritize larger meteorites in sampling
        gdf_large = gdf[gdf['mass'].fillna(0) > 1000]
        gdf_small = gdf[gdf['mass'].fillna(0) <= 1000]

        n_large = min(len(gdf_large), max_points // 3)
        n_small = max_points - n_large

        if n_large > 0:
            sample_large = gdf_large.sample(n=n_large, random_state=42) if len(gdf_large) > n_large else gdf_large
        else:
            sample_large = pd.DataFrame()

        if n_small > 0 and len(gdf_small) > 0:
            sample_small = gdf_small.sample(n=min(n_small, len(gdf_small)), random_state=42)
        else:
            sample_small = pd.DataFrame()

        gdf_display = pd.concat([sample_large, sample_small], ignore_index=True)
    else:
        gdf_display = gdf

    # Create base map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')

    # Add world boundaries if available (simplified)
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
    for idx, row in gdf_display.iterrows():
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
