import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Coral Reef Monitoring Dashboard",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-moderate {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Generate sample data similar to your North Coral Gardens dataset"""
    np.random.seed(42)
    
    # Create date range for 2024
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    n_points = len(dates)
    
    # Generate realistic oceanographic data
    # Base temperatures with seasonal variation
    day_of_year = dates.dayofyear
    seasonal_temp = 28.5 + 2 * np.sin(2 * np.pi * day_of_year / 365.25)
    
    # Add noise and anomalies
    temp_noise = np.random.normal(0, 0.3, n_points)
    
    # Create bleaching event in February
    bleaching_event = np.where(
        (dates.month == 2) & (dates.day.isin([10, 11, 12, 13, 14, 15])),
        np.random.normal(2.5, 0.5, n_points),
        0
    )
    
    data = {
        'timestamp': dates,
        'satellite_temperature_noaa': seasonal_temp + temp_noise,
        'sst_anomaly_noaa': np.random.normal(0.6, 0.8, n_points) + bleaching_event,
        'dhw_noaa': np.maximum(0, np.random.exponential(0.8, n_points) + bleaching_event * 2),
        'top_temperature_spotter': seasonal_temp + temp_noise + np.random.normal(0, 0.2, n_points),
        'bottom_temperature_spotter': seasonal_temp + temp_noise - 0.15 + np.random.normal(0, 0.15, n_points),
        'significant_wave_height_spotter': np.random.gamma(2, 0.05, n_points),
        'wind_speed_spotter': np.random.gamma(3, 2, n_points),
        'wind_direction_spotter': np.random.uniform(0, 360, n_points)
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

@st.cache_data
def calculate_risk_metrics(df):
    """Calculate coral bleaching risk metrics"""
    # Risk thresholds
    sst_threshold = df['sst_anomaly_noaa'].mean() + 2 * df['sst_anomaly_noaa'].std()
    dhw_threshold = 4.0  # Standard DHW threshold for bleaching
    
    # Risk flags
    df['temp_stress'] = df['sst_anomaly_noaa'] > sst_threshold
    df['bleaching_risk'] = df['dhw_noaa'] > dhw_threshold
    
    # Overall risk score (0-100)
    df['risk_score'] = (
        (df['sst_anomaly_noaa'] / df['sst_anomaly_noaa'].max()) * 40 +
        (df['dhw_noaa'] / df['dhw_noaa'].max()) * 60
    ) * 100
    
    return df

def create_risk_clusters(df):
    """Create risk clusters using K-means"""
    features = ['sst_anomaly_noaa', 'dhw_noaa', 'top_temperature_spotter', 'bottom_temperature_spotter']
    df_features = df[features].dropna()
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Map clusters to risk levels
    risk_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
    df_features['risk_level'] = [risk_mapping[c] for c in clusters]
    
    return df_features

def main():
    # Header
    st.markdown('<div class="main-header">🪸 Coral Reef Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Real-time coral reef health monitoring using SST and DHW data**")
    
    # Load data
    df = load_sample_data()
    df = calculate_risk_metrics(df)
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Site selector (for future multi-site support)
    site_options = ["North Coral Gardens", "Palmyra Atoll", "Great Barrier Reef"]
    selected_site = st.sidebar.selectbox("Select Reef Site", site_options)
    
    # Date range selector
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df.index.min().date(), df.index.max().date()),
        min_value=df.index.min().date(),
        max_value=df.index.max().date()
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df.loc[start_date:end_date]
    else:
        df_filtered = df
    
    # Current status metrics
    st.header("🚨 Current Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_sst = df_filtered['sst_anomaly_noaa'].iloc[-1]
        st.metric(
            "SST Anomaly",
            f"{current_sst:.2f}°C",
            delta=f"{current_sst - df_filtered['sst_anomaly_noaa'].iloc[-48]:.2f}°C (48h)"
        )
    
    with col2:
        current_dhw = df_filtered['dhw_noaa'].iloc[-1]
        st.metric(
            "Degree Heating Weeks",
            f"{current_dhw:.1f}",
            delta=f"{current_dhw - df_filtered['dhw_noaa'].iloc[-48]:.1f} (48h)"
        )
    
    with col3:
        current_risk = df_filtered['risk_score'].iloc[-1]
        st.metric(
            "Risk Score",
            f"{current_risk:.0f}/100",
            delta=f"{current_risk - df_filtered['risk_score'].iloc[-48]:.0f} (48h)"
        )
    
    with col4:
        high_risk_hours = len(df_filtered[df_filtered['risk_score'] > 70])
        st.metric(
            "High Risk Hours",
            f"{high_risk_hours}",
            delta=f"in selected period"
        )
    
    # Risk alert
    if current_risk > 70:
        st.markdown('<div class="alert-high">🚨 <strong>HIGH BLEACHING RISK</strong> - Immediate monitoring recommended</div>', unsafe_allow_html=True)
    elif current_risk > 40:
        st.markdown('<div class="alert-moderate">⚠️ <strong>MODERATE RISK</strong> - Enhanced monitoring advised</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-low">✅ <strong>LOW RISK</strong> - Normal conditions</div>', unsafe_allow_html=True)
    
    # Main visualizations
    st.header("📈 Environmental Trends")
    
    # Temperature trends
    fig_temp = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Temperature Trends', 'Thermal Stress Indicators'),
        vertical_spacing=0.1
    )
    
    # Temperature subplot
    fig_temp.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered['top_temperature_spotter'],
                  name='Top Temperature', line=dict(color='red')),
        row=1, col=1
    )
    fig_temp.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered['bottom_temperature_spotter'],
                  name='Bottom Temperature', line=dict(color='blue')),
        row=1, col=1
    )
    fig_temp.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered['satellite_temperature_noaa'],
                  name='Satellite Temperature', line=dict(color='green', dash='dash')),
        row=1, col=1
    )
    
    # Stress indicators subplot
    fig_temp.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered['sst_anomaly_noaa'],
                  name='SST Anomaly', line=dict(color='orange')),
        row=2, col=1
    )
    fig_temp.add_trace(
        go.Scatter(x=df_filtered.index, y=df_filtered['dhw_noaa'],
                  name='DHW', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Add bleaching threshold line
    fig_temp.add_hline(y=4.0, line_dash="dash", line_color="red", 
                      annotation_text="Bleaching Threshold", row=2, col=1)
    
    fig_temp.update_layout(height=600, showlegend=True)
    fig_temp.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig_temp.update_yaxes(title_text="Anomaly/DHW", row=2, col=1)
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Risk analysis
    st.header("🎯 Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score over time
        fig_risk = px.line(df_filtered.reset_index(), x='timestamp', y='risk_score',
                          title='Coral Bleaching Risk Score Over Time')
        fig_risk.add_hline(y=70, line_dash="dash", line_color="red", 
                          annotation_text="High Risk")
        fig_risk.add_hline(y=40, line_dash="dash", line_color="orange", 
                          annotation_text="Moderate Risk")
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Risk distribution
        risk_counts = pd.cut(df_filtered['risk_score'], 
                           bins=[0, 40, 70, 100], 
                           labels=['Low', 'Moderate', 'High']).value_counts()
        
        fig_dist = px.pie(values=risk_counts.values, names=risk_counts.index,
                         title='Risk Level Distribution',
                         color_discrete_map={'Low': 'green', 'Moderate': 'orange', 'High': 'red'})
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation analysis
    st.header("🔗 Environmental Correlations")
    
    correlation_vars = ['sst_anomaly_noaa', 'dhw_noaa', 'top_temperature_spotter', 
                       'bottom_temperature_spotter', 'significant_wave_height_spotter', 
                       'wind_speed_spotter']
    
    corr_matrix = df_filtered[correlation_vars].corr()
    
    fig_corr = px.imshow(corr_matrix, 
                        title='Correlation Matrix of Environmental Variables',
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Extreme events detection
    st.header("⚡ Extreme Events")
    
    # Detect extreme SST anomalies
    sst_threshold = df_filtered['sst_anomaly_noaa'].mean() + 2 * df_filtered['sst_anomaly_noaa'].std()
    extreme_events = df_filtered[df_filtered['sst_anomaly_noaa'] > sst_threshold]
    
    if len(extreme_events) > 0:
        st.write(f"**{len(extreme_events)} extreme temperature events detected:**")
        
        fig_extreme = px.scatter(df_filtered.reset_index(), x='timestamp', y='sst_anomaly_noaa',
                               title='SST Anomalies with Extreme Events Highlighted')
        fig_extreme.add_scatter(x=extreme_events.index, y=extreme_events['sst_anomaly_noaa'],
                              mode='markers', marker=dict(color='red', size=10),
                              name='Extreme Events')
        fig_extreme.add_hline(y=sst_threshold, line_dash="dash", line_color="red",
                            annotation_text="Extreme Event Threshold")
        st.plotly_chart(fig_extreme, use_container_width=True)
        
        # Show extreme events table
        st.subheader("Extreme Events Details")
        extreme_summary = extreme_events[['sst_anomaly_noaa', 'dhw_noaa', 'risk_score']].round(2)
        st.dataframe(extreme_summary)
    else:
        st.write("No extreme temperature events detected in the selected time period.")
    
    # Data export
    st.header("💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Filtered Data"):
            csv = df_filtered.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f'coral_monitoring_{selected_site}_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
    
    with col2:
        if st.button("Generate Report"):
            st.write("📋 **Monitoring Report Summary:**")
            st.write(f"- Site: {selected_site}")
            st.write(f"- Period: {date_range[0]} to {date_range[1]}")
            st.write(f"- Current Risk Level: {current_risk:.0f}/100")
            st.write(f"- Extreme Events: {len(extreme_events)}")
            st.write(f"- Max DHW: {df_filtered['dhw_noaa'].max():.1f}")
            st.write(f"- Max SST Anomaly: {df_filtered['sst_anomaly_noaa'].max():.2f}°C")
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Sources:** NOAA Coral Reef Watch, Aqualink • **Last Updated:** Real-time")

if __name__ == "__main__":
    main()
