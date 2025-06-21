import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CoralDash - Coral Reef Monitoring Dashboard",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-low {
        background-color: #00aa00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data():
    """Generate comprehensive sample coral reef monitoring data"""
    np.random.seed(42)

    # Define reef locations with realistic coordinates
    reef_locations = {
        'Great Barrier Reef': {'lat': -16.2839, 'lon': 145.7781, 'region': 'Australia'},
        'Coral Triangle': {'lat': -8.5069, 'lon': 116.1537, 'region': 'Indonesia'},
        'Caribbean Reefs': {'lat': 18.2208, 'lon': -66.5901, 'region': 'Caribbean'},
        'Red Sea Reefs': {'lat': 25.7617, 'lon': 34.4658, 'region': 'Red Sea'},
        'Maldives Reefs': {'lat': 3.2028, 'lon': 73.2207, 'region': 'Indian Ocean'},
        'Hawaii Reefs': {'lat': 21.3099, 'lon': -157.8581, 'region': 'Pacific'},
        'Fiji Reefs': {'lat': -17.7134, 'lon': 178.0650, 'region': 'Pacific'},
        'Seychelles Reefs': {'lat': -4.6796, 'lon': 55.4920, 'region': 'Indian Ocean'},
        'Palmyra Atoll': {'lat': 5.8719, 'lon': -162.0864, 'region': 'Pacific'},
        'Bahamas Reefs': {'lat': 25.0343, 'lon': -77.3963, 'region': 'Caribbean'}
    }

    # Generate 2 years of daily data
    date_range = pd.date_range(start='2022-01-01', end='2024-06-20', freq='D')
    data = []

    for reef_name, location in reef_locations.items():
        for date in date_range:
            # Seasonal temperature variation
            day_of_year = date.timetuple().tm_yday
            seasonal_temp = 26 + 4 * np.sin(2 * np.pi * day_of_year / 365.25)

            # Add regional variations
            if location['region'] in ['Caribbean', 'Red Sea']:
                base_temp = seasonal_temp + 2
            elif location['region'] == 'Australia':
                base_temp = seasonal_temp - 1
            else:
                base_temp = seasonal_temp

            # Add random variation and occasional heat stress events
            temp_noise = np.random.normal(0, 1.5)

            # Simulate heat stress events (higher probability in summer months)
            if 150 < day_of_year < 250:  # Summer months
                heat_stress_prob = 0.15
            else:
                heat_stress_prob = 0.05

            if np.random.random() < heat_stress_prob:
                temp_noise += np.random.exponential(3)

            sst = base_temp + temp_noise

            # Calculate SST anomaly (compared to long-term average)
            long_term_avg = 26 + 4 * np.sin(2 * np.pi * day_of_year / 365.25)
            sst_anomaly = sst - long_term_avg

            # Calculate DHW (simplified - normally uses more complex calculation)
            if sst_anomaly > 1:
                dhw_contribution = sst_anomaly / 7  # Weekly contribution
            else:
                dhw_contribution = 0

            # Coral health metrics
            coral_cover = max(10, 85 - max(0, sst_anomaly - 2) * 15 + np.random.normal(0, 5))
            bleaching_severity = max(0, min(100, (sst_anomaly - 1) * 20 + np.random.normal(0, 10)))

            # Water quality parameters
            turbidity = max(0, 2 + np.random.exponential(1))
            ph = 8.1 + np.random.normal(0, 0.1)
            dissolved_oxygen = 7.5 + np.random.normal(0, 0.5)

            data.append({
                'reef_name': reef_name,
                'date': date,
                'latitude': location['lat'],
                'longitude': location['lon'],
                'region': location['region'],
                'sst': round(sst, 2),
                'sst_anomaly': round(sst_anomaly, 2),
                'dhw_contribution': round(dhw_contribution, 3),
                'coral_cover': round(coral_cover, 1),
                'bleaching_severity': round(bleaching_severity, 1),
                'turbidity': round(turbidity, 2),
                'ph': round(ph, 2),
                'dissolved_oxygen': round(dissolved_oxygen, 2)
            })

    df = pd.DataFrame(data)

    # Calculate rolling DHW (12-week rolling sum)
    df = df.sort_values(['reef_name', 'date'])
    df['dhw'] = df.groupby('reef_name')['dhw_contribution'].rolling(window=84, min_periods=1).sum().values
    df['dhw'] = df['dhw'].round(2)

    return df

@st.cache_data
def calculate_bleaching_risk(df):
    """Calculate bleaching risk based on DHW and other factors"""
    def risk_level(row):
        dhw = row['dhw']
        sst_anomaly = row['sst_anomaly']

        if dhw >= 8 or sst_anomaly >= 3:
            return 'High'
        elif dhw >= 4 or sst_anomaly >= 2:
            return 'Medium'
        else:
            return 'Low'

    df['risk_level'] = df.apply(risk_level, axis=1)
    return df

def create_risk_map(df, selected_date):
    """Create an interactive map showing coral reef locations and risk levels"""
    # Filter data for selected date
    date_data = df[df['date'] == selected_date].copy()

    # Create base map
    center_lat = date_data['latitude'].mean()
    center_lon = date_data['longitude'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles='OpenStreetMap'
    )

    # Color mapping for risk levels
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}

    for _, row in date_data.iterrows():
        color = color_map[row['risk_level']]

        popup_text = f"""
        <b>{row['reef_name']}</b><br>
        Region: {row['region']}<br>
        SST: {row['sst']}°C<br>
        SST Anomaly: {row['sst_anomaly']:+.1f}°C<br>
        DHW: {row['dhw']}<br>
        Risk Level: {row['risk_level']}<br>
        Coral Cover: {row['coral_cover']}%<br>
        Bleaching: {row['bleaching_severity']}%
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Bleaching Risk</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Low</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium</p>
    <p><i class="fa fa-circle" style="color:red"></i> High</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def create_time_series_plot(df, reef_name, metrics):
    """Create time series plots for selected reef and metrics"""
    reef_data = df[df['reef_name'] == reef_name].copy()

    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=metrics,
        vertical_spacing=0.08
    )

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(
                x=reef_data['date'],
                y=reef_data[metric],
                mode='lines',
                name=metric,
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            row=i+1, col=1
        )

        # Add threshold lines for certain metrics
        if metric == 'dhw':
            fig.add_hline(y=4, line_dash="dash", line_color="orange",
                         annotation_text="Medium Risk", row=i+1, col=1)
            fig.add_hline(y=8, line_dash="dash", line_color="red",
                         annotation_text="High Risk", row=i+1, col=1)
        elif metric == 'sst_anomaly':
            fig.add_hline(y=1, line_dash="dash", line_color="orange",
                         annotation_text="Stress Threshold", row=i+1, col=1)

    fig.update_layout(
        height=200 * len(metrics),
        title_text=f"Time Series Analysis - {reef_name}",
        showlegend=False
    )

    return fig

def build_prediction_model(df):
    """Build a machine learning model to predict bleaching severity"""
    # Prepare features
    features = ['sst', 'sst_anomaly', 'dhw', 'turbidity', 'ph', 'dissolved_oxygen']
    target = 'bleaching_severity'

    # Remove rows with missing values
    model_data = df[features + [target]].dropna()

    X = model_data[features]
    y = model_data[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, scaler, {'mse': mse, 'r2': r2}, feature_importance

def main():
    # Header
    st.markdown('<h1 class="main-header">🪸 CoralDash</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real-time Coral Reef Monitoring & Bleaching Risk Assessment</p>', unsafe_allow_html=True)

    # Load data
    with st.spinner('Loading coral reef monitoring data...'):
        df = generate_sample_data()
        df = calculate_bleaching_risk(df)

    # Sidebar
    st.sidebar.header("🔧 Dashboard Controls")

    # Date selection
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    selected_date = pd.to_datetime(selected_date)

    # Reef selection
    reef_options = sorted(df['reef_name'].unique())
    selected_reef = st.sidebar.selectbox(
        "Select Reef for Analysis",
        reef_options,
        index=reef_options.index('Great Barrier Reef')
    )

    # Metrics selection
    available_metrics = ['sst', 'sst_anomaly', 'dhw', 'coral_cover', 'bleaching_severity']
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics for Time Series",
        available_metrics,
        default=['sst_anomaly', 'dhw', 'bleaching_severity']
    )

    # Main dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Global Overview", "📊 Time Series Analysis", "🤖 Predictive Analytics", "📈 Correlation Analysis", "📋 Data Explorer"])

    with tab1:
        st.header("Global Coral Reef Risk Assessment")

        # Current date data
        current_data = df[df['date'] == selected_date]

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            high_risk_count = len(current_data[current_data['risk_level'] == 'High'])
            st.markdown(f'<div class="metric-card"><h3>{high_risk_count}</h3><p>High Risk Reefs</p></div>', unsafe_allow_html=True)

        with col2:
            avg_sst = current_data['sst'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_sst:.1f}°C</h3><p>Average SST</p></div>', unsafe_allow_html=True)

        with col3:
            max_dhw = current_data['dhw'].max()
            st.markdown(f'<div class="metric-card"><h3>{max_dhw:.1f}</h3><p>Max DHW</p></div>', unsafe_allow_html=True)

        with col4:
            avg_coral_cover = current_data['coral_cover'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_coral_cover:.1f}%</h3><p>Avg Coral Cover</p></div>', unsafe_allow_html=True)

        # Interactive map
        st.subheader("Interactive Risk Map")
        risk_map = create_risk_map(df, selected_date)
        st_folium(risk_map, width=700, height=500)

        # Risk summary table
        st.subheader("Current Risk Summary")
        risk_summary = current_data.groupby('risk_level').agg({
            'reef_name': 'count',
            'sst': 'mean',
            'dhw': 'mean',
            'bleaching_severity': 'mean'
        }).round(2)
        risk_summary.columns = ['Count', 'Avg SST (°C)', 'Avg DHW', 'Avg Bleaching (%)']
        st.dataframe(risk_summary, use_container_width=True)

    with tab2:
        st.header("Time Series Analysis")

        if selected_metrics:
            # Time series plot
            ts_fig = create_time_series_plot(df, selected_reef, selected_metrics)
            st.plotly_chart(ts_fig, use_container_width=True)

            # Recent trends
            st.subheader("Recent Trends (Last 30 Days)")
            recent_data = df[(df['reef_name'] == selected_reef) &
                           (df['date'] >= selected_date - timedelta(days=30))]

            col1, col2 = st.columns(2)

            with col1:
                # SST trend
                fig_sst = px.line(recent_data, x='date', y='sst',
                                 title='Sea Surface Temperature Trend')
                fig_sst.add_hline(y=recent_data['sst'].mean(), line_dash="dash",
                                 annotation_text="30-day Average")
                st.plotly_chart(fig_sst, use_container_width=True)

            with col2:
                # DHW trend
                fig_dhw = px.line(recent_data, x='date', y='dhw',
                                 title='Degree Heating Weeks Trend')
                fig_dhw.add_hline(y=4, line_dash="dash", line_color="orange")
                fig_dhw.add_hline(y=8, line_dash="dash", line_color="red")
                st.plotly_chart(fig_dhw, use_container_width=True)
        else:
            st.warning("Please select at least one metric from the sidebar.")

    with tab3:
        st.header("Predictive Analytics")

        with st.spinner('Training machine learning model...'):
            model, scaler, metrics, feature_importance = build_prediction_model(df)

        # Model performance
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Performance")
            st.metric("R² Score", f"{metrics['r2']:.3f}")
            st.metric("Mean Squared Error", f"{metrics['mse']:.3f}")

        with col2:
            st.subheader("Feature Importance")
            fig_importance = px.bar(feature_importance, x='importance', y='feature',
                                   orientation='h', title='Feature Importance in Bleaching Prediction')
            st.plotly_chart(fig_importance, use_container_width=True)

        # Prediction interface
        st.subheader("Bleaching Severity Prediction")
        st.write("Adjust the parameters below to predict bleaching severity:")

        col1, col2, col3 = st.columns(3)

        with col1:
            pred_sst = st.slider("Sea Surface Temperature (°C)", 20.0, 35.0, 28.0, 0.1)
            pred_anomaly = st.slider("SST Anomaly (°C)", -3.0, 6.0, 0.0, 0.1)

        with col2:
            pred_dhw = st.slider("Degree Heating Weeks", 0.0, 20.0, 2.0, 0.1)
            pred_turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 2.0, 0.1)

        with col3:
            pred_ph = st.slider("pH", 7.5, 8.5, 8.1, 0.01)
            pred_do = st.slider("Dissolved Oxygen (mg/L)", 5.0, 10.0, 7.5, 0.1)

        # Make prediction
        input_data = np.array([[pred_sst, pred_anomaly, pred_dhw, pred_turbidity, pred_ph, pred_do]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Display prediction
        if prediction > 70:
            alert_class = "alert-high"
            risk_text = "SEVERE BLEACHING RISK"
        elif prediction > 30:
            alert_class = "alert-medium"
            risk_text = "MODERATE BLEACHING RISK"
        else:
            alert_class = "alert-low"
            risk_text = "LOW BLEACHING RISK"

        st.markdown(f'<div class="{alert_class}"><h3>Predicted Bleaching Severity: {prediction:.1f}%</h3><p>{risk_text}</p></div>',
                   unsafe_allow_html=True)

    with tab4:
        st.header("Correlation Analysis")

        # Correlation matrix
        numeric_cols = ['sst', 'sst_anomaly', 'dhw', 'coral_cover', 'bleaching_severity', 'turbidity', 'ph', 'dissolved_oxygen']
        correlation_matrix = df[numeric_cols].corr()

        fig_corr = px.imshow(correlation_matrix,
                            title="Correlation Matrix of Environmental Parameters",
                            color_continuous_scale='RdBu_r',
                            aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter plots
        st.subheader("Relationship Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig_scatter1 = px.scatter(df, x='dhw', y='bleaching_severity',
                                     color='risk_level',
                                     title='DHW vs Bleaching Severity',
                                     trendline='ols')
            st.plotly_chart(fig_scatter1, use_container_width=True)

        with col2:
            fig_scatter2 = px.scatter(df, x='sst_anomaly', y='coral_cover',
                                     color='risk_level',
                                     title='SST Anomaly vs Coral Cover',
                                     trendline='ols')
            st.plotly_chart(fig_scatter2, use_container_width=True)

        # Regional analysis
        st.subheader("Regional Comparison")
        regional_stats = df.groupby('region').agg({
            'sst': 'mean',
            'dhw': 'mean',
            'bleaching_severity': 'mean',
            'coral_cover': 'mean'
        }).round(2)

        fig_regional = px.bar(regional_stats.reset_index(),
                             x='region', y=['sst', 'dhw', 'bleaching_severity'],
                             title='Regional Environmental Comparison',
                             barmode='group')
        st.plotly_chart(fig_regional, use_container_width=True)

    with tab5:
        st.header("Data Explorer")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            region_filter = st.multiselect("Filter by Region",
                                         df['region'].unique(),
                                         default=df['region'].unique())

        with col2:
            risk_filter = st.multiselect("Filter by Risk Level",
                                       ['Low', 'Medium', 'High'],
                                       default=['Low', 'Medium', 'High'])

        with col3:
            date_range = st.date_input("Date Range",
                                     value=[min_date, max_date],
                                     min_value=min_date,
                                     max_value=max_date)

        # Apply filters
        filtered_df = df[
            (df['region'].isin(region_filter)) &
            (df['risk_level'].isin(risk_filter)) &
            (df['date'] >= pd.to_datetime(date_range[0])) &
            (df['date'] <= pd.to_datetime(date_range[1]))
        ]

        # Display filtered data
        st.subheader(f"Filtered Dataset ({len(filtered_df):,} records)")
        st.dataframe(filtered_df, use_container_width=True)

        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="coral_reef_data.csv",
            mime="text/csv"
        )

        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(filtered_df.describe(), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🪸 CoralDash - Empowering coral reef conservation through data-driven insights</p>
        <p>Data sources: Simulated data based on Aqualink and NOAA Coral Reef Watch methodologies</p>
        <p>Built with Streamlit • Machine Learning • Interactive Visualizations</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
