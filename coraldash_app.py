import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Professional styling
st.set_page_config(
    page_title="CoralDash Pro",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
    }
    .alert-critical {
        background: #fee2e2;
        border: 1px solid #fca5a5;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .status-healthy { color: #10b981; font-weight: bold; }
    .status-warning { color: #f59e0b; font-weight: bold; }
    .status-critical { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🪸 CoralDash Pro</h1>
    <p>Advanced Coral Reef Health Monitoring & Predictive Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

# Generate realistic coral reef data
@st.cache_data
def load_coral_data():
    np.random.seed(42)
    reefs = [
        {"name": "Great Barrier Reef - North", "lat": -16.2839, "lon": 145.7781},
        {"name": "Great Barrier Reef - Central", "lat": -20.3444, "lon": 149.0969},
        {"name": "Coral Triangle - Palawan", "lat": 9.8349, "lon": 118.7384},
        {"name": "Red Sea - Eilat", "lat": 29.5581, "lon": 34.9482},
        {"name": "Caribbean - Bonaire", "lat": 12.1696, "lon": -68.2900},
        {"name": "Maldives - North Malé", "lat": 4.1755, "lon": 73.5093}
    ]

    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    data = []

    for reef in reefs:
        base_temp = np.random.uniform(26, 29)
        base_ph = np.random.uniform(7.9, 8.2)

        for i, date in enumerate(dates):
            # Seasonal patterns
            seasonal_temp = base_temp + 2 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 0.5)
            seasonal_ph = base_ph + 0.1 * np.cos(2 * np.pi * i / 365) + np.random.normal(0, 0.02)

            # Coral health based on temperature stress
            temp_stress = max(0, seasonal_temp - 29)
            coral_cover = max(20, 85 - temp_stress * 15 + np.random.normal(0, 5))

            # Other parameters
            turbidity = np.clip(np.random.lognormal(0, 0.5), 0.1, 10)
            salinity = np.random.normal(35, 0.5)
            dissolved_oxygen = np.random.normal(8, 0.5)

            # Bleaching events (temperature > 30°C)
            bleaching_severity = 0
            if seasonal_temp > 30:
                bleaching_severity = min(100, (seasonal_temp - 30) * 25)

            data.append({
                'reef_name': reef['name'],
                'latitude': reef['lat'],
                'longitude': reef['lon'],
                'date': date,
                'temperature': seasonal_temp,
                'ph': seasonal_ph,
                'coral_cover': coral_cover,
                'turbidity': turbidity,
                'salinity': salinity,
                'dissolved_oxygen': dissolved_oxygen,
                'bleaching_severity': bleaching_severity
            })

    return pd.DataFrame(data)

# Load data
df = load_coral_data()

# Sidebar controls
st.sidebar.header("🎛️ Control Panel")

# Reef selection
selected_reefs = st.sidebar.multiselect(
    "Select Reef Sites",
    options=df['reef_name'].unique(),
    default=df['reef_name'].unique()[:3]
)

# Date range
date_range = st.sidebar.date_input(
    "Date Range",
    value=[df['date'].min().date(), df['date'].max().date()],
    min_value=df['date'].min().date(),
    max_value=df['date'].max().date()
)

# Filter data
filtered_df = df[
    (df['reef_name'].isin(selected_reefs)) &
    (df['date'] >= pd.to_datetime(date_range[0])) &
    (df['date'] <= pd.to_datetime(date_range[1]))
]

# Health status function
def get_health_status(temp, ph, coral_cover):
    if temp > 30 or ph < 7.8 or coral_cover < 50:
        return "Critical", "🔴"
    elif temp > 29 or ph < 8.0 or coral_cover < 70:
        return "Warning", "🟡"
    else:
        return "Healthy", "🟢"

# Current status overview
st.header("📊 Real-Time Reef Health Dashboard")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

latest_data = filtered_df.groupby('reef_name').last().reset_index()
avg_temp = latest_data['temperature'].mean()
avg_ph = latest_data['ph'].mean()
avg_coral = latest_data['coral_cover'].mean()
critical_reefs = sum(1 for _, row in latest_data.iterrows()
                    if get_health_status(row['temperature'], row['ph'], row['coral_cover'])[0] == "Critical")

with col1:
    st.metric("Average Temperature", f"{avg_temp:.1f}°C",
             delta=f"{avg_temp - 28:.1f}°C from optimal")

with col2:
    st.metric("Average pH", f"{avg_ph:.2f}",
             delta=f"{avg_ph - 8.1:.2f} from optimal")

with col3:
    st.metric("Average Coral Cover", f"{avg_coral:.1f}%",
             delta=f"{avg_coral - 80:.1f}% from target")

with col4:
    st.metric("Reefs at Risk", f"{critical_reefs}",
             delta=f"{critical_reefs} critical")

# Alert system
alerts = []
for _, row in latest_data.iterrows():
    status, icon = get_health_status(row['temperature'], row['ph'], row['coral_cover'])
    if status in ["Critical", "Warning"]:
        alerts.append({
            'reef': row['reef_name'],
            'status': status,
            'icon': icon,
            'temp': row['temperature'],
            'ph': row['ph'],
            'coral': row['coral_cover']
        })

if alerts:
    st.subheader("⚠️ Active Alerts")
    for alert in alerts:
        if alert['status'] == "Critical":
            st.markdown(f"""
            <div class="alert-critical">
                {alert['icon']} <strong>CRITICAL:</strong> {alert['reef']}<br>
                Temperature: {alert['temp']:.1f}°C | pH: {alert['ph']:.2f} | Coral Cover: {alert['coral']:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-warning">
                {alert['icon']} <strong>WARNING:</strong> {alert['reef']}<br>
                Temperature: {alert['temp']:.1f}°C | pH: {alert['ph']:.2f} | Coral Cover: {alert['coral']:.1f}%
            </div>
            """, unsafe_allow_html=True)

# Main visualizations
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌡️ Temperature Trends & Bleaching Risk")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Sea Surface Temperature", "Coral Cover & Bleaching Events"),
        vertical_spacing=0.1
    )

    colors = px.colors.qualitative.Set3
    for i, reef in enumerate(selected_reefs):
        reef_data = filtered_df[filtered_df['reef_name'] == reef]

        # Temperature plot
        fig.add_trace(
            go.Scatter(
                x=reef_data['date'],
                y=reef_data['temperature'],
                name=reef.split(' - ')[0],
                line=dict(color=colors[i % len(colors)]),
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Temperature: %{y:.1f}°C<extra></extra>"
            ),
            row=1, col=1
        )

        # Coral cover plot
        fig.add_trace(
            go.Scatter(
                x=reef_data['date'],
                y=reef_data['coral_cover'],
                name=reef.split(' - ')[0],
                line=dict(color=colors[i % len(colors)]),
                showlegend=False,
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Coral Cover: %{y:.1f}%<extra></extra>"
            ),
            row=2, col=1
        )

    # Add bleaching threshold line
    fig.add_hline(y=29, line_dash="dash", line_color="orange",
                  annotation_text="Bleaching Threshold", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="red",
                  annotation_text="Critical Threshold", row=1, col=1)

    fig.update_layout(height=500, hovermode='x unified')
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Coral Cover (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Current Reef Status")

    # Status summary chart
    status_data = []
    for _, row in latest_data.iterrows():
        status, icon = get_health_status(row['temperature'], row['ph'], row['coral_cover'])
        status_data.append({
            'reef': row['reef_name'].split(' - ')[0],
            'status': status,
            'temperature': row['temperature'],
            'ph': row['ph'],
            'coral_cover': row['coral_cover']
        })

    status_df = pd.DataFrame(status_data)
    status_counts = status_df['status'].value_counts()

    colors_status = {'Healthy': '#10b981', 'Warning': '#f59e0b', 'Critical': '#ef4444'}

    fig_pie = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        color=status_counts.index,
        color_discrete_map=colors_status,
        title="Reef Health Distribution"
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed status table
    st.subheader("📋 Detailed Status")
    for _, row in status_df.iterrows():
        status_class = f"status-{row['status'].lower()}"
        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; background: #f8fafc;">
            <strong>{row['reef']}</strong><br>
            <span class="{status_class}">{row['status']}</span><br>
            🌡️ {row['temperature']:.1f}°C | 🧪 {row['ph']:.2f} | 🪸 {row['coral_cover']:.0f}%
        </div>
        """, unsafe_allow_html=True)

# Water quality analysis
st.header("🧪 Water Quality Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("pH Levels")
    fig_ph = px.box(
        filtered_df,
        x='reef_name',
        y='ph',
        color='reef_name',
        title="pH Distribution by Reef"
    )
    fig_ph.add_hline(y=8.1, line_dash="dash", line_color="green",
                     annotation_text="Optimal pH")
    fig_ph.add_hline(y=7.8, line_dash="dash", line_color="red",
                     annotation_text="Critical pH")
    fig_ph.update_xaxes(title="", tickangle=45)
    fig_ph.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_ph, use_container_width=True)

with col2:
    st.subheader("Turbidity Levels")
    fig_turb = px.violin(
        filtered_df,
        x='reef_name',
        y='turbidity',
        color='reef_name',
        title="Turbidity Distribution by Reef"
    )
    fig_turb.update_xaxes(title="", tickangle=45)
    fig_turb.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_turb, use_container_width=True)

# Predictive analytics
st.header("🔮 Predictive Analytics")

# Simple ML model for demonstration
@st.cache_resource
def train_model():
    # Prepare features
    model_df = df.copy()
    model_df['month'] = model_df['date'].dt.month
    model_df['day_of_year'] = model_df['date'].dt.dayofyear

    features = ['temperature', 'ph', 'turbidity', 'salinity', 'dissolved_oxygen', 'month', 'day_of_year']
    X = model_df[features]
    y = model_df['coral_cover']

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, features

model, scaler, features = train_model()

st.subheader("🎯 Coral Cover Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    pred_temp = st.slider("Temperature (°C)", 25.0, 35.0, 28.0, 0.1)
    pred_ph = st.slider("pH", 7.5, 8.5, 8.1, 0.01)

with col2:
    pred_turbidity = st.slider("Turbidity (NTU)", 0.1, 10.0, 1.0, 0.1)
    pred_salinity = st.slider("Salinity (ppt)", 30.0, 40.0, 35.0, 0.1)

with col3:
    pred_do = st.slider("Dissolved Oxygen (mg/L)", 5.0, 12.0, 8.0, 0.1)
    pred_month = st.selectbox("Month", range(1, 13), index=5)

# Make prediction
pred_data = pd.DataFrame({
    'temperature': [pred_temp],
    'ph': [pred_ph],
    'turbidity': [pred_turbidity],
    'salinity': [pred_salinity],
    'dissolved_oxygen': [pred_do],
    'month': [pred_month],
    'day_of_year': [pred_month * 30]  # Approximation
})

pred_scaled = scaler.transform(pred_data)
prediction = model.predict(pred_scaled)[0]

# Display prediction with color coding
if prediction >= 75:
    pred_color = "🟢"
    pred_status = "Excellent"
elif prediction >= 60:
    pred_color = "🟡"
    pred_status = "Moderate"
else:
    pred_color = "🔴"
    pred_status = "Poor"

st.markdown(f"""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 1rem 0;">
    <h2>Predicted Coral Cover</h2>
    <h1>{pred_color} {prediction:.1f}%</h1>
    <h3>Status: {pred_status}</h3>
</div>
""", unsafe_allow_html=True)

# Export functionality
st.header("📤 Data Export")

col1, col2 = st.columns(2)

with col1:
    # Summary statistics
    summary_stats = filtered_df.groupby('reef_name').agg({
        'temperature': ['mean', 'max', 'min'],
        'ph': ['mean', 'max', 'min'],
        'coral_cover': ['mean', 'max', 'min'],
        'bleaching_severity': 'max'
    }).round(2)

    csv_summary = summary_stats.to_csv()
    st.download_button(
        label="📊 Download Summary Statistics",
        data=csv_summary,
        file_name=f"coral_summary_{datetime.date.today()}.csv",
        mime="text/csv"
    )

with col2:
    # Full dataset
    csv_full = filtered_df.to_csv(index=False)
    st.download_button(
        label="📋 Download Full Dataset",
        data=csv_full,
        file_name=f"coral_data_{datetime.date.today()}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🪸 <strong>CoralDash Pro</strong> | Advanced Coral Reef Monitoring Platform</p>
    <p>Powered by AI & Machine Learning | Real-time Environmental Monitoring</p>
</div>
""", unsafe_allow_html=True)
