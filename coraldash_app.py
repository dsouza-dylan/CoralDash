# main.py

import streamlit as st
import pandas as pd

from data_generator import CoralReefDataGenerator
from analytics import CoralAnalyticsEngine
from visualization import CoralVisualizationEngine

st.set_page_config(page_title="CoralDash™", layout="wide")

# Sidebar
st.sidebar.title("🪸 CoralDash Controls")

@st.cache_data
def load_data():
    generator = CoralReefDataGenerator()
    return generator.generate_dataset()

data = load_data()
reefs = data['reef_name'].unique()
selected_reefs = st.sidebar.multiselect("Select Reefs", options=reefs, default=list(reefs))

# Filtered Data
filtered_data = data[data['reef_name'].isin(selected_reefs)]

# Engines
analytics = CoralAnalyticsEngine(filtered_data)
viz = CoralVisualizationEngine(filtered_data)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🌡 Visualizations"])

with tab1:
    st.header("Coral Cover Overview")
    avg_cover = filtered_data['coral_cover'].mean()
    avg_bleaching = filtered_data['bleaching_severity'].mean()
    max_dhw = filtered_data['dhw'].max()
    st.metric("Average Coral Cover", f"{avg_cover:.2f}%")
    st.metric("Average Bleaching Severity", f"{avg_bleaching:.1f}%")
    st.metric("Max DHW", f"{max_dhw:.2f} °C-weeks")

    st.subheader("Coral Cover Over Time")
    reef = st.selectbox("Pick a reef to view trend", selected_reefs)
    st.plotly_chart(viz.coral_cover_trend(reef), use_container_width=True)

with tab2:
    st.header("Trend Analysis")
    trends = analytics.trend_summary()
    for key, result in trends.items():
        st.subheader(f"{key.upper()} Trend")
        st.write(f"Annual Change: {result['annual_change']:.4f}")
        st.write(f"R²: {result['r_squared']:.3f} — {'Significant' if result['significant'] else 'Not Significant'}")
        st.markdown("---")

    st.subheader("Bleaching Event Detection")
    events = analytics.detect_bleaching_events()
    st.dataframe(events)

with tab3:
    st.header("Temperature & Bleaching Visualizations")
    st.plotly_chart(viz.temperature_timeline(selected_reefs), use_container_width=True)
    st.plotly_chart(viz.coral_bleaching_scatter(), use_container_width=True)
