import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Coral Reef Health Monitor",
    page_icon="🐠",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        color: #ffffff;
    }
    
    .reef-card {
        background: rgba(30, 41, 59, 0.8);
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        color: #ffffff;
    }
    .healthy { 
        border-color: #22c55e; 
        background: rgba(21, 128, 61, 0.2);
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
    }
    .warning { 
        border-color: #f59e0b; 
        background: rgba(180, 83, 9, 0.2);
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
    }
    .critical { 
        border-color: #ef4444; 
        background: rgba(153, 27, 27, 0.2);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
        color: #ffffff;
    }
    .status-text {
        font-size: 1.1rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #475569;
        color: #ffffff;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e293b, #334155);
        border: 1px solid #475569;
        padding: 1rem;
        border-radius: 10px;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_real_data():

    site_files = {
        'Raja Ampat': {
            'file': 'aqualink_data/data_site_1164_2021_06_07_2025_07_17.csv',
            'lat': -0.5, 'lon': 130.5
        },
        'Bonaire': {
            'file': 'aqualink_data/data_site_1097_2021_06_07_2025_07_17.csv',
            'lat': 12.2, 'lon': -68.3
        },
        'Aldabra': {
            'file': 'aqualink_data/data_site_3006_2021_02_19_2025_07_17.csv',
            'lat': -9.4, 'lon': 46.4
        },
        'Palmyra': {
            'file': 'aqualink_data/data_site_3213_2021_07_14_2025_07_17.csv',
            'lat': 5.9, 'lon': -162.1
        },
        'Great Barrier Reef': {
            'file': 'aqualink_data/data_site_3501_2024_01_31_2025_07_17.csv',
            'lat': -16.3, 'lon': 145.8
        },
        'Ras Mohamed': {
            'file': 'aqualink_data/data_site_4402_2024_11_20_2025_07_17.csv',
            'lat': 27.7, 'lon': 34.2
        },
        'Darwin': {
            'file': 'aqualink_data/data_site_944_2021_06_07_2025_07_17.csv',
            'lat': -12.5, 'lon': 130.8
        },
        'Moorea': {
            'file': 'aqualink_data/data_site_959_2021_06_07_2025_07_17.csv',
            'lat': -17.5, 'lon': -149.8
        }
    }

    all_data = []

    for reef_name, info in site_files.items():
        try:
            df = pd.read_csv(info['file'])
            df['reef'] = reef_name
            df['lat'] = info['lat']
            df['lon'] = info['lon']
            all_data.append(df)
        except FileNotFoundError:
            st.error(f"Could not find data file: {info['file']}")
            continue

    if not all_data:
        st.error("No data files could be loaded. Please check file paths.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    if combined_df['timestamp'].dt.tz is not None:
        combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize(None)

    return combined_df

def get_reef_status(dhw, temp_alert):
    if pd.isna(dhw) or pd.isna(temp_alert):
        return "❓ Unknown", "warning", 2

    if dhw >= 8 or temp_alert >= 2:
        return "🚨 Critical", "critical", 3
    elif dhw >= 4 or temp_alert >= 1:
        return "⚠️ Watch", "warning", 2
    else:
        return "✅ Healthy", "healthy", 1

def create_world_map(df, selected_year, selected_month):

    month_data = df[
        (df['timestamp'].dt.year == selected_year) &
        (df['timestamp'].dt.month == selected_month)
    ]

    if month_data.empty:
        return None

    selected_data = []
    for reef in df['reef'].unique():
        reef_month_data = month_data[month_data['reef'] == reef]
        if not reef_month_data.empty:
            reef_summary = {
                'reef': reef,
                'lat': reef_month_data['lat'].iloc[0],
                'lon': reef_month_data['lon'].iloc[0],
                'satellite_temperature_noaa': reef_month_data['satellite_temperature_noaa'].mean(),
                'dhw_noaa': reef_month_data['dhw_noaa'].max(),  # Peak stress
                'temp_alert_noaa': reef_month_data['temp_alert_noaa'].max(),  # Worst alert
                'days_in_month': len(reef_month_data)
            }
            selected_data.append(reef_summary)

    if not selected_data:
        return None

    map_df = pd.DataFrame(selected_data)

    map_df['status_text'], map_df['status_class'], map_df['status_num'] = zip(*[
        get_reef_status(row['dhw_noaa'], row['temp_alert_noaa'])
        for _, row in map_df.iterrows()
    ])

    color_map = {1: '#22c55e', 2: '#f59e0b', 3: '#ef4444'}
    map_df['color'] = map_df['status_num'].map(color_map)

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=map_df['lat'],
        lon=map_df['lon'],
        mode='markers',
        marker=dict(
            size=15,
            color=map_df['color'],
            sizemode='diameter'
        ),
        text=map_df['reef'],
        customdata=np.column_stack((
            map_df['satellite_temperature_noaa'],
            map_df['dhw_noaa'],
            map_df['status_text'],
            map_df['days_in_month']
        )),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Avg Temperature: %{customdata[0]:.1f}°C<br>'
            'Peak Heat Stress: %{customdata[1]:.1f} DHW<br>'
            'Status: %{customdata[2]}<br>'
            'Days of data: %{customdata[3]}<br>'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=1,
            center=dict(lat=0, lon=0)
        ),
        height=400,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_smart_plot(data, metric, reef_name):

    fig = go.Figure()

    if metric == 'temp_alert_noaa':
        alert_counts = data[metric].value_counts().sort_index()

        colors = ['#22c55e', '#fbbf24', '#f97316', '#ef4444', '#dc2626']

        fig.add_trace(go.Bar(
            x=alert_counts.index,
            y=alert_counts.values,
            marker_color=[colors[int(i)] if i < len(colors) else colors[-1] for i in alert_counts.index],
            hovertemplate='Alert Level %{x}<br>Days: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title=f"Alert Level Distribution - {reef_name}",
            xaxis_title="Alert Level",
            yaxis_title="Number of Days",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )

    else:
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data[metric],
            mode='lines',
            name=metric.replace('_', ' ').title(),
            line=dict(color='#0ea5e9', width=2),
            hovertemplate=f'<b>%{{x}}</b><br>Value: %{{y:.2f}}<extra></extra>'
        ))

        if metric == 'dhw_noaa':
            fig.add_hline(y=4, line_dash="dash", line_color="orange",
                         annotation_text="Bleaching Possible")
            fig.add_hline(y=8, line_dash="dash", line_color="red",
                         annotation_text="Severe Bleaching Likely")

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=metric.replace('_', ' ').title(),
            hovermode='x unified'
        )

    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )

    return fig

def main():
    st.title("🐠 Coral Reef Health Monitor")
    st.markdown("*Historical monitoring of coral reef stress and bleaching risk*")

    df = load_real_data()

    if df.empty:
        st.stop()

    st.markdown("---")
    st.header("📅 Select Month to Explore")

    col1, col2 = st.columns(2)

    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    with col1:
        selected_year = st.selectbox(
            "Year:",
            range(min_date.year, max_date.year + 1),
            index=len(range(min_date.year, max_date.year + 1)) - 1
        )

    with col2:
        selected_month = st.selectbox(
            "Month:",
            range(1, 13),
            index=datetime.now().month - 1,
            format_func=lambda x: datetime(2021, x, 1).strftime('%B')
        )

    selected_period = pd.Period(year=selected_year, month=selected_month, freq='M')

    st.markdown("---")

    st.header("🗺️ Global Reef Status Map")

    try:
        world_map = create_world_map(df, selected_year, selected_month)
        if world_map:
            st.plotly_chart(world_map, use_container_width=True)
        else:
            st.warning(f"No data available for {datetime(selected_year, selected_month, 1).strftime('%B %Y')}")
    except Exception as e:
        st.error(f"Could not create map: {str(e)}")

    month_data = df[
        (df['timestamp'].dt.year == selected_year) &
        (df['timestamp'].dt.month == selected_month)
    ]

    monthly_summaries = []
    if not month_data.empty:
        for reef in df['reef'].unique():
            reef_month_data = month_data[month_data['reef'] == reef]
            if not reef_month_data.empty:
                reef_summary = {
                    'reef': reef,
                    'satellite_temperature_noaa': reef_month_data['satellite_temperature_noaa'].mean(),
                    'dhw_noaa': reef_month_data['dhw_noaa'].max(),  # Peak stress for the month
                    'temp_alert_noaa': reef_month_data['temp_alert_noaa'].max(),  # Worst alert
                    'days_data': len(reef_month_data)
                }
                monthly_summaries.append(reef_summary)

        latest_data = pd.DataFrame(monthly_summaries)

        st.markdown("---")

        st.header(f"📊 Reef Status Summary for {datetime(selected_year, selected_month, 1).strftime('%B %Y')}")

        num_reefs = len(latest_data)
        cols = st.columns(min(num_reefs, 4))

        for i, (_, reef_data) in enumerate(latest_data.iterrows()):
            col_idx = i % 4

            status_text, status_class, _ = get_reef_status(
                reef_data.get('dhw_noaa', np.nan),
                reef_data.get('temp_alert_noaa', np.nan)
            )

            with cols[col_idx]:
                temp = reef_data.get('satellite_temperature_noaa', 'N/A')
                dhw = reef_data.get('dhw_noaa', 0)
                days = reef_data.get('days_data', 0)

                st.markdown(f"""
                <div class="reef-card {status_class}">
                    <h4>{reef_data['reef']}</h4>
                    <p class="big-number">{temp:.1f}°C</p>
                    <p class="status-text">{status_text}</p>
                    <small>Peak DHW: {dhw:.1f} | {days} days</small>
                </div>
                """, unsafe_allow_html=True)

            # Start new row after every 4 items
            if (i + 1) % 4 == 0 and i + 1 < num_reefs:
                cols = st.columns(min(num_reefs - i - 1, 4))

    st.markdown("---")

    st.header("🔍 Detailed Time Series Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")

        selected_reef = st.selectbox(
            "Choose a reef to analyze:",
            df['reef'].unique(),
            index=0
        )

        reef_data = df[df['reef'] == selected_reef].copy()
        reef_data = reef_data.sort_values('timestamp')

        if len(reef_data) > 0:
            max_days = len(reef_data)
            time_options = ["Full period"]

            if max_days >= 365:
                time_options.insert(0, "Last year")
            if max_days >= 180:
                time_options.insert(0, "Last 6 months")
            if max_days >= 90:
                time_options.insert(0, "Last 90 days")
            if max_days >= 30:
                time_options.insert(0, "Last 30 days")

            time_period = st.radio("Time period:", time_options)

        available_metrics = [col for col in reef_data.columns if 'noaa' in col.lower()]
        metric_options = {
            'satellite_temperature_noaa': 'Sea Surface Temperature',
            'dhw_noaa': 'Heat Stress (DHW)',
            'sst_anomaly_noaa': 'Temperature Anomaly',
            'temp_alert_noaa': 'Alert Level Distribution'
        }

        available_display = [metric_options.get(col, col) for col in available_metrics if col in metric_options]
        selected_display = st.selectbox("Show:", available_display)
        selected_metric = [k for k, v in metric_options.items() if v == selected_display][0]

        if selected_metric != 'temp_alert_noaa':
            show_alerts = st.checkbox("Highlight critical events", value=True)
        else:
            show_alerts = False

    with col2:
        st.subheader(f"{selected_display} - {selected_reef}")

        if time_period == "Last 30 days":
            reef_data = reef_data.tail(30)
        elif time_period == "Last 90 days":
            reef_data = reef_data.tail(90)
        elif time_period == "Last 6 months":
            reef_data = reef_data.tail(180)
        elif time_period == "Last year":
            reef_data = reef_data.tail(365)

        if len(reef_data) == 0:
            st.warning("No data available for selected reef.")
        else:
            fig = create_smart_plot(reef_data, selected_metric, selected_reef)

            if show_alerts and selected_metric != 'temp_alert_noaa' and 'dhw_noaa' in reef_data.columns:
                critical_data = reef_data[reef_data['dhw_noaa'] >= 8]
                if not critical_data.empty:
                    fig.add_trace(go.Scatter(
                        x=critical_data['timestamp'],
                        y=critical_data[selected_metric],
                        mode='markers',
                        name='Critical Events',
                        marker=dict(color='red', size=8, symbol='circle'),
                        hovertemplate='<b>Critical Event</b><br>%{x}<br>Value: %{y:.2f}<extra></extra>'
                    ))

            st.plotly_chart(fig, use_container_width=True)

    if len(reef_data) > 0:
        st.subheader(f"📈 Summary for {selected_reef} ({time_period})")

        col1, col2, col3, col4 = st.columns(4)

        current_temp = reef_data['satellite_temperature_noaa'].iloc[-1]
        current_dhw = reef_data.get('dhw_noaa', pd.Series([0])).iloc[-1]
        max_dhw = reef_data.get('dhw_noaa', pd.Series([0])).max()
        critical_days = len(reef_data[reef_data.get('dhw_noaa', pd.Series()) >= 8])

        col1.metric("Latest Temperature", f"{current_temp:.1f}°C")
        col2.metric("Latest Heat Stress", f"{current_dhw:.1f} DHW")
        col3.metric("Peak Heat Stress", f"{max_dhw:.1f} DHW")
        col4.metric("Critical Days", f"{critical_days}")

    st.markdown("---")

    with st.expander("📚 Understanding the Data"):
        st.markdown("""
        This dashboard uses historical NOAA coral reef monitoring data.
        
        **Sea Surface Temperature**: Satellite measurements in Celsius (°C)
        
        **Heat Stress (DHW)**: Degree Heating Weeks - accumulated heat stress over 12 weeks:
        - 0-4 DHW: Normal conditions 🟢
        - 4-8 DHW: Bleaching possible 🟡  
        - 8+ DHW: Severe bleaching likely 🔴
        
        **Temperature Anomaly**: How much current temperature differs from long-term average
        
        **Alert Level**: NOAA's coral bleaching alert system:
        - 0: No stress
        - 1: Bleaching watch
        - 2: Bleaching warning  
        - 3: Bleaching alert level 1
        - 4: Bleaching alert level 2
        
        **Health Status Legend**:
        - ✅ **Healthy**: DHW < 4, no significant alerts
        - ⚠️ **Watch**: DHW 4-8 or temperature alerts present
        - 🚨 **Critical**: DHW ≥ 8 or severe alerts active
        """)

    st.markdown("---")
    st.caption("🔬 Data source: NOAA Coral Reef Watch Program | 🌊 Historical satellite monitoring data")

if __name__ == "__main__":
    main()

