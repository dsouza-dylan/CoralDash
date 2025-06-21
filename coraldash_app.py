# CoralDash MegaApp - The Ultimate Coral Reef Monitoring & AI Platform

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import altair as alt
import shap
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pydeck as pdk

# ---- 1. Data Generation & API Simulation ----

@st.cache_data(ttl=300)
def generate_multimodal_data():
    reefs = ['Reef_X', 'Reef_Y', 'Reef_Z']
    latlon = [(21.3,-157.8), (20.9,-157.5), (21.5,-157.9)]
    dates = pd.date_range('2022-01-01', '2025-06-01', freq='D')
    records = []
    for i, reef in enumerate(reefs):
        lat, lon = latlon[i]
        for d in dates:
            day = d.timetuple().tm_yday
            temp = 25 + 6*np.sin(day/365*2*np.pi) + np.random.normal(0,0.3)
            ph = 8.1 - 0.02*np.cos(day/365*2*np.pi) + np.random.normal(0,0.015)
            turbidity = max(0.1, np.random.normal(1,0.4))
            satellite_ndvi = np.clip(0.5 + 0.3*np.sin(day/365*4*np.pi) + np.random.normal(0,0.05),0,1)
            bleaching_index = max(0, (temp - 28) * 15 + np.random.normal(0,4))
            coral_cover = np.clip(100 - bleaching_index*1.6 + np.random.normal(0,6),0,100)
            records.append([reef, lat, lon, d, temp, ph, turbidity, satellite_ndvi, bleaching_index, coral_cover])
    return pd.DataFrame(records, columns=[
        'Reef','Latitude','Longitude','Date','Sea_Temperature','pH','Turbidity_NTU',
        'Satellite_NDVI','Bleaching_Index','Coral_Cover'])

# Simulated live sensor API returning random noise with trending signal
async def async_sensor_api(reef_name):
    base_temp = {'Reef_X': 27, 'Reef_Y': 26, 'Reef_Z': 28}[reef_name]
    base_ph = 8.1
    while True:
        temp = base_temp + np.random.normal(0, 0.5)
        ph = base_ph - np.random.normal(0, 0.02)
        turbidity = np.clip(np.random.normal(1, 0.3), 0.1, 3)
        yield {'temp': temp, 'ph': ph, 'turbidity': turbidity, 'timestamp': datetime.datetime.now()}
        await asyncio.sleep(2)

# ---- 2. Train ML Model with SHAP ----

def train_ml_model(df):
    df_ml = df.copy()
    df_ml['DayOfYear'] = df_ml['Date'].dt.dayofyear
    features = ['Sea_Temperature', 'pH', 'Turbidity_NTU', 'Satellite_NDVI', 'DayOfYear']
    X = df_ml[features]
    y = df_ml['Coral_Cover']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=42)
    pipe = Pipeline([('scaler',StandardScaler()),('rf',RandomForestRegressor(n_estimators=300,random_state=42))])
    pipe.fit(X_train,y_train)
    score = pipe.score(X_test,y_test)
    return pipe, score, X_test

@st.cache_resource
def create_shap_explainer(model, X_sample):
    explainer = shap.TreeExplainer(model.named_steps['rf'])
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

# ---- 3. Deep Learning Coral Species Classifier (Mock) ----
# In reality, load a pre-trained model (e.g. EfficientNet, ResNet on coral images)
def mock_coral_classifier(image_bytes):
    # Random species & bleaching severity
    species = ['Acropora', 'Montipora', 'Porites', 'Fungia']
    severity = ['Healthy', 'Mild Bleaching', 'Severe Bleaching']
    np.random.seed(sum(image_bytes) % 1000)
    pred_species = np.random.choice(species)
    pred_severity = np.random.choice(severity, p=[0.6, 0.3, 0.1])
    confidence = np.random.uniform(0.75, 0.99)
    return pred_species, pred_severity, confidence

# ---- 4. Probabilistic Forecasting with Uncertainty (Mock Bayesian) ----
def coral_cover_forecast_with_uncertainty(temp, ph, turbidity, ndvi, day_of_year):
    base_pred = 90 - (temp - 27)*10 - (8.1 - ph)*15 + np.random.normal(0,3)
    uncertainty = np.clip(np.random.normal(5, 2), 2, 10)
    lower = max(0, base_pred - 1.96*uncertainty)
    upper = min(100, base_pred + 1.96*uncertainty)
    mean = np.clip(base_pred, 0, 100)
    return mean, lower, upper

# ---- 5. ENSO Phases & Anomaly Detection ----
def get_enso_phase(date):
    # Simplified repeating ENSO cycle mock (El Nino, Neutral, La Nina)
    year = date.year
    phases = ['El Niño', 'Neutral', 'La Niña']
    return phases[year % 3]

def detect_anomalies(df):
    anomalies = df[(df['pH'] < 7.85) | (df['Sea_Temperature'] > 30)]
    return anomalies[['Reef','Date','Sea_Temperature','pH']]

# ---- 6. App UI & Logic ----

st.set_page_config(page_title="CoralDash MegaApp", layout="wide", page_icon="🪸")
st.title("🪸 CoralDash MEGA - Ultimate Coral Reef Monitoring & AI Platform")

# Tabs for UI organization
tabs = st.tabs(["Dashboard", "AI Prediction & Explainability", "Coral Species Classifier", "3D Reef Visualization", "Anomaly Alerts & ENSO", "Collaboration & Export"])

df = generate_multimodal_data()
model, model_score, X_test = train_ml_model(df)
explainer, shap_values = create_shap_explainer(model, X_test)

with tabs[0]:
    st.header("🌊 Interactive Reef Dashboard")

    reefs = df['Reef'].unique()
    selected_reefs = st.multiselect("Select Reef(s)", reefs, default=list(reefs))
    date_min, date_max = df['Date'].min(), df['Date'].max()
    date_range = st.date_input("Select Date Range", [date_min, date_max], min_value=date_min, max_value=date_max)

    filtered_df = df[(df['Reef'].isin(selected_reefs)) & (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Sea Temp (°C)", f"{filtered_df['Sea_Temperature'].mean():.2f}")
    col2.metric("Avg pH", f"{filtered_df['pH'].mean():.2f}")
    col3.metric("Avg Turbidity (NTU)", f"{filtered_df['Turbidity_NTU'].mean():.2f}")
    col4.metric("Avg Coral Cover (%)", f"{filtered_df['Coral_Cover'].mean():.2f}")

    st.markdown("### Time Series Trends")
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, reef in enumerate(selected_reefs):
        df_r = filtered_df[filtered_df['Reef'] == reef]
        fig.add_trace(go.Scatter(x=df_r['Date'], y=df_r['Sea_Temperature'], mode='lines', name=f'{reef} Temp', line=dict(color=colors[i%len(colors)])))
        fig.add_trace(go.Scatter(x=df_r['Date'], y=df_r['Satellite_NDVI'], mode='lines', name=f'{reef} NDVI', line=dict(color=colors[i%len(colors)], dash='dot')))
        fig.add_trace(go.Scatter(x=df_r['Date'], y=df_r['Coral_Cover'], mode='lines', name=f'{reef} Coral Cover', line=dict(color=colors[i%len(colors)], dash='dash')))

    fig.update_layout(height=400, hovermode='x unified', legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Spatial Coral Reef Health Map")
    mean_bleaching = filtered_df.groupby(['Latitude','Longitude']).agg({'Bleaching_Index':'mean'}).reset_index()
    min_b, max_b = mean_bleaching['Bleaching_Index'].min(), mean_bleaching['Bleaching_Index'].max()
    import branca.colormap as cm
    colormap = cm.linear.YlOrRd_09.scale(min_b, max_b)
    colormap.caption = 'Avg Bleaching Index'

    m = folium.Map(location=[21.2, -157.7], zoom_start=10, tiles='CartoDB positron')
    for _, row in mean_bleaching.iterrows():
        color = colormap(row['Bleaching_Index'])
        folium.CircleMarker(location=[row['Latitude'],row['Longitude']],
                            radius=10,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            tooltip=f"Bleaching Index: {row['Bleaching_Index']:.2f}").add_to(m)
    colormap.add_to(m)
    st_folium(m, width=750, height=450)

with tabs[1]:
    st.header("🤖 AI Prediction & Explainability")

    st.markdown(f"**Model R² Score:** {model_score:.3f}")

    st.markdown("Input parameters to predict coral cover with uncertainty estimates:")

    col1, col2 = st.columns(2)
    with col1:
        temp_input = st.slider("Sea Temperature (°C)", 20.0, 35.0, 27.0)
        ph_input = st.slider("pH Level", 7.5, 8.5, 8.1, 0.01)
        turbidity_input = st.slider("Turbidity (NTU)", 0.1, 5.0, 1.0, 0.1)
    with col2:
        ndvi_input = st.slider("Satellite NDVI", 0.0, 1.0, 0.5, 0.01)
        date_input = st.date_input("Prediction Date", value=df['Date'].max())

    day_of_year = date_input.timetuple().tm_yday
    pred, lower, upper = coral_cover_forecast_with_uncertainty(temp_input, ph_input, turbidity_input, ndvi_input, day_of_year)

    st.metric("Predicted Coral Cover (%)", f"{pred:.2f}%", delta=f"± {(upper-lower)/2:.2f}%")
    st.markdown(f"Confidence Interval: [{lower:.2f}%, {upper:.2f}%]")

    # SHAP explainability - for demonstration, approximate by running model pipeline predict and explain
    feature_df = pd.DataFrame({'Sea_Temperature':[temp_input], 'pH':[ph_input], 'Turbidity_NTU':[turbidity_input], 'Satellite_NDVI':[ndvi_input], 'DayOfYear':[day_of_year]})
    shap_values = explainer.shap_values(feature_df)[0]
    st.markdown("### Model Feature Contributions (SHAP Values)")

    shap.initjs()
    shap_fig = shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values, feature_df.columns.tolist(), show=False)
    buf = io.BytesIO()
    shap_fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.image(buf)

with tabs[2]:
    st.header("🖼️ Coral Species & Bleaching Severity Classifier")

    uploaded_file = st.file_uploader("Upload Coral Image (JPEG/PNG)", type=['jpg','jpeg','png'])
    if uploaded_file:
        bytes_data = uploaded_file.read()
        species, severity, confidence = mock_coral_classifier(bytes_data)
        st.image(uploaded_file, caption="Uploaded Coral Image", use_column_width=True)
        st.success(f"Detected Species: **{species}**")
        st.info(f"Bleaching Severity: **{severity}**")
        st.write(f"Model Confidence: **{confidence*100:.1f}%**")
    else:
        st.info("Upload a coral photo to classify species and bleaching severity.")

with tabs[3]:
    st.header("🌐 3D Coral Reef Photogrammetry Explorer")

    # Dummy reef point cloud coords for demo
    np.random.seed(42)
    points = pd.DataFrame({
        'lat': np.random.uniform(21.0,21.6, 1000),
        'lon': np.random.uniform(-158.0,-157.4, 1000),
        'depth': np.random.uniform(-15,-1, 1000),
        'species': np.random.choice(['Acropora','Montipora','Porites'], 1000)
    })
    species_color = {'Acropora':'#1f77b4','Montipora':'#ff7f0e','Porites':'#2ca02c'}
    points['color'] = points['species'].map(species_color).fillna('#d62728')

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=points,
        get_position='[lon, lat, depth]',
        get_color='color',
        get_radius=20,
        pickable=True,
        auto_highlight=True,
        radius_scale=20,
        radius_min_pixels=2,
        radius_max_pixels=15,
    )
    view_state = pdk.ViewState(latitude=21.3, longitude=-157.7, zoom=12, pitch=45)

    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{species}"})
    st.pydeck_chart(r)

with tabs[4]:
    st.header("⚠️ Anomaly Detection & ENSO Phases")

    anomalies = detect_anomalies(df)
    st.markdown("### Detected Water Chemistry Anomalies")
    st.dataframe(anomalies)

    st.markdown("### ENSO Phases Over Time")
    df['ENSO_Phase'] = df['Date'].apply(get_enso_phase)
    enso_counts = df.groupby(['Date','ENSO_Phase']).size().reset_index(name='Count')
    fig_enso = px.area(enso_counts, x='Date', y='Count', color='ENSO_Phase', title="ENSO Phase Temporal Distribution")
    st.plotly_chart(fig_enso, use_container_width=True)

with tabs[5]:
    st.header("🤝 Collaborative Annotation & Data Export")

    if 'annotations' not in st.session_state:
        st.session_state['annotations'] = []

    st.markdown("### Annotate Reef Health Notes")

    reef_for_note = st.selectbox("Select Reef to Annotate", reefs)
    date_for_note = st.date_input("Select Date for Note", value=datetime.date.today())
    note_text = st.text_area("Enter Annotation Text")

    if st.button("Add Annotation"):
        st.session_state.annotations.append({
            'reef': reef_for_note,
            'date': date_for_note,
            'note': note_text
        })
        st.success("Annotation added!")

    if st.session_state.annotations:
        st.markdown("### Current Annotations")
        st.dataframe(pd.DataFrame(st.session_state.annotations))

    # Export filtered data + annotations
    combined_export = filtered_df.copy()
    if st.session_state.annotations:
        ann_df = pd.DataFrame(st.session_state.annotations)
        ann_df['Date'] = pd.to_datetime(ann_df['date'])
        combined_export = combined_export.merge(ann_df, left_on=['Reef','Date'], right_on=['reef','Date'], how='left')

    csv_export = combined_export.to_csv(index=False).encode('utf-8')
    st.download_button("Download Reef Data + Annotations CSV", csv_export, "coraldash_full_export.csv", "text/csv")

# ---- End of MegaApp ----
