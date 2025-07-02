import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import io
import base64
from typing import Dict, List, Tuple, Optional
import json

# Configuration
st.set_page_config(
    page_title="CoralYO - AI Coral Research Platform",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .coral-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        background-size: 200% 200%;
        animation: gradient 5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4ECDC4;
    }
    .species-tag {
        background: #E3F2FD;
        color: #1976D2;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class CoralClassifier:
    """Mock AI classifier for coral species identification and health assessment"""
    
    def __init__(self):
        self.species_database = {
            'Acropora cervicornis': {
                'common_name': 'Staghorn Coral',
                'family': 'Acroporidae',
                'growth_form': 'Branching',
                'threat_level': 'Critically Endangered',
                'description': 'Fast-growing branching coral, critical for reef structure'
            },
            'Acropora palmata': {
                'common_name': 'Elkhorn Coral',
                'family': 'Acroporidae', 
                'growth_form': 'Plate-like',
                'threat_level': 'Critically Endangered',
                'description': 'Large plate-like coral providing habitat for many species'
            },
            'Montastraea cavernosa': {
                'common_name': 'Great Star Coral',
                'family': 'Montastraeidae',
                'growth_form': 'Massive',
                'threat_level': 'Near Threatened',
                'description': 'Large boulder coral with distinctive star-shaped polyps'
            },
            'Diploria strigosa': {
                'common_name': 'Symmetrical Brain Coral',
                'family': 'Mussidae',
                'growth_form': 'Massive',
                'threat_level': 'Near Threatened',
                'description': 'Brain-like coral with symmetrical ridge patterns'
            },
            'Porites astreoides': {
                'common_name': 'Mustard Hill Coral',
                'family': 'Poritidae',
                'growth_form': 'Massive',
                'threat_level': 'Least Concern',
                'description': 'Hardy coral with small polyps, often yellow-green'
            }
        }
        
        self.health_indicators = {
            'healthy': {'color': '#00C851', 'description': 'Normal coloration and polyp extension'},
            'stressed': {'color': '#FF8800', 'description': 'Pale coloration, possible environmental stress'},
            'bleached': {'color': '#FF4444', 'description': 'White/pale, symbiotic algae loss'},
            'diseased': {'color': '#AA00FF', 'description': 'Tissue necrosis or disease signs'},
            'dead': {'color': '#666666', 'description': 'No living tissue, algae-covered skeleton'}
        }
    
    def identify_species(self, image: np.ndarray) -> Dict:
        """Mock species identification - in reality would use trained CNN model"""
        # Simulate AI processing
        species_list = list(self.species_database.keys())
        # Mock confidence scores
        confidences = np.random.dirichlet(np.ones(len(species_list)) * 0.5)
        predicted_species = species_list[np.argmax(confidences)]
        
        return {
            'predicted_species': predicted_species,
            'confidence': float(np.max(confidences)),
            'all_predictions': {species: float(conf) for species, conf in zip(species_list, confidences)},
            'species_info': self.species_database[predicted_species]
        }
    
    def assess_health(self, image: np.ndarray) -> Dict:
        """Mock health assessment - would use trained model for health indicators"""
        # Simulate health analysis
        health_states = list(self.health_indicators.keys())
        health_probs = np.random.dirichlet(np.ones(len(health_states)) * 0.3)
        predicted_health = health_states[np.argmax(health_probs)]
        
        # Mock additional metrics
        coverage_percent = np.random.uniform(20, 95)
        polyp_density = np.random.uniform(50, 200)
        tissue_thickness = np.random.uniform(0.5, 3.0)
        
        return {
            'health_status': predicted_health,
            'confidence': float(np.max(health_probs)),
            'health_probabilities': {state: float(prob) for state, prob in zip(health_states, health_probs)},
            'coverage_percent': coverage_percent,
            'polyp_density_per_cm2': polyp_density,
            'tissue_thickness_mm': tissue_thickness,
            'color_info': self.health_indicators[predicted_health]
        }
    
    def get_research_insights(self, species: str, health_status: str) -> List[str]:
        """Generate research insights based on identification and health assessment"""
        insights = []
        
        if species in self.species_database:
            species_info = self.species_database[species]
            
            if species_info['threat_level'] == 'Critically Endangered':
                insights.append(f"⚠️ {species} is critically endangered - document location for conservation efforts")
            
            if health_status == 'bleached' and species_info['growth_form'] == 'Branching':
                insights.append("🌡️ Branching corals are highly susceptible to thermal stress - check water temperature")
            
            if health_status == 'healthy' and species_info['threat_level'] != 'Least Concern':
                insights.append("✅ Healthy specimen of threatened species - ideal for genetic sampling")
        
        return insights

def create_sample_data():
    """Create sample historical data for dashboard"""
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
    
    # Mock data for different monitoring sites
    sites = ['Reef Site A', 'Reef Site B', 'Reef Site C', 'Reef Site D']
    data = []
    
    for site in sites:
        for date in dates:
            # Simulate seasonal patterns and trends
            base_health = 0.7 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
            health_score = base_health + np.random.normal(0, 0.1)
            health_score = np.clip(health_score, 0, 1)
            
            data.append({
                'date': date,
                'site': site,
                'health_score': health_score,
                'coral_coverage': np.random.uniform(30, 80),
                'species_count': np.random.randint(8, 15),
                'temperature_c': 25 + 3 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 1)
            })
    
    return pd.DataFrame(data)

def main():
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = CoralClassifier()
    
    # Header
    st.markdown('<h1 class="coral-header">🪸 CoralYO</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Coral Ecology Research Platform</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-style: italic;">Jennifer Smith Lab - UC San Diego</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔬 Research Tools")
    selected_tool = st.sidebar.selectbox(
        "Select Research Module:",
        ["Coral Identification", "Health Assessment", "Monitoring Dashboard", "Research Database"]
    )
    
    if selected_tool == "Coral Identification":
        coral_identification_module()
    elif selected_tool == "Health Assessment":
        health_assessment_module()
    elif selected_tool == "Monitoring Dashboard":
        monitoring_dashboard()
    elif selected_tool == "Research Database":
        research_database()

def coral_identification_module():
    st.header("🔍 Coral Species Identification")
    st.markdown("Upload coral images for AI-powered species identification")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a coral image...",
        type=['jpg', 'jpeg', 'png', 'tiff'],
        help="Upload underwater coral images for species identification"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Coral Image", use_column_width=True)
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Process with AI
        with st.spinner("🤖 Analyzing coral species..."):
            results = st.session_state.classifier.identify_species(img_array)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏷️ Identification Results")
            
            # Main prediction
            predicted_species = results['predicted_species']
            confidence = results['confidence']
            species_info = results['species_info']
            
            st.success(f"**Predicted Species:** {predicted_species}")
            st.info(f"**Common Name:** {species_info['common_name']}")
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Species information
            st.markdown("### Species Information")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Family:** {species_info['family']}")
                st.write(f"**Growth Form:** {species_info['growth_form']}")
            with col_b:
                st.write(f"**Threat Level:** {species_info['threat_level']}")
                threat_color = "red" if "Critically" in species_info['threat_level'] else "orange" if "Near" in species_info['threat_level'] else "green"
                st.markdown(f"<span style='color: {threat_color}'>●</span> Conservation Status", unsafe_allow_html=True)
            
            st.write(f"**Description:** {species_info['description']}")
            
            # All predictions
            st.markdown("### All Predictions")
            pred_df = pd.DataFrame(
                [(species, conf) for species, conf in results['all_predictions'].items()],
                columns=['Species', 'Confidence']
            ).sort_values('Confidence', ascending=False)
            
            fig = px.bar(pred_df, x='Confidence', y='Species', orientation='h',
                        title="Species Prediction Confidence",
                        color='Confidence', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Quick Stats")
            st.metric("Image Resolution", f"{img_array.shape[1]}x{img_array.shape[0]}")
            st.metric("Color Channels", img_array.shape[2] if len(img_array.shape) > 2 else 1)
            
            # Export results
            if st.button("📥 Export Results"):
                results_json = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name=f"coral_id_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def health_assessment_module():
    st.header("🏥 Coral Health Assessment")
    st.markdown("Analyze coral health indicators and stress markers")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a coral image for health assessment...",
        type=['jpg', 'jpeg', 'png', 'tiff'],
        key="health_uploader"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Coral Health Assessment", use_column_width=True)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Process with AI
        with st.spinner("🔬 Analyzing coral health..."):
            health_results = st.session_state.classifier.assess_health(img_array)
            species_results = st.session_state.classifier.identify_species(img_array)
        
        # Display health results
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("🏥 Health Assessment")
            
            health_status = health_results['health_status']
            confidence = health_results['confidence']
            color_info = health_results['color_info']
            
            # Health status with color coding
            st.markdown(f"**Health Status:** <span style='color: {color_info['color']}; font-weight: bold;'>{health_status.title()}</span>", unsafe_allow_html=True)
            st.write(f"**Description:** {color_info['description']}")
            st.metric("Assessment Confidence", f"{confidence:.1%}")
            
            # Health probabilities
            health_probs = health_results['health_probabilities']
            prob_df = pd.DataFrame(
                [(status, prob) for status, prob in health_probs.items()],
                columns=['Health Status', 'Probability']
            ).sort_values('Probability', ascending=False)
            
            fig = px.pie(prob_df, values='Probability', names='Health Status',
                        title="Health Status Probabilities",
                        color_discrete_map={
                            'healthy': '#00C851',
                            'stressed': '#FF8800', 
                            'bleached': '#FF4444',
                            'diseased': '#AA00FF',
                            'dead': '#666666'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Health Metrics")
            st.metric("Coral Coverage", f"{health_results['coverage_percent']:.1f}%")
            st.metric("Polyp Density", f"{health_results['polyp_density_per_cm2']:.0f}/cm²")
            st.metric("Tissue Thickness", f"{health_results['tissue_thickness_mm']:.1f}mm")
        
        with col3:
            st.subheader("🔬 Research Insights")
            insights = st.session_state.classifier.get_research_insights(
                species_results['predicted_species'], health_status
            )
            
            for insight in insights:
                st.info(insight)
            
            if not insights:
                st.info("No specific research insights for this combination")
        
        # Detailed analysis
        st.subheader("📊 Detailed Analysis")
        
        # Create metrics visualization
        metrics_data = {
            'Metric': ['Coverage %', 'Polyp Density', 'Tissue Thickness'],
            'Value': [
                health_results['coverage_percent'],
                health_results['polyp_density_per_cm2'],
                health_results['tissue_thickness_mm']
            ],
            'Unit': ['%', 'per cm²', 'mm']
        }
        
        # Normalize values for radar chart
        normalized_values = [
            health_results['coverage_percent'] / 100,
            health_results['polyp_density_per_cm2'] / 200,
            health_results['tissue_thickness_mm'] / 5
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=metrics_data['Metric'],
            fill='toself',
            name='Health Metrics',
            line_color='#4ECDC4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Normalized Health Metrics"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def monitoring_dashboard():
    st.header("📊 Coral Reef Monitoring Dashboard")
    st.markdown("Long-term monitoring and trend analysis")
    
    # Generate sample data
    df = create_sample_data()
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['date'].min())
    with col2:
        end_date = st.date_input("End Date", df['date'].max())
    
    # Filter data
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df[mask]
    
    # Summary metrics
    st.subheader("📈 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_health = filtered_df['health_score'].mean()
        st.metric("Average Health Score", f"{avg_health:.2f}", f"{avg_health-0.7:.2f}")
    
    with col2:
        avg_coverage = filtered_df['coral_coverage'].mean()
        st.metric("Average Coverage", f"{avg_coverage:.1f}%", f"{avg_coverage-60:.1f}%")
    
    with col3:
        avg_species = filtered_df['species_count'].mean()
        st.metric("Average Species Count", f"{avg_species:.0f}", f"{avg_species-12:.0f}")
    
    with col4:
        avg_temp = filtered_df['temperature_c'].mean()
        st.metric("Average Temperature", f"{avg_temp:.1f}°C", f"{avg_temp-26:.1f}°C")
    
    # Time series plots
    st.subheader("📊 Temporal Trends")
    
    # Health score over time
    fig = px.line(filtered_df, x='date', y='health_score', color='site',
                  title="Coral Health Score Over Time",
                  labels={'health_score': 'Health Score', 'date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-metric dashboard
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Coral Coverage', 'Species Count', 'Temperature', 'Health vs Coverage'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Add traces for each site
    for site in filtered_df['site'].unique():
        site_data = filtered_df[filtered_df['site'] == site]
        
        fig.add_trace(
            go.Scatter(x=site_data['date'], y=site_data['coral_coverage'], name=f"{site} Coverage", mode='lines'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=site_data['date'], y=site_data['species_count'], name=f"{site} Species", mode='lines'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=site_data['date'], y=site_data['temperature_c'], name=f"{site} Temp", mode='lines'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=site_data['coral_coverage'], y=site_data['health_score'], 
                      name=f"{site} Health vs Coverage", mode='markers'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Multi-Metric Monitoring Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Site comparison
    st.subheader("🏝️ Site Comparison")
    
    # Box plots for each metric
    metrics = ['health_score', 'coral_coverage', 'species_count', 'temperature_c']
    metric_names = ['Health Score', 'Coral Coverage (%)', 'Species Count', 'Temperature (°C)']
    
    selected_metric = st.selectbox("Select Metric for Comparison:", 
                                  options=metrics, 
                                  format_func=lambda x: metric_names[metrics.index(x)])
    
    fig = px.box(filtered_df, x='site', y=selected_metric, 
                 title=f"{metric_names[metrics.index(selected_metric)]} by Site")
    st.plotly_chart(fig, use_container_width=True)

def research_database():
    st.header("🗄️ Research Database")
    st.markdown("Coral species database and research resources")
    
    # Species database
    st.subheader("🐠 Species Database")
    classifier = st.session_state.classifier
    
    # Create DataFrame from species database
    species_data = []
    for species, info in classifier.species_database.items():
        species_data.append({
            'Scientific Name': species,
            'Common Name': info['common_name'],
            'Family': info['family'],
            'Growth Form': info['growth_form'],
            'Threat Level': info['threat_level'],
            'Description': info['description']
        })
    
    species_df = pd.DataFrame(species_data)
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        family_filter = st.multiselect("Filter by Family:", 
                                      options=species_df['Family'].unique(),
                                      default=species_df['Family'].unique())
    with col2:
        threat_filter = st.multiselect("Filter by Threat Level:",
                                      options=species_df['Threat Level'].unique(),
                                      default=species_df['Threat Level'].unique())
    
    # Apply filters
    filtered_species = species_df[
        (species_df['Family'].isin(family_filter)) &
        (species_df['Threat Level'].isin(threat_filter))
    ]
    
    # Display table
    st.dataframe(filtered_species, use_container_width=True)
    
    # Species statistics
    st.subheader("📊 Database Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Species", len(species_df))
        
    with col2:
        threat_counts = species_df['Threat Level'].value_counts()
        endangered_count = sum(1 for level in threat_counts.index if 'Endangered' in level)
        st.metric("Endangered Species", endangered_count)
    
    with col3:
        st.metric("Families Represented", len(species_df['Family'].unique()))
    
    # Threat level distribution
    fig = px.pie(species_df, names='Threat Level', 
                title="Species Distribution by Threat Level",
                color_discrete_map={
                    'Critically Endangered': '#FF4444',
                    'Endangered': '#FF8800',
                    'Near Threatened': '#FFD700',
                    'Least Concern': '#00C851'
                })
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth form distribution
    fig = px.histogram(species_df, x='Growth Form', 
                      title="Species Distribution by Growth Form",
                      color='Growth Form')
    st.plotly_chart(fig, use_container_width=True)
    
    # Research resources
    st.subheader("📚 Research Resources")
    
    resources = [
        {"title": "Coral Species Identification Guide", "type": "PDF", "size": "15.2 MB"},
        {"title": "Health Assessment Protocols", "type": "PDF", "size": "8.7 MB"},
        {"title": "Monitoring Best Practices", "type": "PDF", "size": "12.3 MB"},
        {"title": "Statistical Analysis Templates", "type": "R Script", "size": "2.1 MB"},
        {"title": "Image Processing Workflows", "type": "Python", "size": "5.4 MB"}
    ]
    
    for resource in resources:
        with st.expander(f"📄 {resource['title']}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"Type: {resource['type']}")
                st.write(f"Size: {resource['size']}")
            with col2:
                st.button(f"Download", key=f"download_{resource['title']}")

if __name__ == "__main__":
    main()
