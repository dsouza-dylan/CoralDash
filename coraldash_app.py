import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
from typing import Dict, List, Tuple
import json

# Configure the page
st.set_page_config(
    page_title="Coral Species Identifier",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive coral and algae species database
CORAL_SPECIES_DB = {
    # ACROPORA SPECIES
    "Acropora cervicornis": {
        "common_name": "Staghorn Coral",
        "description": "Fast-growing branching coral with cylindrical branches resembling antlers",
        "habitat": "Shallow reef environments, 1-30m depth",
        "conservation_status": "Critically Endangered",
        "characteristics": ["Branching growth form", "Cylindrical branches", "Small polyps", "Fast growth rate"],
        "distribution": "Caribbean, Western Atlantic",
        "category": "Hard Coral"
    },
    "Acropora palmata": {
        "common_name": "Elkhorn Coral",
        "description": "Large branching coral with flattened, plate-like branches",
        "habitat": "Shallow reef crests, 1-20m depth",
        "conservation_status": "Critically Endangered",
        "characteristics": ["Flattened branches", "Plate-like structure", "Fast growth", "Wave-resistant"],
        "distribution": "Caribbean, Western Atlantic",
        "category": "Hard Coral"
    },
    "Acropora millepora": {
        "common_name": "Small Polyp Staghorn",
        "description": "Branching coral with dense, small polyps and fine skeletal structure",
        "habitat": "Reef slopes and lagoons, 5-40m depth",
        "conservation_status": "Near Threatened",
        "characteristics": ["Dense branching", "Small polyps", "Fine skeleton", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Acropora tenuis": {
        "common_name": "Slender Staghorn",
        "description": "Delicate branching coral with thin, tapering branches",
        "habitat": "Protected reef areas, 3-25m depth",
        "conservation_status": "Near Threatened",
        "characteristics": ["Thin branches", "Delicate structure", "Small polyps", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Acropora digitifera": {
        "common_name": "Finger Acropora",
        "description": "Robust branching coral with thick, finger-like projections",
        "habitat": "Shallow reefs, 1-15m depth",
        "conservation_status": "Near Threatened",
        "characteristics": ["Thick branches", "Finger-like growth", "Robust structure", "Fast growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Acropora formosa": {
        "common_name": "Beautiful Staghorn",
        "description": "Elegant branching coral with symmetrical growth pattern",
        "habitat": "Reef slopes, 5-30m depth",
        "conservation_status": "Vulnerable",
        "characteristics": ["Symmetrical branching", "Elegant structure", "Medium-sized polyps", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    
    # POCILLOPORA SPECIES
    "Pocillopora damicornis": {
        "common_name": "Cauliflower Coral",
        "description": "Branching coral with short, thick branches and warty surface texture",
        "habitat": "Shallow reefs, tide pools, 0-40m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Short thick branches", "Warty surface", "Hardy species", "Fast growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Pocillopora verrucosa": {
        "common_name": "Brush Coral",
        "description": "Densely branching coral with fine, brush-like appearance",
        "habitat": "Shallow reef areas, 1-25m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Dense branching", "Brush-like appearance", "Small branches", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Pocillopora eydouxi": {
        "common_name": "Cluster Coral",
        "description": "Compact branching coral forming dense clusters",
        "habitat": "Reef flats and slopes, 2-30m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Compact growth", "Dense clusters", "Short branches", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    
    # MONTIPORA SPECIES
    "Montipora capitata": {
        "common_name": "Rice Coral",
        "description": "Plate-forming or branching coral with small tubercles covering the surface",
        "habitat": "Shallow reefs, 1-30m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Plate or branching form", "Small tubercles", "Smooth surface", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Montipora monasteriata": {
        "common_name": "Spiny Cup Coral",
        "description": "Encrusting to massive coral with distinctive spiny polyps",
        "habitat": "Reef slopes, 5-40m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Encrusting growth", "Spiny polyps", "Massive form", "Slow growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Montipora digitata": {
        "common_name": "Finger Montipora",
        "description": "Branching coral with finger-like projections and small polyps",
        "habitat": "Shallow lagoons, 3-20m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Finger-like branches", "Small polyps", "Delicate structure", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    
    # PORITES SPECIES
    "Porites cylindrica": {
        "common_name": "Cylinder Coral",
        "description": "Branching coral with cylindrical branches and very small polyps",
        "habitat": "Shallow reefs, lagoons, 1-25m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Cylindrical branches", "Very small polyps", "Dense skeleton", "Slow growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Porites lobata": {
        "common_name": "Lobe Coral",
        "description": "Massive coral forming large lobes with tiny polyps",
        "habitat": "Reef slopes, 3-50m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Massive lobes", "Tiny polyps", "Long-lived", "Very slow growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Porites rus": {
        "common_name": "Small Polyp Finger Coral",
        "description": "Branching coral with small finger-like projections",
        "habitat": "Shallow reefs, 2-20m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Small finger branches", "Tiny polyps", "Compact growth", "Slow growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Porites porites": {
        "common_name": "Finger Coral",
        "description": "Branching coral with finger-like projections and small polyps",
        "habitat": "Shallow reefs, lagoons, 1-20m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Finger-like branches", "Small polyps", "Dense skeleton", "Moderate growth"],
        "distribution": "Caribbean, Western Atlantic",
        "category": "Hard Coral"
    },
    
    # PAVONA SPECIES
    "Pavona clavus": {
        "common_name": "Leaf Coral",
        "description": "Plate-forming coral with thin, leaf-like vertical plates",
        "habitat": "Reef slopes, 5-50m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Leaf-like plates", "Vertical growth", "Thin structure", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Pavona decussata": {
        "common_name": "Cactus Coral",
        "description": "Branching coral with flattened, cactus-like plates",
        "habitat": "Reef slopes, 3-40m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Flattened plates", "Cactus-like growth", "Branching form", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Pavona varians": {
        "common_name": "Variable Leaf Coral",
        "description": "Coral with variable growth forms from encrusting to plate-like",
        "habitat": "Reef slopes, 5-30m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Variable growth forms", "Encrusting to plating", "Adaptable", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    
    # FUNGIA SPECIES
    "Fungia granulosa": {
        "common_name": "Granular Mushroom Coral",
        "description": "Solitary, disc-shaped coral with granular surface texture",
        "habitat": "Sandy bottoms, reef slopes, 5-40m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Solitary disc shape", "Granular surface", "Free-living", "Slow growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Fungia scutaria": {
        "common_name": "Plate Coral",
        "description": "Large, plate-shaped solitary coral with radiating septa",
        "habitat": "Sandy areas, reef slopes, 3-35m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Large plate shape", "Radiating septa", "Solitary", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Fungia fungites": {
        "common_name": "Mushroom Coral",
        "description": "Oval-shaped solitary coral with prominent central mouth",
        "habitat": "Sandy bottoms, 5-30m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Oval shape", "Central mouth", "Free-living", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    
    # ALGAE SPECIES
    "Lobophora variegata": {
        "common_name": "Fan Algae",
        "description": "Brown algae forming fan-shaped or cup-shaped thalli",
        "habitat": "Coral reefs, rocky substrates, 0-60m depth",
        "conservation_status": "Not Evaluated",
        "characteristics": ["Fan-shaped thalli", "Brown coloration", "Flexible structure", "Fast growth"],
        "distribution": "Tropical and subtropical waters worldwide",
        "category": "Brown Algae"
    },
    "Peyssonnelia sp.": {
        "common_name": "Encrusting Red Algae",
        "description": "Calcified red algae forming thin, encrusting layers",
        "habitat": "Coral reefs, hard substrates, 0-100m depth",
        "conservation_status": "Not Evaluated",
        "characteristics": ["Encrusting growth", "Calcified structure", "Pink to red color", "Slow growth"],
        "distribution": "Tropical and temperate waters worldwide",
        "category": "Red Algae"
    },
    "Turf Algae": {
        "common_name": "Turf Algae",
        "description": "Short, dense mat of mixed filamentous algae species",
        "habitat": "Coral reefs, rocky surfaces, 0-40m depth",
        "conservation_status": "Not Evaluated",
        "characteristics": ["Short filaments", "Dense mat", "Mixed species", "Fast growth"],
        "distribution": "Worldwide",
        "category": "Mixed Algae"
    },
    "Crustose Coralline Algae": {
        "common_name": "CCA",
        "description": "Calcified algae forming hard, pink crusts on reef surfaces",
        "habitat": "Coral reefs, rocky substrates, 0-150m depth",
        "conservation_status": "Not Evaluated",
        "characteristics": ["Calcified crust", "Pink coloration", "Hard structure", "Very slow growth"],
        "distribution": "Worldwide",
        "category": "Red Algae"
    },
    
    # ADDITIONAL CORAL SPECIES
    "Stylophora pistillata": {
        "common_name": "Smooth Cauliflower Coral",
        "description": "Branching coral with smooth, rounded branch tips",
        "habitat": "Shallow reefs, 1-40m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Smooth branches", "Rounded tips", "Dense growth", "Fast growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Seriatopora hystrix": {
        "common_name": "Needle Coral",
        "description": "Delicate branching coral with needle-like branch tips",
        "habitat": "Protected reef areas, 3-30m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Needle-like tips", "Delicate structure", "Fine branching", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Galaxea fascicularis": {
        "common_name": "Galaxy Coral",
        "description": "Massive coral with large, prominent polyps arranged in clusters",
        "habitat": "Reef slopes, 5-40m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Large polyps", "Clustered arrangement", "Massive form", "Aggressive"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Turbinaria reniformis": {
        "common_name": "Yellow Scroll Coral",
        "description": "Plate coral forming scroll-like or cup-shaped structures",
        "habitat": "Reef slopes, 10-50m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Scroll-like plates", "Yellow coloration", "Cup-shaped", "Moderate growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    },
    "Platygyra daedalea": {
        "common_name": "Brain Coral",
        "description": "Massive coral with meandering valleys resembling brain tissue",
        "habitat": "Reef slopes, 3-50m depth",
        "conservation_status": "Least Concern",
        "characteristics": ["Meandering valleys", "Brain-like appearance", "Massive form", "Slow growth"],
        "distribution": "Indo-Pacific",
        "category": "Hard Coral"
    }
}

def analyze_coral_image(image: Image.Image) -> Dict:
    """
    Mock function to analyze coral image and return species prediction
    In a real implementation, this would use a trained ML model
    """
    # Simulate image analysis
    image_array = np.array(image)
    
    # Mock analysis based on image properties
    height, width = image_array.shape[:2]
    avg_color = np.mean(image_array, axis=(0, 1))
    
    # Mock prediction logic (replace with actual ML model)
    species_list = list(CORAL_SPECIES_DB.keys())
    
    # Simulate confidence scores
    np.random.seed(42)  # For reproducible results
    confidences = np.random.dirichlet(np.ones(len(species_list)), size=1)[0]
    
    # Sort by confidence
    predictions = sorted(zip(species_list, confidences), key=lambda x: x[1], reverse=True)
    
    return {
        "predictions": predictions,
        "image_stats": {
            "dimensions": f"{width}x{height}",
            "avg_color": avg_color.tolist() if len(avg_color) > 1 else [avg_color],
            "file_size": len(image.tobytes())
        }
    }

def display_species_info(species_name: str, confidence: float):
    """Display detailed information about a coral species"""
    if species_name in CORAL_SPECIES_DB:
        species_info = CORAL_SPECIES_DB[species_name]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Conservation status with color coding
            status = species_info["conservation_status"]
            if status == "Critically Endangered":
                st.error(f"Status: {status}")
            elif status == "Near Threatened":
                st.warning(f"Status: {status}")
            else:
                st.success(f"Status: {status}")
        
        with col2:
            st.write(f"**Common Name:** {species_info['common_name']}")
            st.write(f"**Description:** {species_info['description']}")
            st.write(f"**Habitat:** {species_info['habitat']}")
            st.write(f"**Distribution:** {species_info['distribution']}")
            
            # Characteristics
            st.write("**Key Characteristics:**")
            for char in species_info['characteristics']:
                st.write(f"• {char}")

def main():
    st.title("🪸 Coral Species Identifier")
    st.markdown("Upload an image of coral to identify the species and learn about its characteristics.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About This Tool")
        st.write("""
        This coral species identifier helps marine biologists, divers, and coral enthusiasts 
        identify different coral species from photographs.
        """)
        
        st.header("How to Use")
        st.write("""
        1. Upload a clear image of coral
        2. Wait for the analysis to complete
        3. Review the species predictions
        4. Explore detailed species information
        """)
        
        st.header("Supported Categories")
        categories = {}
        for species, info in CORAL_SPECIES_DB.items():
            category = info.get("category", "Unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(species)
        
        for category, species_list in categories.items():
            st.write(f"**{category}:** {len(species_list)} species")
            
        st.header("Species Count by Genus")
        genus_counts = {}
        for species in CORAL_SPECIES_DB.keys():
            genus = species.split()[0] if " " in species else species
            genus_counts[genus] = genus_counts.get(genus, 0) + 1
        
        for genus, count in sorted(genus_counts.items()):
            st.write(f"• {genus}: {count}")
        
        st.header("Tips for Best Results")
        st.write("""
        • Use high-resolution images
        • Ensure good lighting
        • Capture coral structure clearly
        • Avoid blurry or dark images
        """)
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Choose a coral image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of coral for species identification"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Coral Image for Analysis", use_column_width=True)
            
            # Image information
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Format:** {image.format}")
        
        with col2:
            st.subheader("Analysis Results")
            
            # Analyze the image
            with st.spinner("Analyzing coral image..."):
                results = analyze_coral_image(image)
            
            # Display predictions
            st.write("**Species Predictions:**")
            
            for i, (species, confidence) in enumerate(results["predictions"][:3]):
                with st.expander(f"#{i+1} {species} ({confidence:.2%})", expanded=(i==0)):
                    display_species_info(species, confidence)
        
        # Additional analysis section
        st.subheader("📊 Image Analysis Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Image Dimensions", results["image_stats"]["dimensions"])
        
        with col2:
            avg_color = results["image_stats"]["avg_color"]
            if len(avg_color) >= 3:
                st.write("**Average Color (RGB):**")
                st.write(f"R: {avg_color[0]:.0f}, G: {avg_color[1]:.0f}, B: {avg_color[2]:.0f}")
            else:
                st.write("**Average Intensity:**")
                st.write(f"{avg_color[0]:.0f}")
        
        with col3:
            file_size_mb = results["image_stats"]["file_size"] / (1024 * 1024)
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        
        # Confidence chart
        st.subheader("📈 Prediction Confidence")
        
        # Create a dataframe for the chart
        chart_data = pd.DataFrame({
            'Species': [pred[0] for pred in results["predictions"]],
            'Confidence': [pred[1] for pred in results["predictions"]]
        })
        
        st.bar_chart(chart_data.set_index('Species'))
        
        # Download results
        st.subheader("💾 Download Results")
        
        results_json = {
            "filename": uploaded_file.name,
            "predictions": [{"species": pred[0], "confidence": float(pred[1])} for pred in results["predictions"]],
            "image_stats": results["image_stats"]
        }
        
        st.download_button(
            label="Download Analysis Results (JSON)",
            data=json.dumps(results_json, indent=2),
            file_name=f"coral_analysis_{uploaded_file.name.split('.')[0]}.json",
            mime="application/json"
        )
    
    else:
        # Show example/demo section when no image is uploaded
        st.info("👆 Upload a coral image to get started!")
        
        st.subheader("🔬 What This Tool Does")
        st.write("""
        This coral species identifier uses image analysis to help identify coral species from photographs. 
        The tool analyzes visual characteristics such as growth patterns, polyp structure, and overall morphology 
        to provide species predictions with confidence scores.
        """)
        
        st.subheader("🌊 Coral Conservation")
        st.write("""
        Coral reefs are among the most biodiverse ecosystems on Earth, but they face significant threats from 
        climate change, pollution, and human activities. Accurate species identification is crucial for:
        
        - Monitoring reef health and biodiversity
        - Tracking population changes over time
        - Guiding conservation efforts
        - Supporting marine research initiatives
        """)
        
        # Sample species showcase
        st.subheader("🪸 Featured Species")
        
        for species_name, info in list(CORAL_SPECIES_DB.items())[:2]:
            with st.expander(f"{info['common_name']} ({species_name})"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    status = info["conservation_status"]
                    if status == "Critically Endangered":
                        st.error(f"Status: {status}")
                    elif status == "Near Threatened":
                        st.warning(f"Status: {status}")
                    else:
                        st.success(f"Status: {status}")
                
                with col2:
                    st.write(info["description"])
                    st.write(f"**Habitat:** {info['habitat']}")

if __name__ == "__main__":
    main()
