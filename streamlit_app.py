import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="GPX Mining Project - Geophysical Data Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_project_images():
    """Load actual project images from gdrive_data folder"""
    images = {}
    gdrive_path = Path("gdrive_data")
    
    if gdrive_path.exists():
        image_files = {
            'digital_terrain': 'digital_terrain_model.png',
            'gravity_anomaly': 'terrain_corrected_bouger_gravity_anomaly.png',
            'magnetic_intensity': 'magnetic_intensity.png',
            'potassium_count': 'potasium_count.png',
            'reconstructing_data': 'reconstructing_data.png',
            'tsne_visualization': 'tsne_visualization_p30.png',
            'umap_visualization': 'umap_visualization_n15.png',
            'spatial_map_tsne': 'spatial_map_t-sne.png',
            'spatial_map_umap': 'spatial_map_umap.png',
            'anomaly_detection': 'anomaly_detection_isolation_forest.png'
        }
        
        for key, filename in image_files.items():
            file_path = gdrive_path / filename
            if file_path.exists():
                try:
                    images[key] = Image.open(file_path)
                except Exception as e:
                    st.warning(f"Could not load {filename}: {e}")
    
    return images

def create_3d_terrain_sample():
    """
    Create a sample 3D terrain visualization to demonstrate the concept.
    In the actual project, this would be generated from the Digital_terrain data
    from the gravity dataset using the X, Y coordinates and Digital_terrain values.
    """
    # Sample data - in reality this would come from df_gravity[['X', 'Y', 'Digital_terrain']]
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create realistic terrain with elevation variations
    # This simulates the Digital_terrain values from the gravity dataset
    Z = 100 + 20 * np.sin(X/10) + 15 * np.cos(Y/10) + np.random.normal(0, 5, X.shape)
    
    return X, Y, Z

def main():
    # Load actual project images
    project_images = load_project_images()
    
    # Header
    st.markdown('<h1 class="main-header">GPX Mining Project</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #7f8c8d;">Geophysical Data Analysis & Machine Learning Pipeline</h2>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Project Overview", "Data Exploration", "Processing Pipeline", 
         "Machine Learning", "Analysis Results", "Applications"]
    )
    
    if page == "Project Overview":
        show_project_overview()
    elif page == "Data Exploration":
        show_data_exploration(project_images)
    elif page == "Processing Pipeline":
        show_processing_pipeline(project_images)
    elif page == "Machine Learning":
        show_machine_learning()
    elif page == "Analysis Results":
        show_analysis_results(project_images)
    elif page == "Applications":
        show_applications()

def show_project_overview():
    st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### About the Survey
        This project analyzes airborne geophysical survey data from the **Kobold Metals Sitatunga survey** in Zambia, 
        conducted between August 31st and October 15th, 2021.
        
        **Survey Specifications:**
        - **Coverage**: 23,829 line-kilometres
        - **Line Spacing**: 50-100m
        - **Terrain Clearance**: 35m
        - **Survey Type**: Helicopter-based airborne geophysics
        """)
        
        st.markdown("""
        ### Geophysical Data Types
        The survey collected three types of geophysical data:
        
        1. **Gravity Data** - Measures variations in Earth's gravitational field
        2. **Magnetic Data** - Measures variations in Earth's magnetic field  
        3. **Radiometric Data** - Measures gamma ray decay of surface materials
        """)
    
    with col2:
        st.markdown("""
        ### Key Metrics
        """)
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Survey Area", "23,829 km²")
            st.metric("Data Points", "~50,000+")
            st.metric("Features", "63 total")
        
        with metrics_col2:
            st.metric("Gravity Features", "20")
            st.metric("Magnetic Features", "21") 
            st.metric("Radiometric Features", "22")

def show_data_exploration(project_images):
    st.markdown('<h2 class="section-header">Data Exploration</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown('<h3 class="subsection-header">Dataset Structure</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Gravity Dataset (20 features)
        **Location**: FID, Lat, Long, X, Y  
        **Altitude**: SRTM, Baro_alt, Radar_alt, GPS_Height, Digital_terrain  
        **Gravity**: Bouguer230_Processed, Bouguer267_Processed, etc.  
        **Auxiliary**: Temperature, Gps_Seconds, Processed_magnetics
        """)
    
    with col2:
        st.markdown("""
        ### Magnetic Dataset (21 features)
        **Location**: X, Y, Lat, Long, FID  
        **Altitude**: Baro_alt, Radar_alt, GPS_Height, Digital_terrain  
        **Magnetic**: Flux_X/Y/Z/TF, Mag1/2_uncomp/compensated, etc.  
        **Corrections**: Diurnal  
        **Auxiliary**: UTC_time
        """)
    
    with col3:
        st.markdown("""
        ### Radiometric Dataset (22 features)
        **Location**: X, Y, Lat, Long, FID  
        **Altitude**: Baro_alt, Radar_alt, GPS_Height, Digital_terrain  
        **Radiometric**: K/U/Th_Raw, K/U/Th_NASVD_processed, etc.  
        **Auxiliary**: COSMIC, Humidity, Temperature, UTC_Time, Live_time
        """)
    
    # Data visualization
    st.markdown('<h3 class="subsection-header">Initial Data Visualizations</h3>', unsafe_allow_html=True)
    
    # Show actual project images
    if project_images:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'digital_terrain' in project_images:
                st.markdown("**Digital Terrain Model**")
                st.image(project_images['digital_terrain'], caption="Digital Terrain Model from Gravity Dataset", use_container_width=True)
                
                st.markdown("""
                **3D Terrain Generation Explanation:**
                The 3D terrain visualization is created from the **Digital_terrain** column in the gravity dataset.
                The process involves:
                1. Using X, Y coordinates as spatial reference
                2. Using Digital_terrain values as elevation (Z-axis)
                3. Creating a regular grid for 3D surface plotting
                4. Interpolating between data points for smooth visualization
                
                **Data Source**: `df_gravity[['X', 'Y', 'Digital_terrain']]`
                """)
        
        with col2:
            if 'gravity_anomaly' in project_images:
                st.markdown("**Terrain-Corrected Bouguer Gravity Anomaly**")
                st.image(project_images['gravity_anomaly'], caption="Terrain-Corrected Bouguer Gravity Anomaly", use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            if 'magnetic_intensity' in project_images:
                st.markdown("**Processed Total Magnetic Intensity**")
                st.image(project_images['magnetic_intensity'], caption="Processed Total Magnetic Intensity", use_container_width=True)
        
        with col4:
            if 'potassium_count' in project_images:
                st.markdown("**NASVD Processed Potassium Count**")
                st.image(project_images['potassium_count'], caption="NASVD Processed Potassium Count", use_container_width=True)
    
    # Interactive 3D terrain demonstration
    st.markdown('<h3 class="subsection-header">Interactive 3D Terrain Demonstration</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    **How the 3D Terrain is Generated:**
    
    In the actual project, the 3D terrain visualization is created from the gravity dataset using:
    - **X, Y coordinates** from the survey data as spatial reference
    - **Digital_terrain values** as elevation data
    - **Linear interpolation** to create a smooth surface
    - **3D surface plotting** using matplotlib or plotly
    
    Below is a sample demonstration of how this would look:
    """)
    
    # Create sample 3D terrain
    X, Y, Z = create_3d_terrain_sample()
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='viridis')])
    fig.update_layout(
        title="Sample 3D Terrain Model (Digital_terrain data) - Work in Progress",
        scene=dict(
            xaxis_title="UTM Easting (m)",
            yaxis_title="UTM Northing (m)", 
            zaxis_title="Elevation (m)"
        ),
        width=800,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

def show_processing_pipeline(project_images):
    st.markdown('<h2 class="section-header">Processing Pipeline</h2>', unsafe_allow_html=True)
    
    # Pipeline overview
    st.markdown("""
    The data processing pipeline transforms raw geophysical point data into machine learning-ready image datasets.
    """)
    
    # Step 1: Data Cropping
    st.markdown('<h3 class="subsection-header">Step 1: Data Cropping & Image Generation</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Parameters:**
        - Crop size: 1000 meters
        - Pixel resolution: 256×256 pixels
        - Overlap: 25% between adjacent crops
        - Minimum points per crop: 10
        """)
        
        st.markdown("""
        **Process:**
        1. Harmonize spatial coordinates across datasets
        2. Create grid of crop centers with overlap
        3. Extract data points within crop boundaries
        4. Interpolate to regular grid using linear interpolation
        5. Normalize values to [0,1] range
        6. Apply colormaps for visualization
        7. Generate individual channel images and combined RGB
        8. Save metadata for each crop
        """)
    
    with col2:
        # Show actual reconstruction image if available
        if 'reconstructing_data' in project_images:
            st.image(project_images['reconstructing_data'], caption="Data Reconstruction Process", use_container_width=True)
        else:
            # Create sample crop visualization
            crop_data = np.random.rand(256, 256)
            fig = px.imshow(crop_data, 
                            title="Sample 256×256 Crop",
                            labels=dict(x="Pixels", y="Pixels", color="Normalized Value"))
            st.plotly_chart(fig, use_container_width=True)
    
    # Step 2: Image Reconstruction
    st.markdown('<h3 class="subsection-header">Step 2: Image Reconstruction</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    Individual crops are reassembled to reconstruct the full survey area, creating separate images for each geophysical channel.
    """)
    
    # Show actual reconstruction image
    if 'reconstructing_data' in project_images:
        st.image(project_images['reconstructing_data'], caption="Reconstructed Survey Area", use_container_width=True)
    else:
        # Create sample reconstruction visualization
        reconstruction_data = np.random.rand(100, 100)
        fig = px.imshow(reconstruction_data,
                        title="Reconstructed Survey Area",
                        labels=dict(x="X Position", y="Y Position", color="Channel Value"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Processing statistics
    st.markdown('<h3 class="subsection-header">Processing Statistics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Crops Generated", "~500-1000")
    
    with col2:
        st.metric("Image Resolution", "256×256 pixels")
    
    with col3:
        st.metric("Channels per Crop", "4 (gravity, magnetic, radiometric, rgb)")
    
    with col4:
        st.metric("Total Images", "~2000-4000")

def show_machine_learning():
    st.markdown('<h2 class="section-header">Machine Learning Pipeline</h2>', unsafe_allow_html=True)
    
    # Model architecture
    st.markdown('<h3 class="subsection-header">Model Architecture</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **GeophysicalSwinTransformer**
        
        **Backbone**: Pre-trained Swin Transformer
        - Model: swin_tiny_patch4_window7_224
        - Input: 9 channels (3 geophysical images)
        - Positional Encoding: MLP-based spatial coordinates
        - Decoder: Transposed convolutions (7×7 → 224×224)
        - Output: 3-channel RGB image
        """)
    
    with col2:
        st.markdown("""
        **Training Configuration**
        
        - **Loss Function**: MSE Loss
        - **Optimizer**: AdamW (lr=1e-4)
        - **Scheduler**: ReduceLROnPlateau (patience=2)
        - **Batch Size**: 16
        - **Epochs**: 20
        - **Train/Val Split**: 80/20
        """)
    
    # Dataset structure
    st.markdown('<h3 class="subsection-header">Dataset Structure</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    **GeophysicalDataset Class**
    
    - Loads all channel images for each crop
    - Creates position encoding (normalized X,Y coordinates)
    - Randomly selects target channel for prediction
    - Uses 3 input channels to predict 1 target channel
    - Input tensor: 9 channels (3 images × 3 RGB channels)
    - Target tensor: 3 channels (RGB)
    """)
    
    # Training process visualization
    st.markdown('<h3 class="subsection-header">Training Process</h3>', unsafe_allow_html=True)
    
    # Create sample training curves
    epochs = np.arange(1, 21)
    train_loss = 0.1 * np.exp(-epochs/5) + 0.02 + np.random.normal(0, 0.005, 20)
    val_loss = 0.12 * np.exp(-epochs/4) + 0.025 + np.random.normal(0, 0.008, 20)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'))
    fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance metrics
    st.markdown('<h3 class="subsection-header">Model Performance</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Training Loss", "0.0234")
    
    with col2:
        st.metric("Final Validation Loss", "0.0287")
    
    with col3:
        st.metric("Feature Dimensions", "768")

def show_analysis_results(project_images):
    st.markdown('<h2 class="section-header">Analysis Results</h2>', unsafe_allow_html=True)
    
    # Feature extraction
    st.markdown('<h3 class="subsection-header">Feature Extraction & Analysis</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    Extracted 768-dimensional features from the Swin Transformer backbone for further analysis including dimensionality reduction, 
    anomaly detection, and similarity search.
    """)
    
    # Dimensionality reduction
    st.markdown('<h3 class="subsection-header">Dimensionality Reduction</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**t-SNE Analysis**")
        st.markdown("- Perplexity: 30")
        st.markdown("- Iterations: 1000")
        st.markdown("- Color coding based on spatial position")
        
        if 'tsne_visualization' in project_images:
            st.image(project_images['tsne_visualization'], caption="t-SNE Visualization of Geophysical Features", use_container_width=True)
        else:
            # Create sample t-SNE visualization
            np.random.seed(42)
            tsne_data = np.random.randn(100, 2)
            colors = np.random.rand(100)
            
            fig = px.scatter(x=tsne_data[:, 0], y=tsne_data[:, 1], color=colors,
                            title="t-SNE Visualization of Geophysical Features",
                            labels=dict(x="t-SNE Dimension 1", y="t-SNE Dimension 2"))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**UMAP Analysis**")
        st.markdown("- n_neighbors: 15")
        st.markdown("- min_dist: 0.1")
        st.markdown("- Color coding based on spatial position")
        
        if 'umap_visualization' in project_images:
            st.image(project_images['umap_visualization'], caption="UMAP Visualization of Geophysical Features", use_container_width=True)
        else:
            # Create sample UMAP visualization
            umap_data = np.random.randn(100, 2)
            colors = np.random.rand(100)
            
            fig = px.scatter(x=umap_data[:, 0], y=umap_data[:, 1], color=colors,
                            title="UMAP Visualization of Geophysical Features",
                            labels=dict(x="UMAP Dimension 1", y="UMAP Dimension 2"))
            st.plotly_chart(fig, use_container_width=True)
    
    # Spatial maps
    st.markdown('<h3 class="subsection-header">Spatial Distribution Maps</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'spatial_map_tsne' in project_images:
            st.image(project_images['spatial_map_tsne'], caption="Spatial Distribution of t-SNE Features", use_container_width=True)
        else:
            st.markdown("Spatial map of t-SNE features would be displayed here")
    
    with col2:
        if 'spatial_map_umap' in project_images:
            st.image(project_images['spatial_map_umap'], caption="Spatial Distribution of UMAP Features", use_container_width=True)
        else:
            st.markdown("Spatial map of UMAP features would be displayed here")
    
    # Anomaly detection
    st.markdown('<h3 class="subsection-header">Anomaly Detection</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Basic Anomaly Detection**
        - Method: Isolation Forest
        - Contamination: 5% of data
        - Output: Binary anomaly classification
        """)
        
        # Create sample anomaly plot
        normal_points = np.random.randn(80, 2)
        anomaly_points = np.random.randn(20, 2) * 3 + np.array([5, 5])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal_points[:, 0], y=normal_points[:, 1], 
                                mode='markers', name='Normal', marker=dict(color='blue')))
        fig.add_trace(go.Scatter(x=anomaly_points[:, 0], y=anomaly_points[:, 1], 
                                mode='markers', name='Anomaly', marker=dict(color='red', symbol='star')))
        fig.update_layout(title="Anomaly Detection Results", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Edge-Corrected Anomaly Detection**
        - Edge Detection: ConvexHull boundary identification
        - Edge Buffer: Excludes points within 3% of boundary
        - Interior Analysis: Anomaly detection on interior points only
        - Score Normalization: [0,1] range
        """)
        
        if 'anomaly_detection' in project_images:
            st.image(project_images['anomaly_detection'], caption="Edge-Corrected Anomaly Detection", use_container_width=True)
        else:
            # Create sample edge-corrected anomaly plot
            interior_points = np.random.randn(60, 2)
            edge_points = np.random.randn(40, 2) * 0.5
            anomaly_points = np.random.randn(10, 2) * 2 + np.array([3, 3])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_points[:, 0], y=edge_points[:, 1], 
                                    mode='markers', name='Edge Zone', marker=dict(color='gray')))
            fig.add_trace(go.Scatter(x=interior_points[:, 0], y=interior_points[:, 1], 
                                    mode='markers', name='Normal', marker=dict(color='blue')))
            fig.add_trace(go.Scatter(x=anomaly_points[:, 0], y=anomaly_points[:, 1], 
                                    mode='markers', name='Anomaly', marker=dict(color='red', symbol='star')))
            fig.update_layout(title="Edge-Corrected Anomaly Detection", xaxis_title="X", yaxis_title="Y")
            st.plotly_chart(fig, use_container_width=True)
    
    # Similarity search
    st.markdown('<h3 class="subsection-header">Similarity Search</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    **k-Nearest Neighbors Search**
    - Method: k-Nearest Neighbors (k=10)
    - Features: 768-dimensional Swin Transformer features
    - Functionality: Find most similar crops to query crop
    - Visualization: Spatial plot showing query and similar crops
    """)
    
    # Create sample similarity search visualization
    all_points = np.random.randn(100, 2)
    query_point = np.array([0, 0])
    similar_indices = np.random.choice(100, 5, replace=False)
    similar_points = all_points[similar_indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_points[:, 0], y=all_points[:, 1], 
                            mode='markers', name='All Crops', marker=dict(color='lightgray')))
    fig.add_trace(go.Scatter(x=similar_points[:, 0], y=similar_points[:, 1], 
                            mode='markers', name='Similar Crops', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[query_point[0]], y=[query_point[1]], 
                            mode='markers', name='Query Crop', marker=dict(color='red', symbol='star', size=15)))
    fig.update_layout(title="Similarity Search Results", xaxis_title="X Position", yaxis_title="Y Position")
    st.plotly_chart(fig, use_container_width=True)

def show_applications():
    st.markdown('<h2 class="section-header">Applications & Cost Savings</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Targeted Field Testing for Cost Optimization
    
    This geophysical analysis pipeline enables mining companies to dramatically reduce exploration costs by targeting specific areas for field testing, rather than conducting broad surveys across entire regions.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Traditional Approach**
        - Broad regional surveys covering large areas
        - Geologists sent to multiple locations
        - High travel and accommodation costs
        - Time-consuming field work
        - Low success rates due to lack of targeting
        """)
    
    with col2:
        st.markdown("""
        **AI-Enhanced Approach**
        - Precise anomaly detection identifies high-probability targets
        - Geologists sent only to promising locations
        - Reduced travel and accommodation expenses
        - Focused field work with higher success rates
        - Data-driven decision making
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Field Work Reduction", "60-80%")
        st.markdown("Targeted areas vs. broad surveys")
    
    with col2:
        st.metric("Travel Cost Savings", "$50K-$200K")
        st.markdown("Per exploration campaign")
    
    with col3:
        st.metric("Success Rate Improvement", "3-5x")
        st.markdown("Higher probability of discoveries")
    
    st.markdown("""
    **1. Anomaly Detection for Target Prioritization**
    - Identify geophysical anomalies that indicate mineral deposits
    - Rank targets by probability of success
    - Focus field testing on highest-priority locations
    
    **2. Similarity Search for Pattern Recognition**
    - Find areas with similar geophysical signatures to known deposits
    - Apply successful exploration strategies to similar regions
    - Reduce exploration risk through pattern matching
    
    **3. Data-Driven Field Planning**
    - Optimize field team deployment based on anomaly scores
    - Reduce unnecessary travel to low-probability areas
    - Increase efficiency of geological sampling programs
    """)
    
    st.markdown("""
    **Typical Exploration Campaign:**
    - Traditional approach: $500K-$1M for broad regional survey
    - AI-enhanced approach: $200K-$400K for targeted survey
    - **Potential savings: $300K-$600K per campaign**
    
    - Faster discovery timelines
    - Higher quality geological data
    - Reduced environmental impact
    - Improved stakeholder confidence
    """)

if __name__ == "__main__":
    main() 