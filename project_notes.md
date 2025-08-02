# GPX Mining Project - Comprehensive Analysis Notes

## Project Overview
This project analyzes airborne geophysical survey data from the Kobold Metals Sitatunga survey in Zambia (August-October 2021). The survey covered 23,829 line-kilometres with 50-100m line spacing and 35m terrain clearance.

## Data Sources & Structure

### 1. Gravity Dataset (20 features)
**Location Data**: FID, Lat, Long, X, Y
**Altitude Measurements**: SRTM, Baro_alt, Radar_alt, GPS_Height, Digital_terrain
**Gravity Measurements**: 
- Bouguer230_Processed (terrain-corrected using 2.30 g/cm³ density)
- Bouguer267_Processed
- Gravity_disturbance_Raw, Gravity_Disturbance_Levelled, Gravity_disturbance_Processed
- Gravity_Model, Grav_model_Processed
**Auxiliary Data**: Temperature, Gps_Seconds, Processed_magnetics

### 2. Magnetic Dataset (21 features)
**Location Data**: X, Y, Lat, Long, FID
**Altitude Measurements**: Baro_alt, Radar_alt, GPS_Height, Digital_terrain
**Magnetic Measurements**:
- Flux_X, Flux_Y, Flux_Z, Flux_TF
- Mag1_uncomp, Mag2_uncomp, Mag1_compensated, Mag2_compensated
- Gradient_levelled, Processed_magnetics (final processed total magnetic intensity)
**Corrections**: Diurnal
**Auxiliary Data**: UTC_time

### 3. Radiometric Dataset (22 features)
**Location Data**: X, Y, Lat, Long, FID
**Altitude Measurements**: Baro_alt, Radar_alt, GPS_Height, Digital_terrain
**Radiometric Measurements**:
- Raw: Potassium_Raw, Uranium_Raw, Thorium_Raw, Total_Count_Raw
- Processed: Potassium_NASVD_processed, Uranium_NASVD_processed, Thorium_NASVD_processed, Total_count_NASVD_processed
**Auxiliary Data**: COSMIC, Humidity, Temperature, UTC_Time, Live_time

## Data Processing Pipeline

### Phase 1: Data Loading & Initial Visualization
- Load three CSV datasets (Gravity, Magnetic, Radiometric)
- Create initial scatter plots for each dataset showing spatial distribution
- Visualize key features: Digital_terrain, Bouguer230_Processed, Processed_magnetics, Potassium_NASVD_processed

### Phase 2: Data Cropping & Image Generation
**Parameters**:
- Crop size: 1000 meters
- Pixel resolution: 256x256 pixels
- Overlap: 25% between adjacent crops
- Minimum points per crop: 10

**Process**:
1. Harmonize spatial coordinates across all datasets
2. Create grid of crop centers with specified overlap
3. Extract data points within each crop boundary
4. Interpolate data to regular grid using linear interpolation
5. Normalize values to [0,1] range
6. Apply colormaps (viridis for gravity, plasma for magnetic, inferno for radiometric)
7. Generate individual channel images and combined RGB image
8. Save metadata for each crop

### Phase 3: Image Reconstruction
- Reconstruct full survey area from individual crops
- Create separate reconstructed images for each channel (gravity, magnetic, radiometric, rgb)
- Verify data integrity and spatial coverage

### Phase 4: Deep Learning Model Development

#### Dataset Class (GeophysicalDataset)
- Loads all channel images for each crop
- Creates position encoding (normalized X,Y coordinates)
- Randomly selects target channel for prediction
- Uses 3 input channels to predict 1 target channel
- Input tensor: 9 channels (3 images × 3 RGB channels)
- Target tensor: 3 channels (RGB)

#### Model Architecture (GeophysicalSwinTransformer)
- **Backbone**: Pre-trained Swin Transformer (swin_tiny_patch4_window7_224)
- **Input**: 9 channels (3 geophysical images)
- **Positional Encoding**: MLP-based encoding of spatial coordinates
- **Decoder**: Transposed convolutions to upsample from 7×7 to 224×224
- **Output**: 3-channel RGB image

#### Training Process
- **Loss Function**: MSE Loss
- **Optimizer**: AdamW with learning rate 1e-4
- **Scheduler**: ReduceLROnPlateau with patience=2
- **Batch Size**: 16
- **Epochs**: 20
- **Train/Val Split**: 80/20

### Phase 5: Feature Extraction & Analysis

#### Feature Extraction
- Extract 768-dimensional features from Swin Transformer backbone
- Save features, positions, and metadata to NPZ file
- Create feature metadata CSV for easier access

#### Dimensionality Reduction & Visualization
**t-SNE Analysis**:
- Perplexity: 30
- Iterations: 1000
- Color coding based on spatial position

**UMAP Analysis**:
- n_neighbors: 15
- min_dist: 0.1
- Color coding based on spatial position

**Spatial Mapping**:
- Map reduced features back to spatial coordinates
- Visualize spatial distribution of feature dimensions

### Phase 6: Anomaly Detection

#### Basic Anomaly Detection
- **Method**: Isolation Forest
- **Contamination**: 5% of data
- **Output**: Binary anomaly classification

#### Edge-Corrected Anomaly Detection
- **Edge Detection**: Uses ConvexHull to identify survey boundaries
- **Edge Buffer**: Excludes points within 3% distance from boundary
- **Interior Analysis**: Applies anomaly detection only to interior points
- **Score Normalization**: Normalizes anomaly scores to [0,1] range
- **Visualization**: Heatmap of anomaly scores with edge zone highlighting

### Phase 7: Similarity Search
- **Method**: k-Nearest Neighbors (k=10)
- **Features**: 768-dimensional Swin Transformer features
- **Functionality**: Find most similar crops to query crop
- **Visualization**: Spatial plot showing query and similar crops

### Phase 8: Interactive Dashboard
- **Framework**: ipywidgets
- **Modes**: Similarity Search, Anomaly Analysis
- **Features**:
  - Dropdown selection of crops
  - Adjustable number of similar crops
  - Anomaly threshold adjustment
  - Real-time visualization updates

## Key Visualizations Generated

### Initial Data Visualization
1. Digital terrain model (topography)
2. Terrain-corrected Bouguer gravity anomaly
3. Processed total magnetic intensity
4. NASVD processed potassium count

### Analysis Visualizations
1. t-SNE visualization (perplexity=30)
2. UMAP visualization (n_neighbors=15)
3. Spatial maps of reduced features
4. Anomaly detection plots
5. Similarity search spatial plots
6. Edge-corrected anomaly heatmaps

## Technical Implementation Details

### Data Processing
- **Interpolation**: Linear interpolation using scipy.griddata
- **Normalization**: Min-max normalization to [0,1]
- **Colormaps**: Custom colormaps for each geophysical channel
- **Image Format**: PNG files with RGB encoding

### Machine Learning
- **Framework**: PyTorch
- **Model**: Swin Transformer with custom decoder
- **Position Encoding**: MLP-based spatial encoding
- **Training**: MSE loss with AdamW optimizer

### Analysis Methods
- **Dimensionality Reduction**: t-SNE, UMAP
- **Anomaly Detection**: Isolation Forest with edge correction
- **Similarity Search**: k-Nearest Neighbors
- **Visualization**: Matplotlib with custom color schemes

## Output Files Generated

### Images
- Individual crop images (gravity, magnetic, radiometric, rgb)
- Reconstructed full survey images
- Analysis visualizations (t-SNE, UMAP, anomalies, similarity)

### Data Files
- Crop metadata CSV
- Feature metadata CSV
- Extracted features NPZ file
- Model checkpoints

### Analysis Results
- Anomaly detection results with scores
- Similarity search functionality
- Interactive dashboard components

## Applications & Use Cases

### Mineral Exploration
- Identify anomalous regions for further investigation
- Find similar geological patterns across survey area
- Combine multiple geophysical signatures for target identification

### Data Quality Assessment
- Detect data collection anomalies
- Identify edge effects and survey boundaries
- Validate processing pipeline results

### Pattern Recognition
- Discover spatial patterns in geophysical data
- Identify regions with similar geophysical signatures
- Map feature space to spatial coordinates

## Technical Challenges Addressed

1. **Multi-modal Data Integration**: Combining gravity, magnetic, and radiometric data
2. **Spatial Consistency**: Ensuring coordinate system alignment across datasets
3. **Scale Handling**: Processing large survey areas with appropriate resolution
4. **Edge Effects**: Accounting for survey boundaries in analysis
5. **Feature Learning**: Extracting meaningful features from geophysical images
6. **Anomaly Detection**: Identifying unusual patterns while avoiding false positives

## Future Enhancements

1. **Real-time Processing**: Streamlit app for live data analysis
2. **Advanced Models**: Transformer-based architectures for geophysical data
3. **Multi-scale Analysis**: Hierarchical processing at different resolutions
4. **Temporal Analysis**: Time-series analysis for repeated surveys
5. **Integration**: Combine with geological and geochemical data
6. **Deployment**: Web-based interface for field geologists 