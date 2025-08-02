About Dataset
Total magnetic intensity (TMI) data measures variations in the intensity of the Earth's magnetic field caused by the contrasting magnetic responses of rock-forming minerals in the Earth crust. Gravity anomaly data measures variations in the gravitational attraction caused by mass variations within the Earth. Radiometric data measures gamma ray decay of naturally-occurring materials on the earth’s surface. The data are processed via standard methods to ensure the response recorded is that due only to the rocks in the ground.

Useful primers on these types of data and surveys:

magnetics
gravity
radiometrics
This helicopter-based survey was acquired by New Resolution Geophysics for KoBold Metals in Zambia between the 31st of August and the 15th of October 2021, and consisted of 23,829 line-kilometres of data at 50-100m line spacing and 35m terrain clearance.


===== GRAVITY DATASET COLUMNS =====
['FID', 'Temperature', 'Lat', 'Long', 'Gps_Seconds', 'X', 'Y', 'SRTM', 'Baro_alt', 'Radar_alt', 'Bouguer230_Processed', 'Bouguer267_Processed', 'Digital_terrain', 'Gravity_disturbance_Raw', 'Gravity_Disturbance_Levelled', 'Gravity_disturbance_Processed', 'Processed_magnetics', 'GPS_Height', 'Gravity_Model', 'Grav_model_Processed']
Total features: 20


===== MAGNETIC DATASET COLUMNS =====
['X', 'Y', 'Flux_X', 'Flux_Y', 'Flux_Z', 'Flux_TF', 'Diurnal', 'Lat', 'Long', 'FID', 'Baro_alt', 'Radar_alt', 'Digital_terrain', 'Mag1_uncomp', 'Mag2_uncomp', 'Mag1_compensated', 'Mag2_compensated', 'GPS_Height', 'Gradient_levelled', 'UTC_time', 'Processed_magnetics']
Total features: 21


===== RADIOMETRIC DATASET COLUMNS =====
['X', 'Y', 'FID', 'Lat', 'Long', 'COSMIC', 'Humidity', 'Baro_alt', 'Radar_alt', 'Digital_terrain', 'GPS_Height', 'Potassium_Raw', 'Uranium_Raw', 'Thorium_Raw', 'Total_Count_Raw', 'Potassium_NASVD_processed', 'Uranium_NASVD_processed', 'Thorium_NASVD_processed', 'Total_count_NASVD_processed', 'Temperature', 'UTC_Time', 'Live_time']
Total features: 22




# Kobold Metals Sitatunga Airborne Geophysics Survey Channels

## 1. Gravity Dataset
The gravity dataset contains 20 features:

### Location Data
- FID
- Lat
- Long
- X
- Y

### Altitude Measurements
- SRTM
- Baro_alt
- Radar_alt
- GPS_Height
- Digital_terrain

### Gravity Measurements
- Bouguer230_Processed
- Bouguer267_Processed
- Gravity_disturbance_Raw
- Gravity_Disturbance_Levelled
- Gravity_disturbance_Processed
- Gravity_Model
- Grav_model_Processed

### Auxiliary Data
- Temperature
- Gps_Seconds
- Processed_magnetics

---

## 2. Magnetic Dataset
The magnetic dataset contains 21 features:

### Location Data
- X
- Y
- Lat
- Long
- FID

### Altitude Measurements
- Baro_alt
- Radar_alt
- GPS_Height
- Digital_terrain

### Magnetic Measurements
- Flux_X
- Flux_Y
- Flux_Z
- Flux_TF
- Mag1_uncomp
- Mag2_uncomp
- Mag1_compensated
- Mag2_compensated
- Gradient_levelled
- Processed_magnetics

### Corrections
- Diurnal

### Auxiliary Data
- UTC_time



---


## 3. Radiometric Dataset
The radiometric dataset contains 22 features:

### Location Data
- X
- Y
- Lat
- Long
- FID

### Altitude Measurements
- Baro_alt
- Radar_alt
- GPS_Height
- Digital_terrain

### Radiometric Measurements
- Potassium_Raw
- Uranium_Raw
- Thorium_Raw
- Total_Count_Raw
- Potassium_NASVD_processed
- Uranium_NASVD_processed
- Thorium_NASVD_processed
- Total_count_NASVD_processed

### Auxiliary Data
- COSMIC
- Humidity
- Temperature
- UTC_Time
- Live_time


# Channels Used for Geophysical Data Visualization

## Channels Used for Visualization

### Spatial Coordinates (All Datasets)
- **X, Y** (UTM coordinates in meters)

### Gravity Dataset
- **Digital_terrain** (for topography visualization)
- **Bouguer230_Processed** (terrain-corrected Bouguer gravity anomaly)

### Magnetic Dataset
- **Processed_magnetics** (final processed total magnetic intensity)

### Radiometric Dataset
- **Potassium_NASVD_processed** (processed potassium concentration)
- **Uranium_NASVD_processed** (typically used, though not shown in example code)
- **Thorium_NASVD_processed** (typically used, though not shown in example code)
- **Total_count_NASVD_processed** (typically used, though not shown in example code)

## Unused Channels

### Raw Measurements
- Gravity_disturbance_Raw
- Mag1_uncomp, Mag2_uncomp
- Potassium_Raw, Uranium_Raw, Thorium_Raw, Total_Count_Raw

### Flight Parameters
- Baro_alt, Radar_alt, GPS_Height
- Humidity, Temperature
- UTC_time, Gps_Seconds, Live_time

### Intermediate Processing Values
- Flux_X, Flux_Y, Flux_Z
- Diurnal corrections
- Multiple gravity models
- Gravity_disturbance_Levelled

### Location Data (Redundant)
- Lat, Long (when using X, Y UTM coordinates)
- FID (fiducial numbers)

## Notes on Visualization Approach
- Standard visualization uses spatial coordinates (X, Y) and final processed values
- Pre-interpolated GeoTIFF files are often preferred over plotting raw point data
- For specialized analysis, some of the unused channels may become relevant
- RGB composite visualizations often combine three radiometric channels (K-Th-U)