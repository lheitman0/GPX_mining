# Doc to work through idea and plan


### Data

KoBold OS GPX data from zambia
    * Geophysical data (magnetic, gravity, and radiometric) from a survey 60 km northwest of Kitwe, Zambia
        * Gravity data with columns X, Y, Digital_terrain, and Bougue230_Processed
        * Magnetic data with columns X, Y, and Processed_magnetics
        * Radiometric data with columns X, Y, and Potassium_NASVD_processed
    note: potential spatial autocorrelation, may require techniques like spatial interpolation. Ensure data quality checks align with the report's processing standards

### Architecture

DATA
    * Ingest, normalize, transform w/ functional pipeline


Streamlit app, SageMaker for cloud training


Possible stack:

1. Data Ingestion: Amazon S3
2. Preprocess data using standard python/pandas
3. Isolation forest anomaly detection?
4. Streamlit app hosted on EC2

Cloud services involved: S3 (storage), SageMaker (training), EC2 (hosting app)


Final stack idea for initial PoC:
    Run a Streamlit application on an EC2 instance that reads processed data from S3, queries the SageMaker endpoints for anomaly scores, and displays results on an interactive map
        * Folium for mapping ? show composite anomaly scores, with filters for user interaction