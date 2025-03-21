# Project flow v0.0.1



## DATA

1. Host KoBold Data in an Isolated S3 Bucket to simulate a client's real data
2. Pull client data into Project S3 Bucket
3. Pre-process/feature engineering pipeline
    3a. ViT prep: Interpolate point data to create grids for each geophysical parameter, saving as images
    3b. Isolation Forest prep: can just take in normalized tabular data, maybe some feature engineering
    3c. GPX_Pred: Take something like 20%

## TRAIN

4. Train w/ SageMaker
    * 3 seperate models, ViT/CV is really the computationally intensive one
    * NOTE: GPX data’s spatial nature requires handling autocorrelation [possibly using spatial cross-validation or monte carlo] to ensure model robustness
    4a. ViT training: pre-trained [?] ViT to extract features from image data, then apply Isolation Forest on features for anomaly detection
    4b. Isolation Forest training: PyTorch, very simple/lightweight
    4c. GPX_Pred training: regression model (e.g., Random Forest) on 20% training set to predict 80% testing set

## STREAMLIT APP

5. Pass model results to Streamlit App hosted by EC2
    * Reads preds from S3 and visualizes using Folium
    * Each model will have its own page to feature/visualize/analyze results

#### Local Dev
* Data handling + model training: can do that in colab to choose model performance then write preprocessing script again here
* Visualizing: can run streamlit locally, test UI


#### AWS Deployment
* Data handling: S3 for storage, transfer scripts
* Model training: SageMaker for training
* Visualization: Host streamlit w/ EC2


#### Additional considerations not to forget about
* Docker, Poetry, Linting, TDD
* Aux datasets done from other public surveys



#### Data concern
* Data is sparse and Zambia gov is ghosting me on OS data so may have to build with only KoBold survey
* Maybe a new pipeline would be simply ViT trained on GPX data, feature extraction, use new stack of images for anomaly detection


