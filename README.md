# DeepPathway: Predicting Pathway Expression from Histopathology Images
# URL: 
      https://www.biorxiv.org/content/10.1101/2025.07.21.665956v1.abstract
This repository contains implementation code for DeepPathway.
# Overview:
- DeepPathway is a bimodal contrastive learning framework that is trained on Spatial Transcriptomics (ST) datasets to predict pathway expression from H&E images.
-  Ucell is used to compute pathway expression for the MSigDB hallmark pathway definitions.
-  Once model is trained, unlike traditional contrasstive learning methods, DeepPathway can be used to directly predict pathway expression of test H&E image without requiring training data (or embeddings).
  
  <img width="944" height="298" alt="image" src="https://github.com/user-attachments/assets/867072a2-4f45-497c-a0ac-608c809f9729" />

  # Pre-requisites:
  Create a conda envoirnment having Python >=3.10.
  - python: 3.10.14
  - torch: 2.7.0+cu118
  - timm: 1.0.12
  - spatialdata: 0.4.0
  - scanpy: 1.11.5
  - numpy: 1.26.4
  - pandas: 2.2.3
  - scikit-learn:: 1.5.2
  - pillow: 11.0.0
  - scipy: 1.12.0
  - opencv: 4.12.0
  - skimage: 0.24.0
  - tiatoolbox: 1.6.0
  - openslide: 1.4.1
  - matplotlib: 3.9.3
# Data Processing
1. Download ST data from HEST-1k (https://huggingface.co/datasets/MahmoodLab/hest). Aslo download Metadata from HEST1-k to obtain H&E image resolutions (in MPP) or use OPenslide to get resolution values.
2. Obtain Login key from H-OPtimus (https://huggingface.co/bioptimus).
3. Save WSIs and ST data (.h5ad) in ./WSIs and ./st folders in the root_path='YOUR ROOT DIR'.
4. Open and modify config.py and put your SAMPLE_IDs in "all_samples" list.
5. Provide your Pathway Definition file. We obtained MsigDB hallmark pathway definitions from here: https://maayanlab.cloud/Enrichr/#libraries.
6. Add MPP resolution of each WSI in config.py or extract from metadata (from HEST-1k in our case.) 
7. Restart Kernel after saving the config.py file. Run "python data_processing.py" OR see Tutorials/data_processing.ipynb for data processing
8. "IMPORTANT": Run Ucell Calculations before creating SpatialData objects (which will be used in model training and Validation.). Use Ucell_code.R file with your configurations to store Spot X pathway matrix of each sample. Use R_max threshold as obtained for pathway expression quantification (which will be calculated during running data processing module.) 

# Training and Validation
1. Set the test and train sample_ids. In default settings, setting up test sample id will create a list of training set ids. 
2. Set the parameter "spot_embedding" as per your number of pathways or number of genes (No. of outputs).
3. Choose your model, e.g., BLEEPOnly, BLEEPWithOptimus, or DeepPathway.
4. Restart Kernel after saving the config.py and run "train.py" file.
5. For predictions, use the test sample id with saved model weights. An example of obtaining predictions is provided in Tutorials/training.ipynb.
6. Optionally, you can use scanpy for Spatial visualization. Current code supports integration of SpatialData for predictions. 
