# DeepPAthway: Predicting Pathway Expression from Histopathology Images#
This repository contains implementation code for DeepPathway.
# Overview:
- DeepPathway is a bimodal contrastive learning framework that is trained on Spatial Transcriptomics (ST) datasets to predict pathway expression from H&E images.
-  Ucell is used to compute pathway expression for the MSigDB hallmark pathway definitions.
-  Once model is trained, unlike traditional contrasstive learning methods, DeepPathway can be used to directly predict pathway expression of test H&E image without requiring training data (or embeddings).
  
  <img width="944" height="298" alt="image" src="https://github.com/user-attachments/assets/867072a2-4f45-497c-a0ac-608c809f9729" />

  # Pre-requisites:
  - python: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
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

