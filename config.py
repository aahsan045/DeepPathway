seed = 42
import pandas as pd
import numpy as np
import random
import torch
import os
random.seed(seed)

# Set seed for NumPy
np.random.seed(seed)

# Set seed for PyTorch
torch.manual_seed(seed)

# Set seed for CUDA
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensure deterministic behavior in CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for Python hash
torch.use_deterministic_algorithms(True) #newly added line
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8" #newly added line

os.environ['PYTHONHASHSEED'] = str(seed)


device= "cuda:0"  #default device, but can be changed
debug = True
lr = 1e-3
patience = 5 # patience=5 for all other models while patience = 3 for training MLP of Deeppathway model.
epochs = 50 # 50 for all other models while 15 for training MLP model only.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet18'
image_embedding = 512
optimus_embedding = 1536 #fixed embeddigns from H-optimus0
spot_embedding = 47 #number of pathways (change for each dataset)

pretrained = True
trainable = True
temperature = 1.0
train_size = 0.8
# for projection head; used for both image and text encoders
projection_dim = 256 # the dimension used for projecting the inputs into same space
mlp_projection_dim = 128
dropout = 0.35
mlp_dropout=0.2
train_batchsize=256
val_batchsize=64
top_k=10  #Number of reference (Neighbour) points used to estimate pathway or gene expression using BLEEP. Default: 10 for pathways, 50 for genes 
method="Bleep+optimus"  # Bleep OR Bleep+optimus OR cnn+mlp or DeepPathway
# set the disease dataset variables with the training and testing sample ids
dataset="prostate"
# all_samples=["MEND154","MEND156","MEND157","MEND158","MEND159","MEND160","MEND161","MEND162"]  # all sample IDs in the dataset. User can choose their own dataset IDs.
all_samples=["MEND61","MEND62"]
test_sample="MEND61"   # One test sample Id for test
train_samples=sorted(list(set(all_samples)-set([test_sample])))
root_path="/home/e90244aa/Bleep/DeepPathwayV2/prostate cancer dataset/"
hest_metadata=pd.read_csv("HEST_v1_1_0 .csv") #download metadata CSV file from https://huggingface.co/datasets/MahmoodLab/hest/tree/main
mpp_res = [hest_metadata[hest_metadata['id']==each]['pixel_size_um_estimated'].values.astype(float)[0] for each in all_samples]
# mpp_res=[0.3415]*len(all_samples)

# second_path="bleep_pathway_expression/"

pathway_dict_file = "pathway_dict_msigdb_hallmark_complete.json"
threshold_pathways=0.70

