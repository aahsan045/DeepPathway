import pandas as pd

seed = 42
device= "cuda:0"  #default device, but can be changed
debug = True
# batch_size = 256
lr = 1e-3
# Set seed for Python modules
patience = 5 # patience=5 for all other models while patience = 3 for MLP model. Change it here for training MLP to 3.
epochs = 50 # 50 for all other models while 15 for training MLP model only.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet18'
image_embedding = 512
optimus_embedding = 1536 #fixed embeddigns from H-optimus0
spot_embedding = 50 #number of pathways (change for each dataset)

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
method="Bleep"  # Bleep OR Bleep+optimus OR cnn+mlp+optimus OR cnn+mlp

# set the disease dataset variables with the training and testing sample ids
dataset="prostate"
all_samples=["INT25","INT26","INT27","INT28","INT35"]  # all sample IDs in the dataset. User can choose their own dataset IDs.
test_sample="INT28"   # One test sample Id for test
train_samples=sorted(list(set(all_samples)-set([test_sample])))
root_path="small prostate cancer dataset/"
pixel_res=pd.read_excel("pixel_res.xlsx")
mpp_res = [float(each) for each in list(pixel_res[pixel_res['dataset']=='prostate small']['pixel_res'].values)[0].split(",")] #dataset can be liver OR prostate_small OR prostate_large. Get the mpp values from HEST-1K metadata.

# second_path="bleep_pathway_expression/"

pathway_dict_file = "MSigDB_Hallmark_2020.txt"
threshold_pathways=0.70

#"itr_01_Bleep_liver_pathways_NCBI672.pt"
#"itr_01_Bleep+optimus_liver_pathways_NCBI672.pt"
#"itr_01_cnn+mlp+optimus_liver_pathways_NCBI672.pt"
#itr_01_cnn+mlp_liver_pathways_NCBI672.pt"

