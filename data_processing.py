import scanpy as sc
import numpy as np
import pandas as pd
import openslide
import scipy
import cv2
from PIL import Image
from tiatoolbox import data, logger
import json
from tiatoolbox.wsicore.wsireader import VirtualWSIReader
from tiatoolbox.tools import patchextraction
from huggingface_hub import login
from spatialdata import SpatialData
from spatialdata.models import TableModel, Image2DModel
import torch
import timm
from torchvision import transforms
from config import *
import os
from pathlib import Path
import config as cfg
def get_adata(paths,uni_pathway_genes):
    hvg_bools=[]
    for path in paths:
        adata = sc.read_h5ad(path)
        # adata = adata[adata.obs['in_tissue']==1]
        sc.pp.filter_cells(adata, min_genes=500)
        sc.pp.filter_genes(adata,min_cells=10)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=10000) #for pathways n_top_genes = 10,000
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)
    hvg_union = hvg_bools[0]
    for i in range(1, len(paths)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]
    unique_genes= list(hvg_union[hvg_union==True].index.values)
    print("No. of unique genes after filtering",len(unique_genes))
    unique_common_genes = list(set(unique_genes).intersection(set(uni_pathway_genes)))
    print("No. of Common genes between unique pathway genes and unique filtered genes from all samples",len(unique_common_genes))
    return unique_common_genes
def save_adatas(paths,unique_common_genes):
    filtered_exp_mtxs = []
    barcodes_updated=[]
    for path in paths:
        adata = sc.read_h5ad(path)
        # adata = adata[adata.obs['in_tissue']==1]
        sc.pp.filter_cells(adata, min_genes=500)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        indices = [i for i, gene in enumerate(adata.var_names.values) if gene in unique_common_genes]
        filtered_exp_mtxs.append(adata[:,indices].X)
        barcodes_updated.append(adata.obs_names)
        print(adata[:, indices].X.shape)
    return filtered_exp_mtxs,barcodes_updated

def check(image,threshold=0.75):
    white=200
    white_pixels = np.sum(np.all(image>=white,axis=-1))
    total = image.shape[0]*image.shape[1]
    percent = white_pixels/total
    return percent>threshold

def check_cropping(samples,image,name,spatial_pos_csv,barcode_csv,res,root_path):
    cropped_images=[]
    white_back_images=[]
    img_filtered_barcodes=[]
    img_file_paths=[]
    parent_dir = root_path + "H&E patches/"  # Change this to your target directory
    # # Create folders
    for folder in samples:
        folder_path = os.path.join(parent_dir, folder)
        os.makedirs(folder_path, exist_ok=True)  # exist_ok=True prevents errors if the folder already exists
    wsi_reader = VirtualWSIReader(image,mpp=(res,res))
    for each, i in zip(barcode_csv,range(len(barcode_csv))):
        v1 =  spatial_pos_csv.loc[spatial_pos_csv[0] == each,4].values[0] #pixel_row_in_full_reS
        v2 =  spatial_pos_csv.loc[spatial_pos_csv[0] == each,5].values[0]  # pixel_col_in_full_res
        patch_extractor = patchextraction.PointsPatchExtractor(input_img=wsi_reader,locations_list=np.array([(v2,v1)]),
                                                               resolution=res,patch_size=(224,224), units='mpp')
        for patch in patch_extractor:
            cropped_image = patch
        if check(cropped_image,0.75):
            white_back_images.append(cropped_image)
        else:
            cropped_images.append(cropped_image)
            img_filtered_barcodes.append(each)
            patch_image = Image.fromarray(cropped_image)
            patch_image.save(parent_dir+name+"/"+each+".png") # saving the spot image into Sample folder.
            img_file_paths.append(parent_dir+name+"/"+each+".png")
    print("More than 75% white background patches",len(white_back_images),"True patches", len(cropped_images))
    return img_filtered_barcodes,img_file_paths

def image_processing(root_path, samples,pixel_res,paths):
    img_fil_paths=[]
    img_fil_barcodes=[]
    for each, res, path in zip(samples, pixel_res,paths):
        print("SAMPLE Name: ", each)
        image_path = root_path + "wsis/" + each + ".tif"
        slide=openslide.OpenSlide(image_path)
        level_dims = slide.level_dimensions[0]
        image = np.array(slide.read_region((0, 0), 0, level_dims).convert("RGB"))
        print("Original Image Shape: ", image.shape)
        print("Pixel Resolution", res)
        spatial_pos_path = root_path +"tissue_position_lists/"+ each + "_tissue_position_list.csv" # if does not exist, please save the sample_name_tissue_position_list.csv into root data directory
        if not Path(spatial_pos_path).exists():
            Path(spatial_pos_path).parent.mkdir(parents=True, exist_ok=True)
            adata = sc.read_h5ad(path)
            temp = adata.obs.iloc[:, 0:5]
            new_order = ['in_tissue', 'array_row', 'array_col','pxl_row_in_fullres','pxl_col_in_fullres']
            temp=temp[new_order]
            temp.to_csv(root_path +"tissue_position_lists/" + each + "_tissue_position_list.csv", header=None,sep=",")
        else:
            pass
        barcode_path = root_path + "filtered_barcodes/"+ each + "_sample_filtered_barcodes.tsv"
        barcode_csv = pd.read_csv(barcode_path, sep="\t", header=None)[0].values
        spatial_pos_csv = pd.read_csv(spatial_pos_path, header=None, sep=",")
        spatial_pos_csv = spatial_pos_csv[spatial_pos_csv[1] == 1]
        img_filtered_barcodes, img_file_paths = check_cropping(samples,image, each, spatial_pos_csv, barcode_csv, res,root_path)
        img_fil_barcodes.append(img_filtered_barcodes)
        img_fil_paths.append(img_file_paths)
    return img_fil_paths,img_fil_barcodes

def filtering_pathway(pathway_dict,unique_common_genes,threshold_pathways=0.70):
    print("Starting Pathway Filtering with 70% threshold")
    filtered_pathway_dic = dict()
    for keys, value in zip(pathway_dict.keys(), pathway_dict.values()):
        intersection = list(set(unique_common_genes).intersection(value))
        if len(intersection) / len(value) >= threshold_pathways and len(intersection) >= 20:
            filtered_pathway_dic[keys] = intersection
    return filtered_pathway_dic

def get_optimus_features(device,root_path,samples):
    # login()  # perform login with your own huggingface key that can be obtained for H-OPtimus Huggingface. Skip login if key is already added.
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            for name in samples:
                feature_vectors = []
                count = 0
                barcodes = pd.read_csv(root_path + "filtered_barcodes/" + name + "_sample_filtered_barcodes_1.tsv", sep="\t", header=None)[0].values
                for bar in barcodes:
                    image = cv2.imread(root_path + "H&E patches/" + name + "/" + bar + ".png")
                    image = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                    features = model(image)
                    feature_vectors.append(features.cpu().numpy())
                    count += 1
                feature_vectors = np.array(feature_vectors)
                feature_vectors = np.squeeze(feature_vectors, axis=1)
                np.save(root_path + "optimus_features/" + name + "_image_features_optimus-h.npy", feature_vectors)
                print(feature_vectors.shape, "Done")
    return

def save_filtered_barcodes_gene(root_path,paths,barcodes,samples):
    for i in range(0,len(paths)):
        temp=pd.DataFrame(list(barcodes[i]))
        temp.to_csv(root_path+"filtered_barcodes/"+samples[i]+"_sample_filtered_barcodes.tsv",sep="\t",index=False,header=None)
    return
def save_filtered_barcodes_image(root_path,img_fil_barcodes,img_fil_paths,samples):
    print("Final Sample-wise Filtered data shape......")
    for bars, fp, i in zip(img_fil_barcodes, img_fil_paths, samples): # saving the spot-level filtered images and their file paths.
        temp = pd.DataFrame(bars)
        print(i, temp.shape, len(fp))
        temp.to_csv(root_path + "filtered_barcodes/" + i + "_sample_filtered_barcodes_1.tsv", sep="\t", index=False, header=None)
        np.save(root_path + "H&E patches/" + i + "_sample_file_paths.npy", fp)
    return
def save_gene_exp_data(root_path,filtered_exp_mtxs,samples):
    for each, i in zip(filtered_exp_mtxs, range(len(samples))):
        aa = pd.read_csv(root_path + "filtered_barcodes/" + samples[i] + "_sample_filtered_barcodes_1.tsv", sep="\t", header=None)[0].values
        bb = pd.read_csv(root_path + "filtered_barcodes/" + samples[i] + "_sample_filtered_barcodes.tsv", sep="\t", header=None)[0].values
        common_indices = np.intersect1d(aa, bb)
        mask = np.isin(bb, common_indices)
        temp = np.array(each.toarray())[mask]
        np.save(root_path + "filtered_gene_expression/" + samples[i] + "_sample_unique_genes_pathways_samples.npy", temp)
    return
def create_objects_for_ucell(root_path,samples):
    for each in samples:
        cell_names = pd.read_csv(root_path + "filtered_barcodes/" + each + "_sample_filtered_barcodes_1.tsv",sep="\t", header=None)[0].values
        sample = np.load(root_path + "filtered_gene_expression/" + each + "_sample_unique_genes_pathways_samples.npy").T  # original or predicted sample.
        gene_names = np.load(root_path+cfg.dataset+'_unique_common_genes.npy',allow_pickle=True)
        scipy.io.savemat(root_path + "data for Ucell calculations/"+ each + "_data.mat",{'x': sample, 'gene_names': gene_names, 'cell_names': cell_names}) # this data will be used in UCell.R code to get the pathway expression matrix of Spots x pathways
    return
    
def get_adata_object(tissue_position_list_data,scalefactor_file_path,data_array,hi_res_image_data,g_names,barcode_names):
    spatial = tissue_position_list_data.loc[tissue_position_list_data[1] == 1, [4, 5]].values
    reversed_array = np.array([[sub_array[1], sub_array[0]] for sub_array in spatial])
    scalefactor_data = scalefactor_file_path
    adata = sc.AnnData(X=data_array)
    if 'spatial' not in adata.uns:
        adata.uns['spatial'] = {}
    if 'C1' not in adata.uns['spatial']:
        adata.uns['spatial']['ST'] = {}

    if 'images' not in adata.uns['spatial']:
        adata.uns['spatial']['ST']['images'] = {}

    if 'scalefactors' not in adata.uns:
        adata.uns['spatial']['ST']['scalefactors'] = {}
    adata.var_names = g_names
    adata.uns['spatial']['ST']['images']['downscaled_fullres'] = hi_res_image_data
    adata.obsm['spatial']= np.array(reversed_array)
    adata.uns['spatial']['ST']['scalefactors']=scalefactor_data
    new_df=pd.DataFrame()
    new_df.index = tissue_position_list_data[0].values
    new_df['in_tissue']=tissue_position_list_data[1].values
    new_df['array_row']=tissue_position_list_data[2].values
    new_df['array_col']=tissue_position_list_data[3].values
    new_df['pxl_row_in_fullres'] = tissue_position_list_data[4].values
    new_df['pxl_col_in_fullres'] = tissue_position_list_data[5].values
    adata.obs=new_df
    return adata

def Sdata_creation(root_path,samples,pathway_dict,dataset='prostate'):
    pathway_names = list(pathway_dict.keys())
    for each in samples:
        adata = sc.read_h5ad(root_path+"st/"+each+".h5ad")
        data_array= np.array(pd.read_csv(root_path+"pathway expression/"+each+"_pathway expression.csv").iloc[:,1:])
        tissue_position_list_data = pd.read_csv(root_path+"tissue_position_list/"+each+"_tissue_positions_list.csv",sep=",",header=None)
        tissue_position_list_data = tissue_position_list_data[tissue_position_list_data[1]==1]
        barcode_data = pd.read_csv(root_path+"filtered_barcodes/"+each+"_sample_filtered_barcodes_1.tsv",sep="\t",header=None)
        res=[]
        for bar in barcode_data[0]:
            temp = list(tissue_position_list_data[tissue_position_list_data[0]==bar].values[0])
            res.append(temp)
        res=pd.DataFrame(res)
        tissue_position_list_data = res
        scalefactor_file_path=adata.uns['spatial']['ST']['scalefactors']
        hi_res_image_data = adata.uns['spatial']['ST']['images']['downscaled_fullres']
        adata_pathway =get_adata_object(tissue_position_list_data,scalefactor_file_path,data_array,hi_res_image_data,pathway_names,barcode_data[0].values)
        optim_feat=np.load(cfg.root_path+"optimus features/"+each+"_image_features_optimus-h.npy")
        gene_exp = np.load(root_path+"/filtered_gene_expression/"+each+"_sample_unique_genes_pathways_samples.npy")
        adata_optim = sc.AnnData(X=optim_feat)
        adata_optim.obs = adata_pathway.obs
        adata_optim.uns=adata_pathway.uns
        adata_pathway.var_names = pathway_names
        adata_optim.obsm = adata_pathway.obsm
        adata_genes = sc.AnnData(X=gene_exp)
        adata_genes.obs=adata_pathway.obs
        adata_genes.uns=adata_pathway.uns
        adata_genes.obsm= adata_pathway.obsm
        adata_genes.var_names=np.load(root_path+dataset+"_unique_common_genes.npy",allow_pickle=True)
    #     adata_for_sdata = TableModel.parse(adata)
    #     adata_for_sdata=from_legacy_anndata(adata)
        sdata=SpatialData(tables={"adata_pathways":adata_pathway},)
        sdata.tables['optim_feat']=adata_optim
        sdata.tables['filtered_gene_exp']=adata_genes
        sdata.write(root_path+"/SpatialData/"+each+"_spatial_data.zarr",overwrite=True)
        return
def calculate_Rmax_threshold(root_path,samples):
    res=[]
    for sample in samples:
        temp=np.load(root_path+"filtered_gene_expression/"+sample+"_sample_unique_genes_pathways_samples.npy")
        sorted_arr = np.sort(temp, axis=1)[:, ::-1]
        zero_indices = np.argmax(sorted_arr == 0, axis=1)
        res.append(np.median(zero_indices))
    print("Rmax Threshold for this data with median is: ",np.median(res))  # Array of indices where the first 0 appears in each row
    return np.median(res)
def main():
    print("Starting gene processing.....")
    print()
    root_path = cfg.root_path
    with open(cfg.pathway_dict_file, 'r') as f:
        pathway_dict = json.load(f) # replace with required pathway database, i.e., GO/KEGG/MsigDB
    uni_pathway_genes=[]
    for value in pathway_dict.values():
        for each in value:
            uni_pathway_genes.append(each)
    print("Total Number of unique genes in all pathways: ",len(set(uni_pathway_genes)))
    paths=[]
    samples=list()
    filtered_exp_mtxs=list()
    barcodes=list()
    samples = cfg.all_samples
    for each in samples:
        paths.append(root_path+"st/"+each+".h5ad")
    uni_common_genes= get_adata(paths,uni_pathway_genes)
    filtered_exp_mtxs, barcodes = save_adatas(paths,uni_common_genes)
    print("Saving the Filtered Barcodes after spot filtering")
    save_filtered_barcodes_gene(root_path, paths, barcodes, samples)
    print("Starting Image Processing......")
    pixel_res = cfg.mpp_res  # Image pixel res are obtained from the Metadata file, provided by HEST-1K database
    img_fil_paths, img_fil_barcodes = image_processing(root_path, samples, pixel_res, paths)
    save_filtered_barcodes_image(root_path, img_fil_barcodes, img_fil_paths, samples)
    print("Saving the pathway-associated unique common genes")
    save_gene_exp_data(root_path, filtered_exp_mtxs, samples)
    print("Saving Unique Common Genes:, ", len(uni_common_genes))
    np.save(root_path + cfg.dataset+'_unique_common_genes.npy', uni_common_genes)
    filtered_pathway_dic = filtering_pathway(pathway_dict, uni_common_genes, threshold_pathways=cfg.threshold_pathways) #more than or equal to 70% pathway genes should be in data to calculate pathway expr. .
    print(f"Saving the filtered pathway dict having number of pathways:{len(filtered_pathway_dic.keys())} in JSON")
    file_path = root_path + 'pathway_dic_' + str(len(filtered_pathway_dic)) + '.json'
    with open(file_path, 'w') as json_file:
        json.dump(filtered_pathway_dic, json_file, indent=4)
    create_objects_for_ucell(root_path, samples)
    rmax = calculate_Rmax_threshold(root_path,samples)
    print("*********************************************************************************")
    print()
    print("Please use the following data in UCell.R code to compute Ucell Scores:")
    print("Filtered Pathway Dictionary file:",file_path)
    print("Samples List:",all_samples)
    print("Data for UCell with path:","/data for Ucell calculations/SAMPLE_ID_data.mat")
    print("Rmax=",rmax)
    print()
    print("*********************************************************************************")
    print("Starting H-Optimus-0 Feature Extraction Process.....")
    get_optimus_features(device, root_path, samples)
    print("Saving processed data as SpatialData Objects. Please perform this step once you have calculated Ucell scores.")
    Sdata_creation(root_path,samples,filtered_pathway_dic,dataset=cfg.dataset)
    return
if __name__ == "__main__":
    main()
