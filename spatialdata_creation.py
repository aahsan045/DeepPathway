# %%capture
import config as cfg
import scanpy as sc
import pandas as pd
import numpy as np


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
def Sdata_creatio():
    for each in cfg.all_samples:
        adata = sc.read_h5ad(cfg.root_path+"st/INT25.h5ad")
        data_array= np.load(cfg.root_path+"/pathway expression/"+each+"_50_pathways_msig_prostate_Ucell.npy")
        tissue_position_list_data = pd.read_csv(cfg.root_path+"tissue_position_list/"+each+"_tissue_position_list.csv",sep=",",header=None)
        tissue_position_list_data = tissue_position_list_data[tissue_position_list_data[1]==1]
        barcode_data = pd.read_csv(cfg.root_path+"filtered_barcodes/"+each+"_sample_filtered_barcodes_1.tsv",sep="\t",header=None)
        res=[]
        for bar in barcode_data[0]:
            temp = list(tissue_position_list_data[tissue_position_list_data[0]==bar].values[0])
            res.append(temp)
        res=pd.DataFrame(res)
        tissue_position_list_data = res
        scalefactor_file_path=adata.uns['spatial']['ST']['scalefactors']
        hi_res_image_data = adata.uns['spatial']['ST']['images']['downscaled_fullres']
        adata_pathway =get_adata_object(tissue_position_list_data,scalefactor_file_path,data_array,hi_res_image_data,g_names,barcode_data[0].values)
        optim_feat=np.load(cfg.root_path+"optimus features/"+each+"_image_features_optimus-h.npy")
        gene_exp = np.load("/home/e90244aa/Bleep/Hist2Pathway/small prostate cancer dataset/filtered_gene_expression/"+each+"_sample_unique_genes_pathways_samples.npy")
        adata_optim = sc.AnnData(X=optim_feat)
        adata_optim.obs = adata_pathway.obs
        adata_optim.uns=adata_pathway.uns
        adata_pathway.var_names = pathway_names
        adata_optim.obsm = adata_pathway.obsm
        adata_genes = sc.AnnData(X=gene_exp)
        adata_genes.obs=adata_pathway.obs
        adata_genes.uns=adata_pathway.uns
        adata_genes.obsm= adata_pathway.obsm
        adata_genes.var_names=np.load("/home/e90244aa/Bleep/Hist2Pathway/small prostate cancer dataset/prostate_unique_common_genes.npy",allow_pickle=True)
    #     adata_for_sdata = TableModel.parse(adata)
    #     adata_for_sdata=from_legacy_anndata(adata)
        sdata=SpatialData(tables={"adata_pathways":adata_pathway},)
        sdata.tables['optim_feat']=adata_optim
        sdata.tables['filtered_gene_exp']=adata_genes
        sdata.write("/home/e90244aa/Bleep/Hist2Pathway/small prostate cancer dataset/SpatialData/"+each+"_spatial_data.zarr",overwrite=True)
        return