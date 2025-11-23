import torch
from tqdm import tqdm
import config as cfg
import numpy as np
import pandas as pd
import torch.nn.functional as F
from numpy.linalg import norm
from scipy import stats
from data_loaders import build_loaders_inference
from models import BLEEPOnly, BLEEPWithOptimus, BLEEP_MLP, DeepPathway
from spatialdata import read_zarr
import scanpy as sc
from visualization import plot_sdata
import math
import os
def get_image_embeddings_with_optimus(test_loader,model_path, model):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    print("Finished loading model")

    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].cuda()
            image_features = model.image_encoder(images)
            optim_feat = batch['st_feat'].cuda()
            image_features = torch.cat((image_features, optim_feat), dim=1)
            preds = model.image_projection(image_features)
            test_image_embeddings.append(preds)
    return torch.cat(test_image_embeddings)


def get_image_embeddings_without_optimus(test_loader,model_path, model):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    print("Finished loading model")

    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].cuda()
            image_features = model.image_encoder(images)
            preds = model.image_projection(image_features)
            test_image_embeddings.append(preds)
    return torch.cat(test_image_embeddings)


def get_predictions_BLEEP_MLP(model_path, model, test_loader): #get_image_embeddings_mlp
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    print("Finished loading model")
    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_model.image_encoder(batch["image"].cuda())
            image_embeddings=model.image_model.image_projection(image_features)
            preds=model.img_linear(image_embeddings)
            test_image_embeddings.append(preds)
    return torch.cat(test_image_embeddings)


def get_embeddings_bleep_optimus(model_path, model,test_loader):
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    test_spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_model.image_encoder(batch["image"].cuda())
            optim_feat = batch['st_feat'].cuda()
            image_features = torch.cat((image_features, optim_feat), dim=1)
            image_embeddings = model.image_model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)
            test_spot_embeddings.append(model.spot_projection(batch["reduced_expression"].cuda()))

    return torch.cat(test_image_embeddings), torch.cat(test_spot_embeddings)

def get_prediction_DeepPathway(model_path, model,test_loader):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    print("Finished loading model")

    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_model.image_encoder(batch["image"].cuda())
            optim_feat = batch['st_feat'].cuda()
            image_features = torch.cat((image_features, optim_feat), dim=1)
            image_embeddings = model.img_linear(model.image_model.image_projection(image_features))
            preds.append(image_embeddings)

    return torch.cat(preds)

def get_embeddings_bleep(model_path, model,test_loader):  # for simple Bleep model
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot')  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    test_image_embeddings = []
    test_spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            enc = batch['pos encodings'].cuda()
            image_embeddings = model.image_projection(image_features)  # +enc
            test_spot_embeddings.append(model.spot_projection(batch["reduced_expression"].cuda()))
            test_image_embeddings.append(image_embeddings)

    return torch.cat(test_image_embeddings), torch.cat(test_spot_embeddings)

def get_loader_bleep(train_samples,test_sample,root_path):
    a = train_samples
    a.insert(0, test_sample)
    test_loader = build_loaders_inference(a, root_path)
    return test_loader,a

def get_loader_mlp(test_sample,root_path):
    test_loader = build_loaders_inference(test_sample, root_path)
    return test_loader

def get_predictions_from_bleep(image_query,expression_gt,spot_key,expression_key,method='average',k=10):
    if image_query.shape[1] != 256:
        image_query = image_query.T
        print("image query shape: ", image_query.shape)
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
        print("expression_gt shape: ", expression_gt.shape)
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
        print("spot_key shape: ", spot_key.shape)
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T
        print("expression_key shape: ", expression_key.shape)

    if method == "simple":
        indices = find_matches(spot_key, image_query, top_k=k)
        matched_spot_embeddings_pred = spot_key[indices[:, 0], :]
        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        matched_spot_expression_pred = expression_key[indices[:, 0], :]
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "average":
        print("finding matches, using average of top 10 expressions")
        indices = find_matches(spot_key, image_query, top_k=k)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        for i in range(indices.shape[0]):
            # print(i)
            matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0)
            matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0)

        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "weighted_average":
        print("finding matches, using weighted average of top 50 expressions")
        indices = find_matches(spot_key, image_query, top_k=k)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        for i in range(indices.shape[0]):
            a = np.sum((spot_key[indices[i, 0], :] - image_query[i, :]) ** 2)  # the smallest MSE
            weights = np.exp(-(np.sum((spot_key[indices[i, :], :] - image_query[i, :]) ** 2, axis=1) - a + 1))
            if i == 0:
                print("weights: ", weights)
            matched_spot_embeddings_pred[i, :] = np.average(spot_key[indices[i, :], :], axis=0, weights=weights)
            matched_spot_expression_pred[i, :] = np.average(expression_key[indices[i, :], :], axis=0, weights=weights)

        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    true = expression_gt
    pred = matched_spot_expression_pred

    # print("Pred Here:",pred.shape)
    # print("True Here",true.shape)
    # print(np.max(pred))
    # print(np.max(true))
    # print(np.min(pred))
    # print(np.min(true))
    return pred

def get_predicted_expressions(a,root_path,img_embeddings_all,spot_embeddings_all,k=10):
    data = []
    datasize = []
    for each in a:
        temp = np.array(pd.read_csv(root_path+"pathway expression/"+each+"_pathway expression.csv").iloc[:,1:]).T
        datasize.append(temp.shape[1])
        data.append(temp)
    for i in range(len(a)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i + 1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        spot_embeddings = spot_embeddings_all[index_start:index_end]
        np.save(root_path + "img_embeddings_" + str(i + 1) + ".npy", image_embeddings.T)
        np.save(root_path + "spot_embeddings_" + str(i + 1) + ".npy", spot_embeddings.T)
    expression_key=np.concatenate([data[i] for i in range(1,len(data))],axis=1)
    spot_key=np.concatenate([np.load(root_path+"spot_embeddings_"+str(i+1)+".npy") for i in range(1,len(data))],axis=1)
    image_query = np.load(root_path + "img_embeddings_1.npy")
    expression_gt = data[0].T
    expression_pred = get_predictions_from_bleep(image_query,expression_gt,spot_key,expression_key,method='average',k=10)
    return expression_pred

def find_matches(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T  # 2277x2265
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()

def metrics_calculation(true,pred):
    corr_cells = np.zeros(pred.shape[0])
    sp_corr_cells = np.zeros(pred.shape[0])
    cosine_cells = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        corr_cells[i] = np.corrcoef(pred[i, :], true[i, :])[0, 1]
        sp_corr_cells[i] = stats.spearmanr(pred[i, :], true[i, :])[0]
        cosine_cells[i] = np.dot(pred[i, :], true[i, :]) / (norm(pred[i, :]) * norm(true[i, :]))

    corr_cells = corr_cells[~np.isnan(corr_cells)]
    sp_corr_cells = sp_corr_cells[~np.isnan(sp_corr_cells)]
    cosine_cells = cosine_cells[~np.isnan(cosine_cells)]

    print("Mean Pearson correlation across All spots (pathwaywuse correlation): ", np.mean(corr_cells))
    print("Mean Spearman correlation across spots: ", np.mean(sp_corr_cells))
    print("Mean Cosine Similarity across spots: ", np.mean(cosine_cells))

    corr_pathways = np.zeros(pred.shape[1])
    sp_corr_paths = np.zeros(pred.shape[1])
    cosine_pathways = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        corr_pathways[i] = np.corrcoef(pred[:, i], true[:, i], )[0, 1]
        sp_corr_paths[i] = stats.spearmanr(pred[:, i], true[:, i])[0]
        cosine_pathways[i] = np.dot(pred[:, i], true[:, i]) / (norm(pred[:, i]) * norm(true[:, i]))

    corr_pathways = corr_pathways[~np.isnan(corr_pathways)]
    sp_corr_paths = sp_corr_paths[~np.isnan(sp_corr_paths)]
    cosine_pathways = cosine_pathways[~np.isnan(cosine_pathways)]

    print("Mean Pearson correlation across pathways: ", np.mean(corr_pathways))

    print("Mean Spearman correlation across pathways: ", np.mean(sp_corr_paths))
    print("Mean Cosine Similarity across pathways: ", np.mean(cosine_pathways))
    true_filetered = []
    for k in range(0, corr_pathways.shape[0]):
        if corr_pathways[k] == np.nan:
            continue
        else:
            true_filetered.append(true[:, k])
    true_filetered = np.array(true_filetered).T
    print(true_filetered.shape)
    ind = np.argsort(np.sum(true_filetered, axis=0))[-5:]
    # print("highly expressed indices",ind)
    print("mean Pearson correlation highly expressed pathways: ", np.mean(corr_pathways[ind]))
    # print("mean Spearman correlation highly expressed pathways: ", np.mean(sp_corr_paths[ind]))
    ind = np.argsort(np.var(true_filetered, axis=0))[-5:]
    # print("highly variable indices",ind)
    print("mean Pearson correlation highly variable pathways: ", np.mean(corr_pathways[ind]))
    # print("mean Spearman correlation highly variable pathways: ", np.mean(sp_corr_paths[ind]))
    return


def copy_adata(adata_true,adata_pred):
    adata_pred.obs = adata_true.obs
    adata_pred.uns = adata_true.uns
    adata_pred.var_names = adata_true.var_names
    adata_pred.obsm = adata_true.obsm
    return adata_pred

def contains_nan(sublist):
    return any(isinstance(item, float) and math.isnan(item) for item in sublist)

def get_top_k_pathways(true,pred,pathway_names,k=5):
    res=[]
    top_pathways=[]
    for i in range(0,pred.shape[1]):
        x = np.corrcoef(true[:, i], pred[:, i])[0, 1]
        res.append([i, x])
    cleaned_data = [sublist for sublist in res if not contains_nan(sublist)]
    sorted_data = sorted(cleaned_data, key=lambda x: x[1], reverse=True)
    for i in range(0,k):
        top_pathways.append(pathway_names[sorted_data[i][0]])
    return top_pathways


def main():
    # Bleep OR Bleep+optimus OR cnn+mlp+optimus OR cnn+mlp
    # "itr_01_Bleep_liver_pathways_NCBI672.pt"
    # "itr_01_Bleep+optimus_liver_pathways_NCBI672.pt"
    # "itr_01_cnn+mlp+optimus_liver_pathways_NCBI672.pt"
    # itr_01_cnn+mlp_liver_pathways_NCBI672.pt"
    sdata = read_zarr(cfg.root_path+"SpatialData/"+cfg.test_sample+"_spatial_data.zarr")
    true=sdata['adata_pathways'].X
    if cfg.method == 'Bleep':
        test_loader,names = get_loader_bleep(cfg.train_samples,cfg.test_sample,cfg.root_path)
        model=BLEEPOnly().to('cuda:0')
        model_path = cfg.root_path  + "saved_weights/itr_01_" + cfg.method + "_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".pt"
        image_embeddings_all, spot_embeddings_all = get_embeddings_bleep(model_path, model, test_loader)
        image_embeddings_all= image_embeddings_all.cpu().numpy()
        spot_embeddings_all  = spot_embeddings_all.cpu().numpy()
        pred = get_predicted_expressions(names, cfg.root_path, image_embeddings_all, spot_embeddings_all,cfg.top_k)
        adata_preds =copy_adata(sdata['adata_pathways'],sc.AnnData(X=pred))
        method=str()
        if "+" in cfg.method:
            method = cfg.method.replace("+","_")
        else:
            method = cfg.method
        sdata.tables['predictions_'+method]=adata_preds
        sdata.write(cfg.root_path+"/SpatialData/" + cfg.test_sample + "_spatial_data.zarr",overwrite=True)
        metrics_calculation(true,pred)
        top_k_pathways = get_top_k_pathways(true,pred,sdata['adata_pathways'].var_names.tolist(),k=3)
        plot_sdata(sdata, cfg.test_sample,top_k_pathway_names=top_k_pathways)

    elif cfg.method == 'Bleep+optimus':
        test_loader,names = get_loader_bleep(cfg.train_samples,cfg.test_sample,cfg.root_path)
        model_path = cfg.root_path  + "saved_weights/itr_01_" + cfg.method + "_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".pt"
        model = BLEEPWithOptimus().to('cuda:0')
        image_embeddings_all, spot_embeddings_all = get_embeddings_bleep_optimus(model_path, model,test_loader)
        image_embeddings_all = image_embeddings_all.cpu().numpy()
        spot_embeddings_all = spot_embeddings_all.cpu().numpy()
        pred = get_predicted_expressions(names, cfg.root_path, image_embeddings_all,spot_embeddings_all,cfg.top_k)
        adata_preds = copy_adata(sdata['adata_pathways'], sc.AnnData(X=pred))
        method=str()
        if "+" in cfg.method:
            method = cfg.method.replace("+","_")
        else:
            method = cfg.method
        sdata.tables['predictions_' + method] = adata_preds
        sdata.write(cfg.root_path + "/SpatialData/" + cfg.test_sample + "_spatial_data.zarr", overwrite=True)
        metrics_calculation(true,pred)
        top_k_pathways = get_top_k_pathways(true, pred, sdata['adata_pathways'].var_names.tolist(), k=3)
        plot_sdata(sdata, cfg.test_sample,top_k_pathway_names=top_k_pathways)
    elif cfg.method == 'DeepPathway':
        test_loader = build_loaders_inference([cfg.test_sample], cfg.root_path)
        model_path=cfg.root_path+"saved_weights/itr_01_"+cfg.method+"_"+cfg.dataset+"_pathways_"+cfg.test_sample+".pt"
        model=DeepPathway().to('cuda:0')
        pred = get_prediction_DeepPathway(model_path, model,test_loader).cpu().numpy()
        adata_preds = copy_adata(sdata['adata_pathways'], sc.AnnData(X=pred))
        method = str()
        if "+" in cfg.method:
            method = cfg.method.replace("+", "_")
        else:
            method = cfg.method
        sdata.tables['predictions_' + method] = adata_preds
        sdata.write(cfg.root_path + "/SpatialData/" + cfg.test_sample + "_spatial_data.zarr", overwrite=True)
        # np.save(cfg.root_path + "prediction_" + cfg.method + "_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".npy",pred)
        metrics_calculation(true,pred)
        top_k_pathways = get_top_k_pathways(true, pred, sdata['adata_pathways'].var_names.tolist(), k=3)
        plot_sdata(sdata, cfg.test_sample, top_k_pathway_names=top_k_pathways)

    else: # case for BLEEP with MLP (DNN)
        test_loader = test_loader = build_loaders_inference([cfg.test_sample], cfg.root_path)
        model_path = cfg.root_path + "saved_weights/" + "itr_01_" + cfg.method + "_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".pt"
        model = BLEEP_MLP().to('cuda:0')
        pred = get_predictions_BLEEP_MLP(model_path, model, test_loader).cpu().numpy()
        adata_preds = copy_adata(sdata['adata_pathways'], sc.AnnData(X=pred))
        method = str()
        if "+" in cfg.method:
            method = cfg.method.replace("+", "_")
        else:
            method = cfg.method
        sdata.tables['predictions_' + method] = adata_preds
        sdata.write(cfg.root_path + "/SpatialData/" + cfg.test_sample + "_spatial_data.zarr", overwrite=True)
        metrics_calculation(true,pred)
        top_k_pathways = get_top_k_pathways(true, pred, sdata['adata_pathways'].var_names.tolist(), k=3)
        plot_sdata(sdata, cfg.test_sample, top_k_pathway_names=top_k_pathways)
        # np.save(cfg.root_path + "prediction_" + cfg.method + "_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".npy",pred)
    return
if __name__ == "__main__":
    main()
