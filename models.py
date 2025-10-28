import torch
from tqdm import tqdm
import config as cfg
import numpy as np
import torch.nn.functional as F
from torch import nn
import random
import os
random.seed(cfg.seed)

# Set seed for NumPy
np.random.seed(cfg.seed)

# Set seed for PyTorch
torch.manual_seed(cfg.seed)

# Set seed for CUDA
torch.cuda.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
os.environ['PYTHONHASHSEED'] = str(cfg.seed)

# Ensure deterministic behavior in CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils import *
from modules import *


class BLEEPOnly(nn.Module):
    def __init__(
        self,
        temperature=1.0,
        image_embedding=cfg.image_embedding,
        spot_embedding=cfg.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #No. of pathways (50) or No. of Genes (3917)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        enc = batch['pos encodings']
        image_embeddings = self.image_projection(image_features) + enc
        spot_embeddings = self.spot_projection(spot_features) + enc
        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
#         l2_regularization = 0.001 * torch.norm(spot_embeddings, p=2)
        l1_regularization_spots = 0.0001 * torch.norm(spot_embeddings, p=1)
        l1_regularization_image = 0.0001 * torch.norm(image_embeddings, p=1)
#     + 0.0001 * torch.norm(image_embeddings, p=1)
        loss =  ((images_loss + spots_loss) / 2.0 ).mean() + l1_regularization_spots + l1_regularization_image # shape: (batch_size)
        return loss

class BLEEPWithOptimus(nn.Module):
    def __init__(
        self,
        temperature=1.0,
        image_embedding=cfg.optimus_embedding + cfg.image_embedding,  #512 + 1536 = 2048
        spot_embedding=cfg.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding)#224 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        enc = batch['pos encodings']
        enc = F.normalize(enc, dim=-1)
        optim_feat = batch['st_feat']
        image_features = torch.cat((image_features,optim_feat),dim=1)
        image_embeddings = self.image_projection(image_features) + enc
        spot_embeddings = self.spot_projection(spot_features) + enc
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        targets = torch.eye(logits.shape[0], logits.shape[1]).cuda()
        spot_loss = cross_entropy(logits, targets, reduction='none')
        image_loss = cross_entropy(logits.T, targets.T, reduction='none')
        l1_regularization_spots = 0.0001 * torch.norm(spot_embeddings, p=1)
        l1_regularization_image = 0.0001 * torch.norm(image_embeddings, p=1)
        loss = ((spot_loss+image_loss)/2).mean() + l1_regularization_spots + l1_regularization_image # shape: (batch_size)
        return loss


class DeepPathway(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(cfg.device)
        self.image_model = BLEEPWithOptimus().to(self.device)
        checkpoint = torch.load(cfg.root_path + "saved_weights/itr_01_Bleep+optimus_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".pt")
        self.image_model.load_state_dict(checkpoint)
        for param in self.image_model.parameters():
            param.requires_grad = False  # Freezing the model parameters

        # Initialize MLP
        self.img_linear = MLP(embedding_dim=cfg.projection_dim)  #embedding dimension

    def forward(self, batch):
        images = batch["image"]
        enc = batch['pos encodings']
        true = batch["reduced_expression"]
        optim_feat = batch['st_feat']
        image_features = self.image_model.image_encoder(images)
        image_features = torch.cat((image_features, optim_feat), dim=1)
        image_features = self.image_model.image_projection(image_features)  # + enc.float()
        l1_regularization_image = 0.0001 * torch.norm(image_features, p=1)
        preds = self.img_linear(image_features.float())
        mae_loss = torch.mean(torch.abs(preds - true))
        # Combine with regularization
        loss = mae_loss + l1_regularization_image
        return loss

class BLEEP_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device(cfg.device)
        self.image_model = BLEEPOnly().to(self.device)
        checkpoint = torch.load(cfg.root_path + "saved_weights/itr_01_Bleep_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".pt")
        self.image_model.load_state_dict(checkpoint)
        for param in self.image_model.parameters():
            param.requires_grad = False  # Freezing the model parameters
        self.img_linear = MLP(embedding_dim=cfg.projection_dim)  #embedding dimension

    def forward(self, batch):
        images = batch["image"]
        enc = batch['pos encodings']
        true = batch["reduced_expression"]
        image_features = self.image_model.image_encoder(images)
        image_features = self.image_model.image_projection(image_features)  # + enc.float()
        l1_regularization_image = 0.0001 * torch.norm(image_features, p=1)
        preds = self.img_linear(image_features.float())
        mae_loss = torch.mean(torch.abs(preds - true))
        # Combine with regularization
        loss = mae_loss + l1_regularization_image
        return loss
