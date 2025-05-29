import timm
from torch import nn
import config as cfg
import torch
import numpy as np
import random
import os
import config as cf
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

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name=cfg.model_name, pretrained=cfg.pretrained, trainable=True
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=cfg.projection_dim,
            dropout=cfg.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class MLP(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=cfg.mlp_projection_dim,
            dropout=cfg.mlp_dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.layer_norm1 = nn.LayerNorm(projection_dim)  # LayerNorm for the projection layer
        self.gelu = nn.GELU()  # You can try ReLU or LeakyReLU too
        self.dropout1 = nn.Dropout(dropout)  # Dropout after the projection

        self.fc1 = nn.Linear(projection_dim, projection_dim)  # Additional layer
        self.layer_norm2 = nn.LayerNorm(projection_dim)  # LayerNorm for the first fully connected layer
        self.dropout2 = nn.Dropout(dropout)  # Dropout after the first FC

        self.fc2 = nn.Linear(projection_dim, cfg.spot_embedding)  # Final layer
        self.last_layer = nn.Sigmoid()

    def forward(self, x):
        projected = self.projection(x)
        x = self.layer_norm1(projected)  # LayerNorm after projection
        x = self.gelu(x)
        x = self.dropout1(x)  # Dropout after projection

        x = self.fc1(x)
        x = self.layer_norm2(x)  # LayerNorm after first FC
        x = self.gelu(x)  # Non-linearity after first FC
        x = self.dropout2(x)  # Dropout after first FC
        x = self.fc2(x)
        x = self.last_layer(x)
        return x
