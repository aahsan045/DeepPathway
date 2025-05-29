import cv2
import numpy as np
import pandas as pd
import config as cfg
from spatialdata import read_zarr
import torch
import torchvision.transforms as transforms
class STDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_data_path, image_files_path):
        self.whole_image = cv2.imread(image_path)
        self.sdata_object = read_zarr(spatial_data_path)
        self.spatial_pos_csv = self.sdata_object['adata_pathways'].obs
        self.spatial_pos_csv = self.spatial_pos_csv.reset_index().rename(columns={'index': 'barcode'})
        self.spatial_pos_csv.columns = range(self.spatial_pos_csv.shape[1])
        self.spatial_pos_csv = self.spatial_pos_csv[self.spatial_pos_csv[1] == 1]
        self.barcode_csv = self.sdata_object['adata_pathways'].obs_names.values
        self.reduced_matrix = np.array(self.sdata_object['adata_pathways'].X)
        self.images_file_path = np.load(image_files_path, allow_pickle=True)
        self.static_features = np.array(self.sdata_object['optim_feat'].X)

        print("Finished loading all files")

    def get_positional_encoding(self, x, y, W, H, d_model=512):
        # Normalize coordinates
        x_norm = x / W
        y_norm = y / H
        # Initialize positional encoding array
        pos_encoding = np.zeros((d_model,))
        # Define frequencies
        freqs = [1 / (10000 ** (2 * i / d_model)) for i in range(d_model // 4)]
        # Apply sine and cosine functions to normalized coordinates
        pos_encoding[0:d_model // 4] = np.sin(x_norm * np.array(freqs))
        pos_encoding[d_model // 4:d_model // 2] = np.cos(x_norm * np.array(freqs))
        pos_encoding[d_model // 2:3 * d_model // 4] = np.sin(y_norm * np.array(freqs))
        pos_encoding[3 * d_model // 4:] = np.cos(y_norm * np.array(freqs))
        return pos_encoding

    def new_transform(self, image):
        transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x / 255.0),])
        return transform(image)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_csv[idx]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0] #need to change this on tissue_position_list of Hest-1k
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0] #need to change this on tissue_position_list of Hest-1k
        pos_encodings = self.get_positional_encoding(v1, v2, self.whole_image.shape[1], self.whole_image.shape[0], cfg.projection_dim)
        image = cv2.cvtColor(cv2.imread(self.images_file_path[idx]), cv2.COLOR_BGR2RGB)
        image = self.new_transform(image)
        item['image'] = image  # color channel first, then XY
        temp = torch.tensor(self.reduced_matrix[idx, :]).float()
        item['st_feat'] = torch.tensor(self.static_features[idx, :]).float()
        item['reduced_expression'] = temp  # cell x features (3467)
        item['barcode'] = barcode
        item['spatial_coords'] = [v1, v2]
        item['pos encodings'] = pos_encodings
        #         item['labels'] = label
        return item

    def __len__(self):
        return len(self.barcode_csv)