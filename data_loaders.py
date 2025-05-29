from dataset import STDataset
import torch
from torch.utils.data import DataLoader
import config as cfg
def build_loaders(a,root_path):
    # One sample will be left out for testing
    print("Building loaders")
    data=[]
    for i in range(0,len(a)):
        d = STDataset(image_path =root_path+"wsis/"+a[i]+".tif",
                      spatial_data_path = root_path+"SpatialData/"+a[i]+"_spatial_data.zarr",
                      image_files_path = root_path+"H&E patches/"+a[i]+"_sample_file_paths.npy"
                      )
        data.append(d)

    dataset = torch.utils.data.ConcatDataset(data)

    train_size = int(cfg.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(cfg.seed))
    print(len(train_dataset), len(test_dataset))
    print("train/test split completed")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Finished building loaders")
    return train_loader, test_loader
def build_loaders_inference(a,root_path):
    print("Building loaders")
    data = []
    for i in range(0, len(a)):
        d = STDataset(image_path=root_path + "wsis/" + a[i] + ".tif",
                      spatial_data_path=root_path + "SpatialData/" + a[i] + "_spatial_data.zarr",
                      image_files_path=root_path + "H&E patches/" + a[i] + "_sample_file_paths.npy"
                      )
        data.append(d)

    dataset = torch.utils.data.ConcatDataset(data)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    print("Finished building inference loaders")
    return test_loader
