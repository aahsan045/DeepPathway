
import numpy as np
import os
from tqdm import tqdm
import config as cfg
import torch
import random
import matplotlib.pyplot as plt

random.seed(cfg.seed)

# Set seed for NumPy
np.random.seed(cfg.seed)

# Set seed for PyTorch
torch.manual_seed(cfg.seed)

# Set seed for CUDA
torch.cuda.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

# Ensure deterministic behavior in CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variable for Python hash
os.environ['PYTHONHASHSEED'] = str(cfg.seed)

from data_loaders import build_loaders
from models import BLEEPOnly, BLEEPWithOptimus, BLEEP_MLP, BLEEP_Optimus_MLP
from utils import *

def train_epoch(model, train_loader, optimizer, lr_scheduler=None):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda(cfg.device) for k, v in batch.items() if k == "image" or k == "reduced_expression" or k =='pos encodings' or k =='st_feat'}
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def test_epoch(model, test_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda(cfg.device) for k, v in batch.items() if k == "image" or k == "reduced_expression" or k =='pos encodings' or k =='st_feat'}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def main():
    tr_loss = []
    tst_loss = []
    best_loss = float('inf')
    best_epoch = 0

    # Bleep OR Bleep+optimus OR cnn+mlp+optimus OR cnn+mlp
    if cfg.method=='Bleep':
        model=BLEEPOnly().to(cfg.device)
        print("Model is BLEEP only with Image Encoder of ResNet18")
    elif cfg.method=='BLEEP+optimus':
        model=BLEEPWithOptimus().to(cfg.device)
        print("Model is BLEEP with Optimus using Image Encoder of ResNet18")
    elif cfg.method=='cnn+mlp+optimus':
        model=BLEEP_Optimus_MLP(cfg.root_path,cfg.test_sample,cfg.dataset).to(cfg.device)
        print("Model is BLEEP+Optimus with MLP using Image Encoder of ResNet18")
    elif cfg.method=='cnn+mlp':
        model=BLEEP_MLP(cfg.root_path,cfg.test_sample,cfg.dataset).to(cfg.device)
        print("Model is BLEEP with MLP using Image Encoder of ResNet18")
    print("Finished Loading Model.....:", cfg.method)
    train_loader, test_loader=build_loaders(cfg.train_samples,cfg.root_path)
    save_path=cfg.root_path + "saved_weights/itr_01" + cfg.method + "_" + cfg.dataset + "_pathways_" + cfg.test_sample + ".pt"
    early_stopping=EarlyStopping(patience=cfg.patience, verbose=True,path=save_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print("Training Started of Model",cfg.method)
    for epoch in range(cfg.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer)
        tr_loss.append(train_loss.avg)

        model.eval()
        with torch.no_grad():
            test_loss = test_epoch(model, test_loader)
            lss = test_loss.avg
            tst_loss.append(lss)

        if lss < best_loss:
            best_loss = lss
            best_epoch = epoch
            print(f"Best loss at Epoch: {epoch + 1}")
            torch.save(model.state_dict(), save_path)
        # Early Stopping Check
        early_stopping(lss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    plt.plot(tr_loss)
    plt.plot(tst_loss)
    plt.legend(['train', 'valid'])
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.show()
    return
if __name__ == "__main__":
    main()