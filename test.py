import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import pdb
from utils import set_seed, plot_training_results
from data import data_transforms, load_and_split_caltech101
from model import get_model, configure_model_for_finetuning
set_seed()
model_path = "scratch_resnet18_lr_0.001_epoch_100_alldata_True.pth"
config = {
    "batch_size": 32,
    "num_epochs": 25,
    "num_epochs_scratch": 100,
    "lr_pretrained_backbone": 1e-4, 
    "lr_pretrained_fc": 1e-3,      
    "lr_scratch": 1e-3,          
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "num_workers": 4,
    "model_name": "resnet18",
    "feature_extract": False,    
    "all_data": False
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data_dir = "dataset/caltech-101/101_ObjectCategories" 

# train_dataset, val_dataset, test_dataset = split_caltech101_dataset(data_dir, data_transforms)
train_dataset, val_dataset, test_dataset, category_to_idx = load_and_split_caltech101(data_dir, data_transforms, config["all_data"])
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
print(f"测试集大小: {len(test_dataset)}")
model_scratch = get_model(model_name=config["model_name"], pretrained=False)
model_scratch.load_state_dict(torch.load(model_path))
model_scratch = model_scratch.to(device)

def evaluate_model(model, test_loader):
    """
    在测试集上评估模型性能
    """
    model.eval()
    
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    return test_acc

acc = evaluate_model(model_scratch, test_loader)
print(f"Test Accuracy: {acc}")

