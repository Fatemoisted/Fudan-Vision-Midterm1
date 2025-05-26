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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data_dir = "dataset/caltech-101/101_ObjectCategories"  

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
    "model_name": "alexnet",
    "feature_extract": False,      
    "all_data": False
}

# train_dataset, val_dataset, test_dataset = split_caltech101_dataset(data_dir, data_transforms)
train_dataset, val_dataset, test_dataset, category_to_idx = load_and_split_caltech101(data_dir, data_transforms, config["all_data"])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

model_pretrained = get_model(model_name=config["model_name"], pretrained=True)
model_scratch = get_model(model_name=config["model_name"], pretrained=False)


model_pretrained = model_pretrained.to(device)
model_scratch = model_scratch.to(device)

optimizer_pretrained = configure_model_for_finetuning(
    model_pretrained, 
    feature_extract=config["feature_extract"],
    lr_backbone=config["lr_pretrained_backbone"],
    lr_fc=config["lr_pretrained_fc"],
    momentum=config["momentum"],
    weight_decay=config["weight_decay"]
)

optimizer_scratch = optim.SGD(
    model_scratch.parameters(), 
    lr=config["lr_scratch"], 
    momentum=config["momentum"],
    weight_decay=config["weight_decay"]
)

criterion = nn.CrossEntropyLoss()

def train_model(model, dataloaders, test_dataloader, criterion, optimizer, config, is_inception=False, model_type="model"):

    if model_type == "scratch":
        run_name = f"{model_type}_{config['model_name']}_lr_{config['lr_scratch']}_epoch_{config['num_epochs_scratch']}_alldata_{config['all_data']}"
    else:
        run_name = f"{model_type}_{config['model_name']}_lrfc_{config['lr_pretrained_fc']}_lrbb_{config['lr_pretrained_backbone']}_extract_{config['feature_extract']}_epoch_{config['num_epochs']}_alldata_{config['all_data']}"
    writer = SummaryWriter(f'runs/{run_name}')
    
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    if model_type == "scratch":
        num_epochs = config["num_epochs_scratch"]
    else:
        num_epochs = config["num_epochs"]
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
   
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
        model.eval()
        test_acc = evaluate_model(model, test_dataloader)
        model.train()
        print(f"Test Acc: {test_acc}")
        writer.add_scalar("Test Accuracy", test_acc, epoch)
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(), f'{run_name}.pth')

    config_str = "\n".join([f"{k}: {v}" for k, v in config.items()])
    writer.add_text("Configuration", config_str)
    
    return model, val_acc_history

def evaluate_model(model, test_loader):
    model.eval()
    
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    
    # 计算准确率
    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    return test_acc

dataloaders = {
    'train': train_loader,
    'val': val_loader
}
print("Training pretrained model...")
model_pretrained, hist_pretrained = train_model(
    model_pretrained, 
    dataloaders, 
    test_loader,
    criterion, 
    optimizer_pretrained, 
    config,
    model_type="pretrained"
)
print("Training model from scratch...")
model_scratch, hist_scratch = train_model(
    model_scratch, 
    dataloaders, 
    test_loader,
    criterion, 
    optimizer_scratch, 
    config,
    model_type="scratch"
)
print("Evaluating pretrained model...")
pretrained_acc = evaluate_model(model_pretrained, test_loader)

print("Evaluating model trained from scratch...")
scratch_acc = evaluate_model(model_scratch, test_loader)

print(f"Pretrained model test accuracy: {pretrained_acc:.4f}")
print(f"Scratch model test accuracy: {scratch_acc:.4f}")
print(f"Performance improvement with pretraining: {(pretrained_acc - scratch_acc) * 100:.2f}%")
plot_training_results(hist_pretrained, hist_scratch)