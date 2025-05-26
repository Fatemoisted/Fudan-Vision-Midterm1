from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import random

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class Caltech101Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法打开图片 {img_path}: {e}")
            image = Image.new('RGB', (100, 100), color='gray')

        if self.transform:
            image = self.transform(image)
            
        return image, label


def load_and_split_caltech101(data_dir, transforms, all_data=False):
    categories = [d for d in os.listdir(data_dir) if not d.startswith('.') and os.path.isdir(os.path.join(data_dir, d))]

    if "BACKGROUND_Google" in categories:
        categories.remove("BACKGROUND_Google")
    
    print(f"发现 {len(categories)} 个类别")
    category_to_idx = {category: idx for idx, category in enumerate(sorted(categories))}
    image_paths = []
    labels = []
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        category_idx = category_to_idx[category]
        files = [f for f in os.listdir(category_dir) if not f.startswith('.') and 
                 os.path.isfile(os.path.join(category_dir, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for file in files:
            image_paths.append(os.path.join(category_dir, file))
            labels.append(category_idx)
    
    print(f"总共找到 {len(image_paths)} 张图片")
    
    samples_by_class = {}
    for i, label in enumerate(labels):
        if label not in samples_by_class:
            samples_by_class[label] = []
        samples_by_class[label].append(i)

    train_indices = []
    test_indices = []
    for label, indices in samples_by_class.items():
        random.seed(42)
        random.shuffle(indices)
        test_size = min(15, len(indices))
        test_indices.extend(indices[:test_size])

        remaining = indices[test_size:]
        if not all_data:
            train_size = min(30, len(remaining))
        else:
            train_size = len(remaining)
        train_indices.extend(remaining[:train_size])
    
    print(f"按类别划分后，训练集包含 {len(train_indices)} 个样本，测试集包含 {len(test_indices)} 个样本")

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )
    
    print(f"从训练集分出验证集后，最终训练集包含 {len(train_paths)} 个样本，验证集包含 {len(val_paths)} 个样本")

    train_dataset = Caltech101Dataset(train_paths, train_labels, transform=transforms['train'])
    val_dataset = Caltech101Dataset(val_paths, val_labels, transform=transforms['val'])
    test_dataset = Caltech101Dataset(test_paths, test_labels, transform=transforms['test'])
    
    return train_dataset, val_dataset, test_dataset, category_to_idx