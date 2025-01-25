import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class EngineKnockDataset(Dataset):
    def __init__(self, data_dir: str, image_paths: List[Tuple[str, str]], labels: List[int], transform=None):
        self.image_paths = image_paths  # List of tuples (subfolder, filename)
        self.labels = labels
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subfolder, filename = self.image_paths[idx]
        img_path = os.path.join(self.data_dir, subfolder, filename)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms(image_size: Tuple[int, int]) -> Dict[str, transforms.Compose]:
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def create_dataloaders(config) -> Dict[str, DataLoader]:
    # Get all image paths and labels
    normal_dir = os.path.join(config.data_dir, 'normal')
    knocking_dir = os.path.join(config.data_dir, 'knocking')
    
    normal_images = [('normal', f) for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    knocking_images = [('knocking', f) for f in os.listdir(knocking_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    all_images = normal_images + knocking_images
    image_paths = all_images
    labels = [0] * len(normal_images) + [1] * len(knocking_images)
    
    # Split dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=config.test_split, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.val_split/(config.train_split + config.val_split),
        random_state=42, stratify=y_train_val
    )
    
    # Create datasets
    transforms_dict = get_data_transforms(config.image_size)
    datasets = {
        'train': EngineKnockDataset(config.data_dir, X_train, y_train, transforms_dict['train']),
        'val': EngineKnockDataset(config.data_dir, X_val, y_val, transforms_dict['val']),
        'test': EngineKnockDataset(config.data_dir, X_test, y_test, transforms_dict['test'])
    }
    
    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=config.batch_size,
            shuffle=(split == 'train'),
            num_workers=config.num_workers
        )
        for split in ['train', 'val', 'test']
    }
    
    return dataloaders 