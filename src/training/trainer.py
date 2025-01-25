import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import time
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, model_name: str, device: torch.device, config):
        self.model_name = model_name
        self.device = device
        self.config = config
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.model_name}")
        logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config.log_dir, f"{self.model_name}_training_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _create_model(self) -> nn.Module:
        if self.model_name == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        elif self.model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        eval_loss = running_loss / len(dataloader)
        eval_acc = 100. * correct / total
        return eval_loss, eval_acc
    
    def train(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, List[float]]:
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.config.epochs):
            # Train phase
            train_loss, train_acc = self.train_epoch(dataloaders['train'])
            
            # Validation phase
            val_loss, val_acc = self.evaluate(dataloaders['val'])
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log progress
            self.logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(),
                          os.path.join(self.config.log_dir, f"{self.model_name}_best.pth"))
        
        return history
    
    def test(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.logger.info("Starting evaluation on test set...")
        test_loss, test_acc = self.evaluate(test_loader)
        self.logger.info(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%")
        return test_loss, test_acc 