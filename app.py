import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass, field
import time

# Configuration
@dataclass
class Config:
    # Data parameters
    data_dir: str = "Engine_knock"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    
    # Training parameters
    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Model parameters
    models: List[str] = field(default_factory=lambda: [
        "mobilenet_v2", 
        "efficientnet_b0", 
        "resnet18",
        "resnet50",
        "densenet121",
        "vgg16",
        "mobilenet_v3_small",
        "convnext_tiny",
        "swin_t",
        "regnet_y_400mf"
    ])
    
    # Paths
    log_dir: str = "logs"
    plot_dir: str = "plots"
    
    # Dataset splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Augmentation flag
    use_augmentation: bool = True

# Dataset
class EngineKnockDataset(Dataset):
    def __init__(self, data_dir: str, image_paths: List[Tuple[str, str]], labels: List[int], transform=None):
        self.image_paths = image_paths  # List of tuples (subfolder, filename)
        self.labels = labels
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            subfolder, filename = self.image_paths[idx]
            img_path = os.path.join(self.data_dir, subfolder, filename)
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return torch.zeros(3, *self.image_size), -1  # Return dummy data

# Data loading functions
def get_data_transforms(image_size: Tuple[int, int], use_augmentation: bool) -> Dict[str, transforms.Compose]:
    if use_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    data_transforms = {
        'train': train_transforms,
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

def create_dataloaders(config: Config) -> Dict[str, DataLoader]:
    normal_dir = os.path.join(config.data_dir, 'normal')
    knocking_dir = os.path.join(config.data_dir, 'knocking')
    
    normal_images = [('normal', f) for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    knocking_images = [('knocking', f) for f in os.listdir(knocking_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    all_images = normal_images + knocking_images
    image_paths = all_images
    labels = [0] * len(normal_images) + [1] * len(knocking_images)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=config.test_split, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config.val_split/(config.train_split + config.val_split),
        random_state=42, stratify=y_train_val
    )
    
    assert abs(len(X_train)/len(all_images) - config.train_split) < 0.01, "Data split mismatch"
    assert abs(len(X_val)/len(all_images) - config.val_split) < 0.01, "Data split mismatch"
    
    transforms_dict = get_data_transforms(config.image_size, config.use_augmentation)
    datasets = {
        'train': EngineKnockDataset(config.data_dir, X_train, y_train, transforms_dict['train']),
        'val': EngineKnockDataset(config.data_dir, X_val, y_val, transforms_dict['val']),
        'test': EngineKnockDataset(config.data_dir, X_test, y_test, transforms_dict['test'])
    }
    
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

# Model Trainer
class ModelTrainer:
    def __init__(self, model_name: str, device: torch.device, config: Config):
        self.model_name = model_name
        self.device = device
        self.config = config
        
        # Get model with pretrained weights
        weights = "DEFAULT"
        if model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(weights=weights)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        elif model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=weights)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)
        elif model_name == "resnet18":
            self.model = models.resnet18(weights=weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        elif model_name == "resnet50":
            self.model = models.resnet50(weights=weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        elif model_name == "densenet121":
            self.model = models.densenet121(weights=weights)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)
        elif model_name == "vgg16":
            self.model = models.vgg16(weights=weights)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 2)
        elif model_name == "mobilenet_v3_small":
            self.model = models.mobilenet_v3_small(weights=weights)
            self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)
        elif model_name == "convnext_tiny":
            self.model = models.convnext_tiny(weights=weights)
            self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 2)
        elif model_name == "swin_t":
            self.model = models.swin_t(weights=weights)
            self.model.head = nn.Linear(self.model.head.in_features, 2)
        elif model_name == "regnet_y_400mf":
            self.model = models.regnet_y_400mf(weights=weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', patience=3, factor=0.5, verbose=True
        )
        self.logger = self._setup_logger()
        self.scaler = torch.cuda.amp.GradScaler()
        self.grad_clip = 1.0
        self.start_time = None
        self.epoch_times = []
        
        # Log model info
        self.logger.info(f"Model architecture:\n{self.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"{self.model_name}")
        logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_file = os.path.join(self.config.log_dir, f"{self.model_name}_training_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.start_time = time.time()
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
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
            'val_loss': [], 'val_acc': [],
            'epoch_time': []
        }
        
        early_stop_counter = 0
        best_val_acc = 0.0
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(dataloaders['train'])
            val_loss, val_acc = self.evaluate(dataloaders['val'])
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['epoch_time'].append(self.epoch_times[-1])
            
            self.logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= 5:
                self.logger.info("Early stopping triggered")
                break
            
            self.scheduler.step(val_acc)
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'epoch': range(1, len(history['train_loss']) + 1),
            'train_loss': history['train_loss'],
            'train_acc': history['train_acc'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc'],
            'epoch_time': history['epoch_time'],
            'params': sum(p.numel() for p in self.model.parameters())
        })
        results_df.to_csv(os.path.join(self.config.plot_dir, f'{self.model_name}_results.csv'), index=False)
        
        return history
    
    def test(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.logger.info("Starting evaluation on test set...")
        test_loss, test_acc = self.evaluate(test_loader)
        self.logger.info(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%")
        return test_loss, test_acc

# Visualization functions
def plot_training_history(history: Dict[str, List[float]], model_name: str, save_dir: str):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'))
    plt.close()

def plot_confusion_matrix(model: nn.Module, test_loader: DataLoader,
                         device: torch.device, model_name: str, save_dir: str):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def compare_models(results: Dict[str, Dict[str, float]], save_dir: str):
    models = list(results.keys())
    df = pd.DataFrame(results).T
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, df['test_acc'])
    plt.title('Model Comparison - Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, df['test_loss'])
    plt.title('Model Comparison - Test Loss')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()
    
    df.to_csv(os.path.join(save_dir, 'model_comparison_results.csv'))

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()
    
    # Run with augmentation
    config_aug = Config(use_augmentation=True, log_dir="logs_aug", plot_dir="plots_aug")
    run_experiment(config_aug, "With Augmentation")
    
    # Run without augmentation
    config_no_aug = Config(use_augmentation=False, log_dir="logs_no_aug", plot_dir="plots_no_aug")
    run_experiment(config_no_aug, "Without Augmentation")

def run_experiment(config: Config, experiment_name: str):
    print(f"\nStarting experiment: {experiment_name}")
    
    # Create necessary directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(config)
    print("Data loaders created successfully")
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    for model_name in config.models:
        print(f"\nTraining {model_name}...")
        
        trainer = ModelTrainer(model_name, device, config)
        history = trainer.train(dataloaders)
        plot_training_history(history, model_name, config.plot_dir)
        plot_confusion_matrix(trainer.model, dataloaders['test'],
                            device, model_name, config.plot_dir)
        
        test_loss, test_acc = trainer.test(dataloaders['test'])
        results[model_name] = {
            'test_loss': test_loss,
            'test_acc': test_acc
        }
    
    # Compare models
    compare_models(results, config.plot_dir)
    print("\nTraining completed. Results saved in plots/ and logs/ directories.")
    
    # After training all models
    analyzer = DataAnalyzer(config)
    analyzer.analyze_all()

class DataAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.plot_dir, exist_ok=True)
        
    def load_all_results(self) -> pd.DataFrame:
        """Load all model results from CSV files"""
        results = []
        for model_name in self.config.models:
            csv_path = os.path.join(self.config.plot_dir, f'{model_name}_results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['model'] = model_name
                results.append(df)
        return pd.concat(results, ignore_index=True)
    
    def plot_accuracy_comparison(self, results: pd.DataFrame):
        """Plot comparison of all models' accuracy metrics"""
        plt.figure(figsize=(14, 6))
        
        # Training and validation accuracy
        plt.subplot(1, 2, 1)
        sns.lineplot(data=results, x='epoch', y='train_acc', hue='model', style='model', 
                    markers=True, dashes=False)
        plt.title('Training Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(title='Model')
        
        # Validation accuracy
        plt.subplot(1, 2, 2)
        sns.lineplot(data=results, x='epoch', y='val_acc', hue='model', style='model',
                    markers=True, dashes=False)
        plt.title('Validation Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(title='Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'accuracy_comparison.png'))
        plt.close()
    
    def plot_loss_comparison(self, results: pd.DataFrame):
        """Plot comparison of all models' loss metrics"""
        plt.figure(figsize=(14, 6))
        
        # Training and validation loss
        plt.subplot(1, 2, 1)
        sns.lineplot(data=results, x='epoch', y='train_loss', hue='model', style='model',
                    markers=True, dashes=False)
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(title='Model')
        
        # Validation loss
        plt.subplot(1, 2, 2)
        sns.lineplot(data=results, x='epoch', y='val_loss', hue='model', style='model',
                    markers=True, dashes=False)
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(title='Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'loss_comparison.png'))
        plt.close()
    
    def plot_epoch_time_comparison(self, results: pd.DataFrame):
        """Plot comparison of epoch times across models"""
        plt.figure(figsize=(8, 6))
        sns.barplot(data=results.groupby('model')['epoch_time'].mean().reset_index(),
                   x='model', y='epoch_time')
        plt.title('Average Epoch Time Comparison')
        plt.xlabel('Model')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'epoch_time_comparison.png'))
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot all models' confusion matrices in a grid"""
        num_models = len(self.config.models)
        cols = 3
        rows = (num_models + cols - 1) // cols
        
        plt.figure(figsize=(cols * 6, rows * 5))
        
        for i, model_name in enumerate(self.config.models):
            cm_path = os.path.join(self.config.plot_dir, f'{model_name}_confusion_matrix.png')
            if os.path.exists(cm_path):
                cm_img = plt.imread(cm_path)
                plt.subplot(rows, cols, i + 1)
                plt.imshow(cm_img)
                plt.title(f'{model_name} Confusion Matrix')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'all_confusion_matrices.png'))
        plt.close()
    
    def plot_parameter_count_comparison(self, results: pd.DataFrame):
        """Plot comparison of model parameter counts"""
        param_counts = results.groupby('model')['params'].first().reset_index()
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=param_counts, x='model', y='params')
        plt.title('Model Parameter Count Comparison')
        plt.xlabel('Model')
        plt.ylabel('Number of Parameters')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plot_dir, 'parameter_count_comparison.png'))
        plt.close()
    
    def analyze_all(self):
        """Run all analysis and generate all plots"""
        results = self.load_all_results()
        
        if not results.empty:
            self.plot_accuracy_comparison(results)
            self.plot_loss_comparison(results)
            self.plot_epoch_time_comparison(results)
            self.plot_parameter_count_comparison(results)
            self.plot_confusion_matrices()
            
            # Save comprehensive results
            results.to_csv(os.path.join(self.config.plot_dir, 'all_results.csv'), index=False)
            print("Data analysis completed. Plots saved in plots/ directory.")
        else:
            print("No results found for analysis.")

if __name__ == "__main__":
    main() 