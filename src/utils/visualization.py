import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import os
from sklearn.metrics import confusion_matrix
import torch
import pandas as pd

def plot_training_history(history: Dict[str, List[float]], model_name: str, save_dir: str):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
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

def plot_confusion_matrix(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
                         device: torch.device, model_name: str, save_dir: str):
    """Generate and plot confusion matrix."""
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
    """Create comparison plots for different models."""
    models = list(results.keys())
    metrics = ['test_acc', 'test_loss']
    
    # Create DataFrame for easy plotting
    df = pd.DataFrame(results).T
    
    plt.figure(figsize=(12, 5))
    
    # Plot test accuracy comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, df['test_acc'])
    plt.title('Model Comparison - Test Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Plot test loss comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, df['test_loss'])
    plt.title('Model Comparison - Test Loss')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()
    
    # Save results to CSV
    df.to_csv(os.path.join(save_dir, 'model_comparison_results.csv')) 