import torch
import os
from utils.config import config
from data.dataset import create_dataloaders
from training.trainer import ModelTrainer
from utils.visualization import plot_training_history, plot_confusion_matrix, compare_models

def main():
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
        
        # Initialize trainer
        trainer = ModelTrainer(model_name, device, config)
        
        # Train model
        history = trainer.train(dataloaders)
        
        # Plot training history
        plot_training_history(history, model_name, config.plot_dir)
        
        # Generate confusion matrix
        plot_confusion_matrix(trainer.model, dataloaders['test'],
                            device, model_name, config.plot_dir)
        
        # Test model
        test_loss, test_acc = trainer.test(dataloaders['test'])
        
        # Store results
        results[model_name] = {
            'test_loss': test_loss,
            'test_acc': test_acc
        }
    
    # Compare models
    compare_models(results, config.plot_dir)
    print("\nTraining completed. Results saved in plots/ and logs/ directories.")

if __name__ == "__main__":
    main() 