from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TrainingConfig:
    # Data parameters
    data_dir: str = "Engine_knock"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    
    # Training parameters
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Model parameters
    models: List[str] = field(default_factory=lambda: ["mobilenet_v2", "efficientnet_b0", "alexnet"])
    
    # Paths
    log_dir: str = "logs"
    plot_dir: str = "plots"
    
    # Dataset splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

config = TrainingConfig() 