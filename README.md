# Engine Knock Detection using CNNs

This project implements and compares different CNN architectures (MobileNetV2, EfficientNet-B0, AlexNet) for engine knock detection using PyTorch.

## Project Structure
```
.
├── src/
│   ├── data/
│   │   └── dataset.py         # Data loading and preprocessing
│   ├── models/
│   ├── training/
│   │   └── trainer.py         # Model training and evaluation
│   └── utils/
│       ├── config.py          # Configuration parameters
│       └── visualization.py    # Plotting and visualization
├── logs/                      # Training logs
├── plots/                     # Performance plots
└── Engine_knock/             # Dataset directory
    ├── normal/
    └── knocking/
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset is organized in the following structure:
```
Engine_knock/
├── normal/
│   ├── image1.jpg
│   └── ...
└── knocking/
    ├── image1.jpg
    └── ...
```

## Usage

Run the training script:
```bash
python src/main.py
```

This will:
1. Train multiple CNN models on the dataset
2. Generate training history plots
3. Create confusion matrices
4. Compare model performances
5. Save all results in the `plots/` and `logs/` directories

## Results

The training results for each model will be saved in:
- `plots/`: Training history plots, confusion matrices, and model comparisons
- `logs/`: Detailed training logs with metrics

## Models

The project includes the following pre-trained models:
- MobileNetV2
- EfficientNet-B0
- AlexNet

Each model is fine-tuned for binary classification of engine knock detection. 