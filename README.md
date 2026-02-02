# Improving CIFAR-10 Classification

A systematic approach to improving image classification accuracy on the CIFAR-10 dataset using PyTorch.

## Overview

This project explores various techniques to improve deep learning model performance on CIFAR-10:
- Experiment tracking system for comparing different configurations
- GPU acceleration (optimized for RTX 5090)
- Systematic hyperparameter tuning
- Model architecture improvements
- Data augmentation strategies

## Features

- **Automated Experiment Tracking**: Every training run is tracked with hyperparameters, accuracy, and timing
- **GPU Support**: Automatic device detection and optimization for CUDA-enabled GPUs
- **Results Visualization**: Color-coded comparison tables showing all experiments
- **Persistent Storage**: CSV export/import for tracking experiments across sessions

## Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib pandas
```

### Running Experiments

1. Open `image_classifier.ipynb` in Jupyter
2. Run cells 1-3 to setup environment and load data
3. Modify hyperparameters in cell 7 for different experiments
4. Run training cells (7-8)
5. Evaluate and track results (cells 13-15)

## Experiment Tracking

Each experiment automatically records:
- Model architecture
- Hyperparameters (learning rate, batch size, optimizer, etc.)
- Training time
- Test accuracy
- Device used (CPU/GPU)

View all experiments in a comparison table with color-coded accuracy metrics.

## Dataset

CIFAR-10: 60,000 32x32 color images in 10 classes
- 50,000 training images
- 10,000 test images
- Classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck

## Project Structure

```
├── image_classifier.ipynb    # Main notebook with experiment tracking
├── data/                      # CIFAR-10 dataset (auto-downloaded)
├── experiment_results.csv     # Saved experiment history
└── cifar_net.pth             # Saved model weights
```

## Improvement Strategies

### Quick Wins
- Increase epochs (50-100+)
- Add data augmentation
- Increase batch size
- Use Adam optimizer

### Advanced Techniques
- Deeper architectures (ResNet)
- Batch normalization
- Learning rate scheduling
- Ensemble methods

## Results

Track your progress! Each run is saved to `experiment_results.csv` and can be visualized in the results table.

## License

MIT License

## Acknowledgments

- CIFAR-10 dataset from [Canadian Institute for Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch team for the excellent framework
