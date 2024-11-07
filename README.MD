# Adapting Video Vision Transformer (ViViT) for Sports Action Recognition

- This project aims to implement and evaluate a **Video Vision Transformer (ViViT)** for sports action
recognition on the UCF Sports Action Dataset, with a specific focus on how well ViViT can
perform on smaller datasets.
- We aim to explore the advantages of using ViViT to capture complex spatiotemporal patterns from limited data and the challenges sports videos pose.
- Broadcast sports footage usually includes diverse sports actions and movements, captured on varying camera viewpoints.
- We will compare ViViT’s performance against other state-of-the-art video classification models such as CNN-RNN, ~~Inflated 3D Convnet(I3D)~~ and the 3D ResNet.

## Overview

This project implements three different video classification architectures:
1. **CNN-RNN**: Combines InceptionV3 for spatial features with GRU layers for temporal analysis
2. **ViViT**: Pure transformer-based approach for end-to-end video classification
3. **ResNet**: Deep residual learning with 3D convolutions (3D-ResNet)

## Key Features

- Multiple architecture implementations for comparison
- Pre-trained backbone networks
- Support for variable-length video inputs
- Comprehensive evaluation metrics
- Data preprocessing and augmentation pipelines

## Requirements

```python
keras
tensorflow
numpy
pandas
opencv-python
scikit-learn
imageio
imutils
kagglehub
einops  # for ViViT
```

## Project Structure

```
├── models/
│   ├── cnn_rnn/           # CNN-RNN model files
│   ├── vivit/             # ViViT model files
│   └── resnet/            # ResNet model files
├── train/                 # Training video files
├── test/                  # Test video files
├── train.csv             # Training metadata
├── test.csv              # Test metadata
└── checkpoints/          # Saved model weights
```

## Model Architectures

### 1. CNN-RNN Model
- Feature Extractor: InceptionV3 (pre-trained on ImageNet)
- Sequence Model: GRU layers (16 → 8 units)
- Dense layers: 64 → 32 → num_classes

### 2. ViViT Model
- Pure transformer architecture for video
- Spatiotemporal attention mechanism
- Patch embedding of video frames
- Multi-head self-attention layers

### 3. ResNet Model
- 3D convolutional layers
- Residual connections
- Temporal pooling
- Global average pooling

<!-- ## Usage

1. Data Preprocessing:
```python
# Load and preprocess videos
train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")
```

2. Training (for any model):
```python
# Select model type
model = get_model(model_type="cnn_rnn")  # or "vivit" or "resnet"

# Train the model
model_trained, history = train_and_evaluate_model(weights_file_path, model)
```

3. Evaluation:
```python
# Evaluate model performance
evaluate_model_on_test_set(model, test_data, test_labels, class_vocab)
``` -->

## Model Performance Comparison

| Model    | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|---------|-----------|
| CNN-RNN  | TBD      | TBD       | TBD     | TBD       |
| ViViT    | TBD      | TBD       | TBD     | TBD       |
| ResNet   | TBD      | TBD       | TBD     | TBD       |

## Evaluation Metrics

Each model is evaluated using:
- Training/validation curves
- Confusion matrix
- Per-class accuracy
- Classification report (Precision, recall, and F1-score)

## Future Improvements

- Ensemble methods combining multiple architectures
- Hyperparameter optimization
- Additional data augmentation techniques
- Model compression for deployment

## Acknowledgments

- Dr. Lina Chato
- UCF101 dataset
- TensorFlow team
- ViViT paper authors
- ResNet paper authors
