# Temporal Sequence Modeling for Sports Action Recognition

This project focuses on fine-grained sports action recognition using two main architectures:

1. **CNN-based Sequence Models**: These models combine CNNs for feature extraction with RNNs(GRU layers) for temporal sequence modeling:
   - **VGG19**
   - **InceptionV3**
   - **InceptionV4-ResNet (hybrid model)**
   - **EfficientNetB4**

2. **ViViT (Video Vision Transformer)**: A pure transformer-based approach for end-to-end video classification, capturing both spatial and temporal features.


## Model Architectures

### 1. CNN-based Sequence Models
- **Feature Extractors**: VGG19, InceptionV3, InceptionV4-ResNet, EfficientNetB4
- **Temporal Model**: GRU layers

### 2. ViViT Model
- Transformer-based model for video classification
- Spatiotemporal attention and tubelet embedding

## Evaluation

Each model is evaluated using:
- Accuracy, Precision, Recall, F1-Score
- Training/validation curves
- Confusion matrix

## Acknowledgments

- Dr. Lina Chato
- UCF101 dataset
- TensorFlow team
- All the cited authors
