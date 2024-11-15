from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class VideoClassificationConfig:
    """
    Configuration settings for the video classification model.
    
    Attributes:
    image_size (int): Image size for input frames.
    image_shape (Tuple[int, int, int]): Shape of the input frames.
    max_sequence_length (int): Maximum sequence length for video frames.
    batch_size (int): Batch size for training.
    num_features (int): Number of features extracted from each frame.
    epochs (int): Number of training epochs.
    learning_rate (float): Learning rate for the optimizer.
    """
    image_size: int = 224
    image_shape: Tuple[int, int, int] = (image_size, image_size, 3)
    max_sequence_length: int = 20 # TODO: Change to 150 for final model test
    batch_size: int = 32
    num_features: int = 2048
    epochs: int = 120
    learning_rate: float = 1e-4