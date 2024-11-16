from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class VideoClassificationConfig:
    image_size: int = 224
    image_shape: Tuple[int, int, int] = (image_size, image_size, 3)
    max_sequence_length: int = 20 # TODO: Change to 150 for final model test
    batch_size: int = 32
    num_features: int = 2048
    epochs: int = 120
    learning_rate: float = 1e-4