from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import VideoClassificationConfig


class FeatureExtractor:
    def __init__(self, config: VideoClassificationConfig):
        self.config = config
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        """
        Builds the InceptionV3 feature extractor.

        Returns:
        keras.Model: Feature extractor model.
        """
        base_model = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=self.config.image_shape,
        )
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input(self.config.image_shape)
        preprocessed = preprocess_input(inputs)
        outputs = base_model(preprocessed)

        return keras.Model(inputs, outputs, name="feature_extractor")

    def _extract_features(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features from frames using the feature extractor model.

        Args:
        frames (np.ndarray): Input frames.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Extracted features and mask.
        """
        if len(frames) == 0:
            return np.zeros(
                (self.config.max_sequence_length, self.config.num_features)
            ), np.zeros(self.config.max_sequence_length)

        # Initialize features and mask arrays with zeros 
        features = np.zeros((self.config.max_sequence_length, self.config.num_features))
        mask = np.zeros(self.config.max_sequence_length)

        for i, frame in enumerate(frames):
            if i >= self.config.max_sequence_length:
                break
            features[i] = self.feature_extractor.predict(
                frame[np.newaxis, ...], verbose=0
            )
            mask[i] = 1

        return features, mask
