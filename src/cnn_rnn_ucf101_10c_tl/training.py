import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score
from cnn_rnn_ucf101_10c_tl.video_data_generator import VideoDataGenerator
from model import VideoClassifierModel
from config import VideoClassificationConfig
import matplotlib.pyplot as plt
import time
import os


class VideoClassifierTrainer:
    """
    Class for training the video classification model.
    """

    def __init__(self, config: VideoClassificationConfig):
        """
        Initializes the VideoClassifierTrainer.
        
        Args:
        config (VideoClassificationConfig): Configuration settings.
        """
        self.config = config

    def create_data_generators(self, train_df, val_df, class_vocab):
        """
        Creates training and validation data generators.
        
        Args:
        train_df (pandas.DataFrame): Training data.
        val_df (pandas.DataFrame): Validation data.
        class_vocab (List[str]): Class labels.
        
        Returns:
        tuple: Training and validation data generators.
        """
        train_generator = VideoDataGenerator(train_df, class_vocab, self.config, is_training=True)
        val_generator = VideoDataGenerator(val_df, class_vocab, self.config, is_training=False)
        return train_generator, val_generator

    def build_model(self, class_vocab):
        """
        Builds the video classification model.
        
        Args:
        class_vocab (List[str]): Class labels.
        
        Returns:
        keras.Model: Video classification model.
        """
        return VideoClassifierModel(self.config, len(class_vocab)).get_model()

    def define_callbacks(self):
        """
        Defines training callbacks.
        
        Returns:
        list: List of training callbacks.
        """
        return [
            keras.callbacks.ModelCheckpoint(
                # TODO: Pass as param
                f'../models/video_classifier_v{int(time.time())}.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=3,
                min_lr=1e-6
            )
        ]

    def train_model(self, train_df, val_df, class_vocab):
        """
        Trains the video classification model.
        
        Args:
        train_df (pandas.DataFrame): Training data.
        val_df (pandas.DataFrame): Validation data.
        class_vocab (List[str]): Class labels.
        
        Returns:
        tuple: Trained model and training history.
        """
        train_generator, val_generator = self.create_data_generators(train_df, val_df, class_vocab)
        model = self.build_model(class_vocab)
        callbacks = self.define_callbacks()
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.config.epochs,
            callbacks=callbacks
        )
        
        return model, history


# Test the VideoClassifierTrainer class
# if __name__ == "__main__":
#     # Load data and configuration
#     train_df = pd.read_csv("train.csv")
#     val_df = pd.read_csv("val.csv")
#     class_vocab = ["class1", "class2", "class3"]
#     config = VideoClassificationConfig()
    
#     # Create trainer
#     trainer = VideoClassifierTrainer(config)
    
#     # Train model
#     model, history = trainer.train_model(train_df, val_df, class_vocab)
    
#     # Plot training history
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.legend()
#     plt.show()