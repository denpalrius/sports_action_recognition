import numpy as np
from tensorflow import keras
from typing import List
from config import VideoClassificationConfig

class VideoDataGenerator(keras.utils.Sequence):
    def __init__(self, df, class_vocab: List[str], config: VideoClassificationConfig, is_training: bool = True):
        self.df = df
        self.config = config
        self.is_training = is_training
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_vocab)}
        self.indexes = np.arange(len(self.df))
        
        # Initialize feature extractor
        self.feature_extractor = self._build_feature_extractor()
        
        # Shuffle on init if training
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.config.batch_size))

    def __getitem__(self, idx):
        # Get batch indexes
        batch_indexes = self.indexes[idx * self.config.batch_size:(idx + 1) * self.config.batch_size]
        
        # Initialize batch arrays
        batch_features = np.zeros((len(batch_indexes), self.config.max_sequence_length, self.config.num_features))
        batch_masks = np.zeros((len(batch_indexes), self.config.max_sequence_length))
        batch_labels = np.zeros(len(batch_indexes))
        
        # Process each video in the batch
        for i, idx in enumerate(batch_indexes):
            # Load and preprocess video
            frames = self._load_and_preprocess_video(self.df.iloc[idx]['video_path'])
            
            # Extract features and create mask
            features, mask = self._extract_features(frames)
            
            # Store in batch arrays
            batch_features[i] = features
            batch_masks[i] = mask
            batch_labels[i] = self.class_to_idx[self.df.iloc[idx]['label']]
        
        return [batch_features, batch_masks], batch_labels

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

    # def _load_and_preprocess_video(self, video_path):
    #     """
    #     Loads and preprocesses a video.
        
    #     Args:
    #     video_path (str): Path to the video file.
        
    #     Returns:
    #     np.ndarray: Preprocessed video frames.
    #     """
    #     frames = []
    #     cap = cv2.VideoCapture(video_path)
        
    #     try:
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
                
    #             # Preprocess frame
    #             frame = self._crop_center_square(frame)
    #             frame = cv2.resize(frame, (self.config.image_size, self.config.image_size))
    #             frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
    #             frames.append(frame)
                
    #             if len(frames) == self.config.max_sequence_length:
    #                 break
    #     finally:
    #         cap.release()
        
    #     return np.array(frames)

    # def _crop_center_square(self, frame):
    #     """
    #     Crops the center square from a frame.
        
    #     Args:
    #     frame (np.ndarray): Input frame.
        
    #     Returns:
    #     np.ndarray: Cropped frame.
    #     """
    #     y, x = frame.shape[0:2]
    #     min_dim = min(y, x)
    #     start_x = (x // 2) - (min_dim // 2)
    #     start_y = (y // 2) - (min_dim // 2)
    #     return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

# # Example usage
# if __name__ == "__main__":
#     # Load data and configuration
#     df = pd.read_csv("data.csv")
#     config = VideoClassificationConfig()
#     class_vocab = ["class1", "class2", "class3"]
    
#     # Create data generator
#     generator = VideoDataGenerator(df, class_vocab, config)
    
#     # Iterate through batches
#     for batch in generator:
#         features, labels = batch
#         # Process batch
#         pass