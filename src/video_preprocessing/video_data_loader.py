import numpy as np
import pandas as pd
from video_processor import VideoProcessor
from feature_extractor import FeatureExtractor
from config import VideoClassificationConfig


class DataLoader:
    def __init__(self, df: pd.DataFrame, class_vocab: list, config: VideoClassificationConfig):
        self.df = df
        self.class_vocab = class_vocab
        self.config = config
        self.video_processor = VideoProcessor(config)
        self.feature_extractor = FeatureExtractor(config)

    def load_data(self, video_path: str) -> tuple:
        frames = self.video_processor.load_and_preprocess_video(video_path)
        label = self.df.loc[self.df['video_path'] == video_path, 'label'].iloc[0]
        label_idx = self.class_vocab.index(label)
        features = self.feature_extractor.extract_features(frames)
        
        return frames, label_idx, features

    def load_batch(self, batch_indexes: list) -> tuple:
        batch_frames = []
        batch_labels = []
        batch_features = []
        
        for idx in batch_indexes:
            video_path = self.df.iloc[idx]['video_path']
            frames, label_idx, features = self.load_data(video_path)
            batch_frames.append(frames)
            batch_labels.append(label_idx)
            batch_features.append(features)
        
        return np.array(batch_frames), np.array(batch_labels), np.array(batch_features)


# Test the DataLoader class
# if __name__ == "__main__":
#     config = VideoClassificationConfig()
#     # TODO: Load the train.csv file from secrets
#     df = pd.read_csv("/Users/mzitoh/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/4/train.csv")
#     class_vocab = ["SkyDiving", "Biking", "HorseRace"]
#     data_loader = DataLoader(df, class_vocab, config)
#     batch_indexes = [1, 2, 3]
#     batch_frames, batch_labels, batch_features = data_loader.load_batch(batch_indexes)
    
#     print(batch_frames.shape, batch_labels.shape, batch_features.shape)