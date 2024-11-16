import pandas as pd
import os
from cnn_rnn_ucf101_10c_tl.vide_data_analyser import VideoDataAnalyzer
from kagglehub import dataset_download
from cnn_rnn_ucf101_10c_tl.video_data_generator import VideoDataGenerator
from training import VideoClassifierTrainer
from utils import plot_training_history, evaluate_model, save_model
from config import VideoClassificationConfig


class VideoClassifierPipeline:
    def __init__(self, config: VideoClassificationConfig):
        self.config = config
        self.trainer = VideoClassifierTrainer(config)

    def download_dataset(self):
        return dataset_download("matthewjansen/ucf101-action-recognition")

    def load_dataset(self, dataset_type, path):
        sports_actions = [
            "SkyDiving",
            "Biking",
            "HorseRace",
            "Surfing",
            "TennisSwing",
            "Punch",
            "Basketball",
            "JumpRope",
            "Archery",
            "Skiing",
        ]
        
        dataset_path = os.path.join(path, f"{dataset_type}.csv")
        dataset = pd.read_csv(dataset_path)
        
        # Filter dataset to only include the specified sports actions
        filtered_dataset = dataset[dataset["label"].isin(sports_actions)]
        
        return pd.DataFrame(
            {
                "label": filtered_dataset["label"],
                "video_name": filtered_dataset["clip_name"],
                "rel_path": filtered_dataset["clip_path"],
                "video_path": filtered_dataset["clip_path"].apply(lambda x: f"{path}{x}"),
            }
        )

    def print_dataset_info(self, train_df, val_df, test_df):
        print(f"Total videos for training: {len(train_df)}")
        print(f"Total videos for validation: {len(val_df)}")
        print(f"Total videos for testing: {len(test_df)}")
        
        print("Number of unique classes in training set: ", len(train_df["label"].unique()))
        print("Number of unique classes in validation set: ", len(val_df["label"].unique()))
        print("Number of unique classes in test set: ", len(test_df["label"].unique()))
        
        print("\nLabels: \n", train_df["label"].unique())

    def train_model(self, train_df, val_df, class_vocab):
        return self.trainer.train_model(train_df, val_df, class_vocab)

    def evaluate_model(self, model, test_df, class_vocab):
        test_generator = VideoDataGenerator(test_df, class_vocab, self.config, is_training=False)
        return evaluate_model(model, test_generator)

    def run_pipeline(self):
        path = self.download_dataset()
        print("Path to dataset files: \n", path)
        print("\nFiles in dataset directory:\n", os.listdir(path))
        
        train_df = self.load_dataset("train", path)
        val_df = self.load_dataset("val", path)
        test_df = self.load_dataset("test", path)
        
        self.print_dataset_info(train_df, val_df, test_df)
        
        # Vide data analysis
        analyzer = VideoDataAnalyzer()
        distribution_df = analyzer.compare_class_distributions(train_df, val_df, test_df)
        print("Combined average number of videos per class:")
        print(distribution_df)
        analyzer.plot_class_distribution(distribution_df)

        frame_counts = analyzer.count_frames_per_video(train_df["video_path"].values)
        print("Standard deviation of frame counts:", np.std(frame_counts))
        analyzer.visualize_frame_distribution(frame_counts)
        
        # Train model
        class_vocab = train_df["label"].unique().tolist()
        model, history = self.train_model(train_df, val_df, class_vocab)
        plot_training_history(history)
        
        # Evaluate model
        accuracy, report, matrix = self.evaluate_model(model, test_df, class_vocab)
        print(f"Test Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(matrix)
        save_model(model, "video_classifier.keras")


# if __name__ == "__main__":
#     pipeline = VideoClassifierPipeline(VideoClassificationConfig())
#     pipeline.run_pipeline()