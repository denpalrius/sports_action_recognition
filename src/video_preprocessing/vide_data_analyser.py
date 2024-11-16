import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


class VideoDataAnalyzer:
    def review_class_distribution(self, dataset, dataset_name):
        class_counts = dataset["label"].value_counts()
        return class_counts

    def compare_class_distributions(self, train_df, val_df, test_df):
        train_class_counts = self.review_class_distribution(train_df, "Train")
        val_class_counts = self.review_class_distribution(val_df, "Validation")
        test_class_counts = self.review_class_distribution(test_df, "Test")

        distribution_df = pd.DataFrame({
            "Train": train_class_counts,
            "Validation": val_class_counts,
            "Test": test_class_counts
        }).fillna(0)

        distribution_df["Average"] = distribution_df.mean(axis=1).round().astype(int)
        return distribution_df

    def plot_class_distribution(self, distribution_df):
        plot_distribution_df = distribution_df.drop(columns="Average")
        plot_distribution_df.plot(kind="bar", figsize=(10, 5))
        plt.title("Class Distribution Comparison Across Train, Validation, and Test Sets")
        plt.xlabel("Class Labels")
        plt.ylabel("Number of Videos")
        plt.legend(title="Dataset")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def count_frames_per_video(self, video_paths):
        frame_counts = []

        for video_path in video_paths: 
            cap = cv2.VideoCapture(video_path)
            count = 0
            
            while cap.isOpened():
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            cap.release()
            frame_counts.append(count)

        return frame_counts

    def visualize_frame_distribution(self, frame_counts):
        plt.figure(figsize=(8, 5))
        sns.violinplot(x=frame_counts)
        plt.title("Violin Plot of Frame Counts per Video")
        plt.xlabel("Number of Frames")
        plt.show()

