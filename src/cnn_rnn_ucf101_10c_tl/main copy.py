import pandas as pd
from cnn_rnn_ucf101_10c_tl.video_data_generator import VideoDataGenerator
from model import build_model
from training import train_model
from utils import plot_training_history, evaluate_model, save_model
from config import VideoClassificationConfig

def main():
    # Load configuration
    config = VideoClassificationConfig()
    
    # Load data
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    class_vocab = ["class1", "class2", "class3"]
    
    # Create data generators
    train_generator = VideoDataGenerator(train_df, class_vocab, config, is_training=True)
    val_generator = VideoDataGenerator(val_df, class_vocab, config, is_training=False)
    test_generator = VideoDataGenerator(test_df, class_vocab, config, is_training=False)
    
    # Build and train model
    model, history = train_model(train_df, val_df, class_vocab, config)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on test dataset
    accuracy, report, matrix = evaluate_model(model, test_generator)
    print(f"Test Accuracy: {accuracy:.3f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    
    # Save model
    save_model(model, "video_classifier.keras")


if __name__ == "__main__":
    main()