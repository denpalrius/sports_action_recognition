import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def plot_training_history(history):
    """
    Plots training history metrics.
    
    Args:
    history (keras.History): Training history.
    """
    metrics = ['loss', 'accuracy']
    plt.figure(figsize=(12, 4))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_generator):
    """
    Evaluates the model on the test dataset.
    
    Args:
    model (keras.Model): Trained model.
    test_generator (VideoDataGenerator): Test data generator.
    
    Returns:
    tuple: Accuracy, classification report, and confusion matrix.
    """
    y_pred = model.predict(test_generator)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true = test_generator.labels
    
    accuracy = accuracy_score(y_true, y_pred_class)
    report = classification_report(y_true, y_pred_class)
    matrix = confusion_matrix(y_true, y_pred_class)
    
    return accuracy, report, matrix


def save_model(model, path):
    """
    Saves the model to the specified path.
    
    Args:
    model (keras.Model): Trained model.
    path (str): Model save path.
    """
    model.save(path)


def load_model(path):
    """
    Loads the model from the specified path.
    
    Args:
    path (str): Model load path.
    
    Returns:
    keras.Model: Loaded model.
    """
    return tf.keras.models.load_model(path)


def plot_confusion_matrix(matrix):
    """
    Plots the confusion matrix.
    
    Args:
    matrix (np.ndarray): Confusion matrix.
    """
    plt.imshow(matrix, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()


def plot_classification_report(report):
    """
    Plots the classification report.
    
    Args:
    report (str): Classification report.
    """
    print(report)
    
    
# TODO: Save all the graphs and models to the output directory