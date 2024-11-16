import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def plot_training_history(history):
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
    y_pred = model.predict(test_generator)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true = test_generator.labels
    
    accuracy = accuracy_score(y_true, y_pred_class)
    report = classification_report(y_true, y_pred_class)
    matrix = confusion_matrix(y_true, y_pred_class)
    
    return accuracy, report, matrix


def save_model(model, path):
    model.save(path)


def load_model(path):
    return tf.keras.models.load_model(path)


def plot_confusion_matrix(matrix):
    plt.imshow(matrix, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()


def plot_classification_report(report):
    print(report)
    
    