# src/utils.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os


def save_metrics(metrics: dict, path: str = "results/metrics.json"):
    """Save metrics as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def plot_confusion_matrix(cm, class_names, path="results/confusion_matrix.png"):
    """Plot and save confusion matrix as an image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def set_seed(seed=42):
    """Ensure reproducibility across numpy and random."""
    np.random.seed(seed)
    random.seed(seed)
