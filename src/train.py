# src/train.py
import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from model import get_model, evaluate
from data import load_data
from utils import save_metrics, plot_confusion_matrix


def main(model_name: str, test_size: float = 0.2, random_state: int = 42):
    # Load dataset
    X_train, X_test, y_train, y_test, class_names = load_data(test_size=test_size, random_state=random_state)

    # Initialize and train model
    model = get_model(model_name)
    model.fit(X_train, y_train)

    # Evaluate
    acc, cm = evaluate(model, X_test, y_test)

    # Print results
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save results
    os.makedirs("results", exist_ok=True)

    # Save metrics
    metrics_path = os.path.join("results", "metrics.json")
    save_metrics({"model": model_name, "accuracy": acc}, metrics_path)

    # Save confusion matrix as image
    plot_confusion_matrix(cm, class_names, "results/confusion_matrix.png")

    print(f"Results saved in /results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Iris Flower Classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        help="Model to use: logistic_regression | decision_tree | random_forest | svm",
    )
    args = parser.parse_args()
    main(args.model)
