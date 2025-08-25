# src/model.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def get_model(name: str):
    """Return a scikit-learn model based on the name."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=200),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "svm": SVC(kernel="linear", probability=True),
        "dummy": DummyClassifier(strategy="most_frequent")
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name]


def evaluate(model, X_test, y_test):
    """Return accuracy and confusion matrix for predictions."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm
