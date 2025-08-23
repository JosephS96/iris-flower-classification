# tests/test_model.py
import pytest
from src.model import get_model, evaluate
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_get_model_valid():
    model = get_model("random_forest")
    assert model is not None


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model("not_a_model")


def test_evaluate_runs():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    model = get_model("decision_tree")
    model.fit(X_train, y_train)
    acc, cm = evaluate(model, X_test, y_test)
    assert 0 <= acc <= 1
    assert cm.shape[0] == cm.shape[1] == len(iris.target_names)
