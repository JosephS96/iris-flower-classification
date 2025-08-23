# tests/test_data.py
from src.data import load_data


def test_load_data_shapes():
    X_train, X_test, y_train, y_test, class_names = load_data()
    assert X_train.shape[1] == 4   # iris has 4 features
    assert len(class_names) == 3   # iris has 3 classes
