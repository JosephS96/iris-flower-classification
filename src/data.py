# src/data.py
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load and split the Iris dataset into train and test sets.

    Returns:
        X_train, X_test, y_train, y_test, class_names
    """
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, class_names
