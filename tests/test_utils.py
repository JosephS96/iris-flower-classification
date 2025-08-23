# tests/test_utils.py
import os
import json
from src.utils import save_metrics, set_seed


def test_save_metrics(tmp_path):
    metrics = {"accuracy": 0.95}
    path = tmp_path / "metrics.json"
    save_metrics(metrics, str(path))
    assert os.path.exists(path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["accuracy"] == 0.95


def test_set_seed_reproducibility():
    set_seed(123)
    import random
    vals1 = [random.randint(0, 100) for _ in range(5)]
    set_seed(123)
    vals2 = [random.randint(0, 100) for _ in range(5)]
    assert vals1 == vals2
