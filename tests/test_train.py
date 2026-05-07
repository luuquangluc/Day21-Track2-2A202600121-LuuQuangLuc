import pytest
import os
import json
import numpy as np
import pandas as pd
from src.train import train

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]

def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.
    """
    rng = np.random.default_rng(0)
    n = 200

    # TODO 2.10.1: Tao mang X co kich thuoc (n, len(FEATURE_NAMES))
    X = rng.random((n, len(FEATURE_NAMES)))

    # TODO 2.10.2: Tao mang y co n phan tu [0, 3)
    y = rng.integers(0, 3, size=n)

    # TODO 2.10.3: Tao DataFrame
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # TODO 2.10.4: Luu file
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"

    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return str(train_path), str(eval_path)

def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc trong khoang [0, 1]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    acc = train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    metrics_file = "outputs/metrics.json"
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists(metrics_file)
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics

def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    model_file = "models/model.pkl"
    if os.path.exists(model_file):
        os.remove(model_file)

    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists(model_file)
