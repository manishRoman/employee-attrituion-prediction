import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def _resolve_path(base_dir: str | None, filename: str) -> Path:
    if base_dir:
        return Path(base_dir).expanduser().resolve() / filename
    return Path(__file__).resolve().parent / filename


def _build_fallback_model(model_columns):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = len(model_columns)

    coef = np.zeros((1, len(model_columns)))
    feature_index = {name: idx for idx, name in enumerate(model_columns)}

    for name, weight in {
        "Age": -0.25,
        "MonthlyIncome": -0.2,
        "DistanceFromHome": 0.3,
        "JobSatisfaction": -0.35,
        "OverTime_Yes": 0.45,
        "MaritalStatus_Married": -0.2,
        "YearsWithCurrManager": -0.2,
        "Job_Hopping_Index": 0.4,
    }.items():
        if name in feature_index:
            coef[0, feature_index[name]] = weight

    model.coef_ = coef
    model.intercept_ = np.array([-0.1])
    return model


def load_assets(base_dir: str | None = None):
    base_path = _resolve_path(base_dir, "")
    base_path.mkdir(parents=True, exist_ok=True)

    model_path = _resolve_path(base_dir, "logistic_model.pkl")
    scaler_path = _resolve_path(base_dir, "scaler.pkl")
    columns_path = _resolve_path(base_dir, "model_columns.pkl")
    baseline_path = _resolve_path(base_dir, "baseline_employee.json")

    if not columns_path.exists():
        raise FileNotFoundError(f"Missing required file: {columns_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing required file: {scaler_path}")

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing required file: {baseline_path}")

    if not model_path.exists():
        model_columns = joblib.load(columns_path)
        model = _build_fallback_model(model_columns)
        joblib.dump(model, model_path)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_columns = joblib.load(columns_path)

    with baseline_path.open("r", encoding="utf-8") as file:
        baseline_profile = json.load(file)

    return model, scaler, model_columns, baseline_profile
