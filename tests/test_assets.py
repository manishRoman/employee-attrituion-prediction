import json
import os
import tempfile

from app_assets import load_assets


def test_load_assets_creates_missing_files(tmp_path):
    assets = load_assets(base_dir=str(tmp_path))

    model, scaler, model_columns, baseline_profile = assets

    assert model is not None
    assert scaler is not None
    assert isinstance(model_columns, list) and len(model_columns) > 0
    assert isinstance(baseline_profile, dict) and baseline_profile

    assert os.path.exists(os.path.join(tmp_path, 'logistic_model.pkl'))
    assert os.path.exists(os.path.join(tmp_path, 'scaler.pkl'))
    assert os.path.exists(os.path.join(tmp_path, 'model_columns.pkl'))
    assert os.path.exists(os.path.join(tmp_path, 'baseline_employee.json'))
