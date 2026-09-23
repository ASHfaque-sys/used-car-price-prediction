import sys
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from src.features import clean_dataframe, form_to_frame


def valid_payload():
    return {
        "brand": "Hyundai", "location": "Mumbai", "year": 2018,
        "kilometers_driven": 45000, "fuel_type": "Diesel",
        "transmission": "Manual", "owner_type": "First", "mileage": 18.5,
        "engine": 1498, "power": 118, "seats": 5, "new_price": 14.5,
        "model": "xgboost",
    }


def test_home_page():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"CarWorth" in response.data


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "xgboost" in data["models"]


def test_presets_endpoint():
    client = app.test_client()
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.get_json()
    assert "presets" in data
    assert len(data["presets"]) >= 3


def test_analytics_endpoint():
    client = app.test_client()
    response = client.get("/api/model-analytics")
    assert response.status_code == 200
    data = response.get_json()
    assert "metrics" in data
    assert "best_model" in data
    assert "feature_importances" in data


def test_prediction():
    client = app.test_client()
    response = client.post("/api/predict", json=valid_payload())
    body = response.get_json()
    assert response.status_code == 200
    assert body["prediction_lakh"] >= 0
    assert body["model"] == "xgboost"


def test_predict_all():
    client = app.test_client()
    response = client.post("/api/predict-all", json=valid_payload())
    body = response.get_json()
    assert response.status_code == 200
    assert "predictions" in body
    assert "linear_regression" in body["predictions"]
    assert "random_forest" in body["predictions"]
    assert "xgboost" in body["predictions"]
    assert body["average_lakh"] > 0


def test_missing_input_is_rejected():
    client = app.test_client()
    response = client.post("/api/predict", json={"brand": "Honda"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_unknown_model_is_rejected():
    client = app.test_client()
    payload = valid_payload()
    payload["model"] = "not_a_model"
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_feature_engineering_cleaner():
    raw_frame = form_to_frame(valid_payload())
    cleaned = clean_dataframe(raw_frame, require_target=False)
    assert "Car_Age" in cleaned.columns
    assert "Km_per_Year" in cleaned.columns
    assert "Power_per_CC" in cleaned.columns
    assert "Has_New_Price" in cleaned.columns
    assert cleaned["Car_Age"].iloc[0] == (2026 - 2018)

