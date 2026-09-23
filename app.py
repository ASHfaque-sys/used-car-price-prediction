import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from src.features import clean_dataframe, form_to_frame

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_NAMES = {"linear_regression", "random_forest", "xgboost"}

app = Flask(__name__)


def metadata():
    path = ARTIFACTS / "metadata.json"
    if not path.exists():
        raise RuntimeError("Models are not trained. Run: python train.py")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=3)
def load_model(name):
    if name not in MODEL_NAMES:
        raise ValueError("Unknown model")
    path = ARTIFACTS / "models" / f"{name}.joblib"
    if not path.exists():
        raise RuntimeError("Model file is missing. Run: python train.py")
    return joblib.load(path)


def validate(payload):
    required = [
        "brand", "location", "year", "kilometers_driven", "fuel_type",
        "transmission", "owner_type", "mileage", "engine", "power", "seats",
    ]
    missing = [key for key in required if payload.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Please provide: {', '.join(missing)}")

    numeric_ranges = {
        "year": (1990, 2030), "kilometers_driven": (0, 2_000_000),
        "mileage": (0, 100), "engine": (100, 10000), "power": (1, 2000),
        "seats": (1, 20), "new_price": (0, 1000),
    }
    for field, (low, high) in numeric_ranges.items():
        if field == "new_price" and payload.get(field) in (None, ""):
            continue
        try:
            value = float(payload[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field.replace('_', ' ').title()} must be a number")
        if not low <= value <= high:
            raise ValueError(f"{field.replace('_', ' ').title()} must be between {low} and {high}")


@app.get("/")
def index():
    data = metadata()
    return render_template("index.html", metadata=data)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "models": sorted(MODEL_NAMES)})


@app.get("/api/presets")
def get_presets():
    presets = [
        {
            "id": "creta_2020",
            "title": "2020 Hyundai Creta",
            "subtitle": "Popular Diesel SUV · Mumbai",
            "badge": "Compact SUV",
            "data": {
                "brand": "Hyundai", "location": "Mumbai", "year": 2020,
                "kilometers_driven": 32000, "fuel_type": "Diesel",
                "transmission": "Manual", "owner_type": "First",
                "mileage": 21.4, "engine": 1493, "power": 113.4,
                "seats": 5, "new_price": 13.5
            }
        },
        {
            "id": "city_2018",
            "title": "2018 Honda City",
            "subtitle": "Automatic Sedan · Delhi",
            "badge": "Sedan",
            "data": {
                "brand": "Honda", "location": "Delhi", "year": 2018,
                "kilometers_driven": 45000, "fuel_type": "Petrol",
                "transmission": "Automatic", "owner_type": "First",
                "mileage": 18.0, "engine": 1497, "power": 117.3,
                "seats": 5, "new_price": 12.0
            }
        },
        {
            "id": "swift_2017",
            "title": "2017 Maruti Swift",
            "subtitle": "Economy Hatchback · Bangalore",
            "badge": "Hatchback",
            "data": {
                "brand": "Maruti", "location": "Bangalore", "year": 2017,
                "kilometers_driven": 55000, "fuel_type": "Petrol",
                "transmission": "Manual", "owner_type": "First",
                "mileage": 20.4, "engine": 1197, "power": 81.8,
                "seats": 5, "new_price": 7.2
            }
        },
        {
            "id": "bmw_3series_2019",
            "title": "2019 BMW 3 Series",
            "subtitle": "Luxury Sedan · Hyderabad",
            "badge": "Luxury",
            "data": {
                "brand": "Bmw", "location": "Hyderabad", "year": 2019,
                "kilometers_driven": 28000, "fuel_type": "Diesel",
                "transmission": "Automatic", "owner_type": "First",
                "mileage": 20.37, "engine": 1995, "power": 190.0,
                "seats": 5, "new_price": 48.5
            }
        },
        {
            "id": "mahindra_thar_2021",
            "title": "2021 Mahindra Thar",
            "subtitle": "4x4 Off-roader · Pune",
            "badge": "Off-Road",
            "data": {
                "brand": "Mahindra", "location": "Pune", "year": 2021,
                "kilometers_driven": 22000, "fuel_type": "Diesel",
                "transmission": "Manual", "owner_type": "First",
                "mileage": 15.0, "engine": 2184, "power": 130.0,
                "seats": 4, "new_price": 16.5
            }
        }
    ]
    return jsonify({"presets": presets})


@app.get("/api/model-analytics")
def get_analytics():
    try:
        data = metadata()
        return jsonify({
            "metrics": data.get("metrics", {}),
            "best_model": data.get("best_model", "random_forest"),
            "feature_importances": data.get("feature_importances", {}),
            "dataset_rows": data.get("dataset_rows", 0)
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        validate(payload)
        selected = payload.get("model") or metadata()["best_model"]
        if selected not in MODEL_NAMES:
            raise ValueError("Select a valid model")
        raw = form_to_frame(payload)
        features = clean_dataframe(raw, require_target=False)
        prediction = max(float(load_model(selected).predict(features)[0]), 0)
        return jsonify({
            "prediction_lakh": round(prediction, 2),
            "formatted_price": f"₹{prediction:.2f} lakh",
            "model": selected,
            "confidence_note": "Estimate based on historical listings; actual market price may differ.",
        })
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the submitted values."}), 500


@app.post("/api/predict-all")
def predict_all():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        validate(payload)
        raw = form_to_frame(payload)
        features = clean_dataframe(raw, require_target=False)

        meta = metadata()
        results = {}
        prices = []

        for name in sorted(MODEL_NAMES):
            pred = max(float(load_model(name).predict(features)[0]), 0)
            pred_round = round(pred, 2)
            prices.append(pred)
            results[name] = {
                "prediction_lakh": pred_round,
                "formatted_price": f"₹{pred_round:.2f} lakh",
                "range_min": round(max(0, pred * 0.9), 2),
                "range_max": round(pred * 1.1, 2),
                "is_best": (name == meta.get("best_model"))
            }

        avg_price = round(float(np.mean(prices)), 2)

        return jsonify({
            "predictions": results,
            "best_model": meta.get("best_model"),
            "average_lakh": avg_price,
            "formatted_average": f"₹{avg_price:.2f} lakh",
            "confidence_note": "Comparing predictions across Linear Regression, Random Forest, and XGBoost models."
        })
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    except Exception:
        app.logger.exception("Predict-all failed")
        return jsonify({"error": "Prediction across models failed. Check submitted values."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
