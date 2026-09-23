import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, clean_dataframe

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def model_candidates():
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=1,
            random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=6,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        ),
    }


def train(data_path: Path, output_dir: Path):
    raw = pd.read_csv(data_path)
    cleaned = clean_dataframe(raw)
    cleaned = cleaned.dropna(subset=["Price"])
    X = cleaned[MODEL_FEATURES]
    y = cleaned["Price"].clip(lower=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    feature_importances = {}

    for name, regressor in model_candidates().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", TransformedTargetRegressor(
                regressor=regressor, func=np.log1p, inverse_func=np.expm1
            )),
        ])
        pipeline.fit(X_train, y_train)
        predictions = np.maximum(pipeline.predict(X_test), 0)

        # Metrics
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
        r2 = float(r2_score(y_test, predictions))
        mape = float(mean_absolute_percentage_error(y_test, predictions))

        metrics[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE": round(mape * 100, 2),
        }

        # Feature Importance Extraction
        try:
            prep = pipeline.named_steps["preprocessor"]
            feature_names = prep.get_feature_names_out()
            inner_model = pipeline.named_steps["regressor"].regressor_
            if hasattr(inner_model, "feature_importances_"):
                importances = inner_model.feature_importances_
                # Clean up column names for visual clarity
                clean_names = [
                    f.replace("numeric__", "").replace("categorical__", "") for f in feature_names
                ]
                top_idx = np.argsort(importances)[::-1][:10]
                feature_importances[name] = [
                    {"feature": clean_names[i], "importance": round(float(importances[i]), 4)}
                    for i in top_idx
                ]
        except Exception as e:
            print(f"Could not extract feature importance for {name}: {e}")

        model_path = output_dir / f"{name}.joblib"
        joblib.dump(pipeline, model_path, compress=("gzip", 3))
        joblib.load(model_path)
        print(f"{name}: {metrics[name]}")

    best_model = max(metrics, key=lambda name: metrics[name]["R2"])
    metadata = {
        "dataset_rows": int(len(cleaned)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "target_unit": "INR lakh",
        "best_model": best_model,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "options": {
            "brands": sorted(cleaned["Brand"].dropna().unique().tolist()),
            "locations": sorted(cleaned["Location"].dropna().unique().tolist()),
            "fuel_types": sorted(cleaned["Fuel_Type"].dropna().unique().tolist()),
            "transmissions": sorted(cleaned["Transmission"].dropna().unique().tolist()),
            "owner_types": sorted(cleaned["Owner_Type"].dropna().unique().tolist()),
        },
    }
    (output_dir.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train used-car price models")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "models")
    args = parser.parse_args()
    result = train(args.data, args.output)
    print(f"Best model: {result['best_model']} with R²: {result['metrics'][result['best_model']]['R2']}")

