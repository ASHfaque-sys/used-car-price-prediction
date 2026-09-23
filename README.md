# CarWorth AI · Used Car Price Prediction Platform

A production-ready Machine Learning web application that predicts a used car's resale value in **Indian Rupees (INR Lakhs)**. Built with Flask, Scikit-Learn, and XGBoost, featuring advanced domain feature engineering, log-transformed target regression, multi-model evaluation, and a sleek **Dark Glassmorphic Web Interface**.

Detailed technical specifications, ML pipelines, and architecture are documented in [PROJECT_DOCUMENTATION.md](file:///c:/Users/ashfaque%20ali/OneDrive/Documents/used-car-price-prediction/PROJECT_DOCUMENTATION.md).

---

## Highlights & Features

- **Domain Feature Engineering**: Automatically computes `Car_Age`, `Km_per_Year`, `Power_per_CC`, and `Has_New_Price` flags to boost model accuracy.
- **Data Normalization & Cleaning**: Cleans text string units (`kmpl`, `CC`, `bhp`, Lakhs/Crores) and imputes missing numeric & categorical values.
- **Log-Transformed Target Regression**: Uses $\log(1+y)$ target scaling inside scikit-learn's `TransformedTargetRegressor` to handle high-value luxury outliers smoothly.
- **Multi-Model Benchmark**: Retrains and evaluates **Linear Regression**, **Random Forest**, and **XGBoost Regressor** on a fixed 80/20 train-test split.
- **Top Model Performance**: **XGBoost Regressor** achieves **$R^2 = 0.8890$**, **$\text{MAE} = 1.28 \text{ Lakhs}$**, and **$\text{MAPE} = 16.62\%$**.
- **Dark Glassmorphism Interface**: Modern, responsive dark UI featuring Quick Sample Car Presets, Single Valuation Estimate, Side-by-Side Multi-Model Comparison, and Top Feature Importance Value Drivers.
- **Expanded REST API**: Full JSON endpoints (`/api/predict`, `/api/predict-all`, `/api/presets`, `/api/model-analytics`, `/api/health`).
- **Comprehensive Unit Testing**: 9 test suites verifying backend endpoints, feature engineering pipelines, and error validation.

---

## Project Structure

```text
used-car-price-prediction/
├── app.py                    # Flask application & REST API server
├── train.py                  # Model training, feature engineering & artifact exporter
├── data/
│   └── dataset.csv           # 6,019 vehicle listings dataset
├── src/
│   ├── __init__.py
│   └── features.py           # Data cleaning, unit extraction & domain feature engineering
├── artifacts/
│   ├── metadata.json         # Evaluation metrics, feature importances & options
│   └── models/               # Serialized compressed ML model pipelines (.joblib)
│       ├── linear_regression.joblib
│       ├── random_forest.joblib
│       └── xgboost.joblib
├── templates/
│   └── index.html            # Dark glassmorphic web interface
├── static/
│   ├── style.css             # Visual design system & glassmorphism utilities
│   └── app.js                # Frontend controller & interactive fetchers
├── tests/
│   └── test_app.py           # Automated test suite
├── PROJECT_DOCUMENTATION.md  # Comprehensive technical documentation & Viva Q&A guide
├── README.md                 # Project README
└── requirements.txt          # Production dependencies
```

---

## Comprehensive Model Evaluation & Bias-Variance Metrics

| Model Algorithm | Training $R^2$ | Testing $R^2$ | MAE (Lakhs) | RMSE (Lakhs) | MAPE (%) | Model Bias | Model Variance | Overall Fit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Regressor** ⭐ | **0.9520** | **0.8890** | **1.2833** | **3.6962** | **16.62%** | **Optimal Low** | **Optimal Low** | **Best Balance (No Overfit)** |
| **Random Forest Regressor** | 0.9650 | 0.8847 | 1.4975 | 3.7661 | 18.28% | Low | Low-Medium | Good Balance |
| **Linear Regression** | 0.8250 | 0.8332 | 1.7870 | 4.5307 | 21.78% | High | Low | Underfitting (High Bias) |

---

## Quickstart & Local Setup

### 1. Install Dependencies
Python 3.10–3.13 is recommended.

```bash
pip install -r requirements.txt
```

### 2. Run Web Application
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser. Pre-trained model artifacts are included.

### 3. Retrain Models
To retrain models and regenerate `metadata.json` and `.joblib` pipelines:
```bash
python train.py
```

### 4. Run Test Suite
```bash
python -c "import tests.test_app as t; t.test_home_page(); t.test_health_endpoint(); t.test_presets_endpoint(); t.test_analytics_endpoint(); t.test_prediction(); t.test_predict_all(); t.test_missing_input_is_rejected(); t.test_unknown_model_is_rejected(); t.test_feature_engineering_cleaner(); print('ALL 9 TESTS PASSED!')"
```

---

## REST API Overview

### `POST /api/predict`
Calculates estimated resale price for a single vehicle input.

```json
{
  "brand": "Hyundai",
  "location": "Mumbai",
  "year": 2020,
  "kilometers_driven": 32000,
  "fuel_type": "Diesel",
  "transmission": "Manual",
  "owner_type": "First",
  "mileage": 21.4,
  "engine": 1493,
  "power": 113.4,
  "seats": 5,
  "new_price": 13.5,
  "model": "xgboost"
}
```

### `POST /api/predict-all`
Runs inference across **Linear Regression**, **Random Forest**, and **XGBoost** simultaneously and returns an ensemble average.

### `GET /api/presets`
Returns curated real-world vehicle sample presets (Hyundai Creta, Honda City, Maruti Swift, BMW 3 Series, Mahindra Thar).

### `GET /api/model-analytics`
Returns evaluation metrics ($R^2$, MAE, RMSE, MAPE) and top feature importances.

---

## Viva & Technical Defense Highlights

- **Dataset Split**: 6,019 total listings split into 4,815 training rows (80%) and 1,204 testing rows (20%).
- **Target Transformation**: Uses $\log(1+y)$ log-scaling to prevent high-value luxury car outliers from distorting loss gradients.
- **Top Value Drivers**: **Power (bhp)** (~20.5%), **Transmission Type** (~19.5%), and **Engine (CC)** (~5.7%).
- **Full Q&A Guide**: Read [PROJECT_DOCUMENTATION.md Section 10](file:///c:/Users/ashfaque%20ali/OneDrive/Documents/used-car-price-prediction/PROJECT_DOCUMENTATION.md#10-viva-defense--technical-interview-qa-guide) for 12 easy-to-hard interview questions and answers.

---

## License & Disclaimer
This is an educational estimator trained on historical vehicle listing data. Actual market prices may vary based on live demand, physical vehicle condition, service history, and regional variations.
