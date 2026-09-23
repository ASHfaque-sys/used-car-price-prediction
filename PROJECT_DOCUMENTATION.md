# CarWorth AI · Comprehensive Technical Documentation & Defense Guide

## 1. Executive Summary

**CarWorth AI** is an end-to-end Machine Learning web platform designed to estimate the resale value of used cars in **Indian Rupees (INR Lakhs)**. The application processes raw vehicular data, cleans and standardizes numeric/categorical inputs, applies domain-specific feature engineering, transforms the target variable, and feeds features into trained regression models (**Linear Regression**, **Random Forest**, and **XGBoost**). 

The platform serves predictions via a high-performance **Flask REST API** connected to an ultra-modern, responsive **Glassmorphism Web Interface**.

---

## 2. System Architecture & File Structure

```text
used-car-price-prediction/
├── app.py                    # Flask server, REST API endpoints, validation & model serving
├── train.py                  # ML Pipeline, model retraining, cross-evaluation, artifact export
├── data/
│   └── dataset.csv           # 6,019 raw vehicle listings dataset
├── src/
│   ├── __init__.py
│   └── features.py           # Data cleaning, unit extraction & domain feature engineering
├── artifacts/
│   ├── metadata.json         # Evaluation metrics, feature importances & dropdown options
│   └── models/               # Serialized compressed scikit-learn & XGBoost pipelines
│       ├── linear_regression.joblib
│       ├── random_forest.joblib
│       └── xgboost.joblib
├── templates/
│   └── index.html            # Ultra-sleek dark glassmorphic HTML5 interface
├── static/
│   ├── style.css             # Glassmorphism design system & visual utilities
│   └── app.js                # Frontend controller, async fetchers, presets & tabs
├── tests/
│   └── test_app.py           # Automated test suite (9 integration & unit tests)
├── PROJECT_DOCUMENTATION.md  # Comprehensive technical documentation & Viva Q&A guide
├── README.md                 # Public repository guide & quickstart
└── requirements.txt          # Production dependencies
```

---

## 3. Machine Learning Workflow & Engineering

### 3.1 Dataset Overview
- **Total Records**: 6,019 vehicle listings
- **Training Split**: 4,815 rows (80%)
- **Test Split**: 1,204 rows (20%)
- **Target Unit**: Price in INR Lakhs ($1 \text{ Lakh} = 100,000 \text{ INR}$)

### 3.2 Data Preprocessing & Cleaning (`src/features.py`)
Raw listing data contains textual unit annotations and missing values:
- **String Parsing**: Strips string suffixes such as `18.9 kmpl`, `1498 CC`, and `118 bhp`.
- **Currency Standardization**: Normalizes `New_Price` strings (e.g., converting `1.2 Crore` to `120.0 Lakhs`).
- **Brand Extraction**: Extracts manufacturer title from vehicular names (e.g., `Maruti Swift VDI` -> `Maruti`).
- **Missing Value Handling**: Numeric features use **Median Imputation**; Categorical features use **Most Frequent Imputation**. Categorical features are One-Hot Encoded with `handle_unknown="ignore"`.

### 3.3 Domain Feature Engineering
Four engineered features were added to improve model predictive power:
1. **`Car_Age`**: `2026 - Year` (measures vehicle age in years).
2. **`Km_per_Year`**: `Kilometers_Driven / (Car_Age + 1)` (quantifies annual usage intensity).
3. **`Power_per_CC`**: `Power / (Engine + 1e-5)` (measures engine tuning efficiency and performance density).
4. **`Has_New_Price`**: Binary indicator (`1.0` if `New_Price` is present, else `0.0`).

### 3.4 Target Variable Transformation
Prices in used car listings follow a heavy right-skewed distribution due to luxury sports cars. Models use scikit-learn's `TransformedTargetRegressor` with a logarithmic target transformation:
$$y_{\text{trans}} = \log(1 + y)$$
Inference automatically applies the inverse exponential transformation:
$$\hat{y} = \exp(\hat{y}_{\text{trans}}) - 1$$

---

## 4. Comprehensive Model Evaluation & Bias-Variance Metrics

| Model Algorithm | Training $R^2$ | Testing $R^2$ | MAE (Lakhs) | RMSE (Lakhs) | MAPE (%) | Model Bias | Model Variance | Overall Fit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Regressor** ⭐ | **0.9520** | **0.8890** | **1.2833** | **3.6962** | **16.62%** | **Optimal Low** | **Optimal Low** | **Best Balance (No Overfit)** |
| **Random Forest Regressor** | 0.9650 | 0.8847 | 1.4975 | 3.7661 | 18.28% | Low | Low-Medium | Good Balance |
| **Linear Regression** | 0.8250 | 0.8332 | 1.7870 | 4.5307 | 21.78% | High | Low | Underfitting (High Bias) |

### Key Metric Definitions:
- **$R^2$ Score (Goodness of Fit)**: Explains **88.90%** of total price variation across listings.
- **MAE (Mean Absolute Error)**: Average absolute prediction error is **₹1.28 Lakhs** (~₹128,000).
- **RMSE (Root Mean Squared Error)**: Measures error variance (**3.70 Lakhs**), penalizing large outlier errors on luxury cars.
- **MAPE (Mean Absolute Percentage Error)**: Relative percentage error is **16.62%** across budget to luxury price bands.
- **Regression Precision & Confidence Interval**: Represented by MAE ($\pm 1.28 \text{ Lakhs}$) and displayed $\pm 10\%$ valuation bands.

---

## 5. Top Feature Importances (Value Drivers)

Extracted from the trained **XGBoost** and **Random Forest** models:

1. **Power (bhp)** – Primary price determinant (~20.5% importance in XGBoost, ~64.4% in RF).
2. **Transmission Type (Automatic vs Manual)** – Heavy price premium for Automatics (~19.5% combined).
3. **Engine Displacement (CC)** – Higher displacement correlates directly with luxury status (~5.7%).
4. **Car Age / Year** – Vehicles lose significant value in early years (~5.5%).
5. **Brand / Make** – Premium brands (Land Rover, Mini, BMW, Mercedes) retain higher resale floors.

---

## 6. REST API Specification

### `GET /api/health`
Checks server operational status and available trained models.
- **Response**: `{"status": "ok", "models": ["linear_regression", "random_forest", "xgboost"]}`

### `GET /api/presets`
Returns curated real-world vehicle configurations for quick frontend auto-filling.
- **Response**: Array of 5 car presets (SUV, Sedan, Hatchback, Luxury, Off-Road).

### `GET /api/model-analytics`
Provides model metrics ($R^2$, MAE, RMSE, MAPE) and top 10 feature importances.

### `POST /api/predict`
Calculates estimated resale price for a single vehicle input.
- **Payload**:
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
- **Response**: `{"prediction_lakh": 11.45, "formatted_price": "₹11.45 lakh", "model": "xgboost", ...}`

### `POST /api/predict-all`
Runs inference across **all 3 models** simultaneously for side-by-side comparison.

---

## 7. Frontend User Interface Features

- **Glassmorphism Aesthetic**: Deep space dark workspace (`#090d16`) with translucent glass cards (`backdrop-filter: blur(20px)`), vibrant neon cyan/emerald gradients, and responsive layout.
- **Quick Sample Presets**: 1-click sample selectors for popular models (Hyundai Creta, Honda City, Maruti Swift, BMW 3 Series, Mahindra Thar).
- **Dual Valuation Views**:
  - **Single Estimate**: Animated price counter with a visual ±10% range bar and depreciation breakdown.
  - **Multi-Model Comparison**: Side-by-side card grid highlighting model variance and top performer badge.
  - **Value Drivers**: Visual bar chart displaying top feature importances.
- **Utilities**: One-click summary copy to clipboard, reset form handler, and interactive segmented controls.

---

## 8. Added Enhancements & Improvements Summary

| Category | Original Codebase | Upgraded Implementation |
| :--- | :--- | :--- |
| **Features** | 12 raw features | **16 features** including `Car_Age`, `Km_per_Year`, `Power_per_CC`, `Has_New_Price` |
| **Models** | XGBoost $R^2 = 0.876$ | **XGBoost $R^2 = 0.8890$**, lower MAE (1.28) & MAPE (16.62%) |
| **API Endpoints** | `/api/predict` only | Expanded with `/api/predict-all`, `/api/presets`, `/api/model-analytics` |
| **Frontend UI** | Static 2-column form | **Dark Glassmorphism Interface** with sample presets, multi-model tabs, and feature importance charts |
| **Testing** | 4 basic tests | **9 comprehensive integration & unit tests** covering feature engineering, presets, analytics, & predictions |

---

## 9. Verification & How to Run

### Run Local Server
```bash
python app.py
```
Open `http://127.0.0.1:5000` in any modern web browser.

### Retrain ML Models
```bash
python train.py
```

### Run Automated Tests
```bash
python -c "import tests.test_app as t; t.test_home_page(); t.test_health_endpoint(); t.test_presets_endpoint(); t.test_analytics_endpoint(); t.test_prediction(); t.test_predict_all(); t.test_missing_input_is_rejected(); t.test_unknown_model_is_rejected(); t.test_feature_engineering_cleaner(); print('ALL TESTS PASSED!')"
```

---

## 10. Viva Defense & Technical Interview Q&A Guide

### **Q1 (Easy): What is the primary objective of this project?**
> **Answer:** The goal is to build an end-to-end Machine Learning web application that accurately estimates the resale market value of used cars in India (in Lakhs) based on vehicular specifications, condition, usage history, and brand.

### **Q2 (Easy): What tech stack did you use to build this system?**
> **Answer:**
> - **Language**: Python 3.10+
> - **ML Frameworks**: Scikit-Learn, XGBoost, Pandas, NumPy
> - **Backend Server**: Flask REST API
> - **Frontend Interface**: Modern HTML5, Custom Glassmorphism CSS, Vanilla JavaScript

### **Q3 (Easy): What is the dataset size and split ratio?**
> **Answer:** The dataset contains **6,019 real-world vehicle listings**. We used an **80/20 train-test split**: 4,815 listings for training and 1,204 listings for testing.

### **Q4 (Medium): How do you clean messy string data like `"18.9 kmpl"`, `"1498 CC"`, or `"1.2 Crore"`?**
> **Answer:** We created regular expression parsers in `src/features.py`:
> 1. `_first_number()` extracts numeric values from string units (`"18.9 kmpl"` $\rightarrow$ `18.9`).
> 2. `_price_in_lakhs()` normalizes Indian currency notation (`"1.2 Crore"` $\rightarrow$ `120.0 Lakhs`).

### **Q5 (Medium): How do you handle missing numeric and categorical values?**
> **Answer:** We handle missing data inside scikit-learn `Pipeline` objects using `SimpleImputer`:
> - **Numeric features**: Imputed using **Median Strategy** (robust against outliers).
> - **Categorical features**: Imputed using **Most Frequent Strategy** (mode) before applying `OneHotEncoder`.

### **Q6 (Medium): Why did you perform Feature Engineering, and what features did you create?**
> **Answer:** We engineered **4 domain features** in `src/features.py` to capture key physical relationships:
> 1. **`Car_Age`** $= 2026 - \text{Year}$
> 2. **`Km_per_Year`** $= \frac{\text{Kilometers\_Driven}}{\text{Car\_Age} + 1}$
> 3. **`Power_per_CC`** $= \frac{\text{Power}}{\text{Engine}}$
> 4. **`Has_New_Price`** $= 1.0$ if original price is known, else $0.0$.

### **Q7 (Medium): Which ML models did you train, and which one performed best?**
> **Answer:** We trained Linear Regression, Random Forest, and XGBoost. **XGBoost Regressor** performed best with $R^2 = \mathbf{0.8890}$, MAE $= \mathbf{1.28 \text{ Lakhs}}$, and MAPE $= \mathbf{16.62\%}$.

### **Q8 (Hard): Why did you apply Logarithmic Transformation $\log(1 + y)$ on the target price variable?**
> **Answer:** Used car prices are **heavily right-skewed** (most cars are ₹3–15 Lakhs, but luxury sports cars exceed ₹100 Lakhs). High-price outliers cause Large Loss Gradient explosions during model training. 
> Using `TransformedTargetRegressor` with $y_{\text{transformed}} = \log(1 + y)$ normalizes the target distribution so the loss function treats relative percentage errors equally across cheap and expensive cars.

### **Q9 (Hard): Explain the Bias-Variance Tradeoff among Linear Regression, Random Forest, and XGBoost in your project.**
> **Answer:**
> - **Linear Regression**: Has **High Bias (Underfitting)** because it forces linear assumptions on non-linear price depreciation curves ($R^2 = 0.8332$).
> - **Random Forest**: Reduces **Variance** by averaging 300 decorrelated decision trees ($R^2 = 0.8847$).
> - **XGBoost**: Achieves the **Optimal Low Bias & Low Variance Balance** ($R^2 = 0.8890$) by sequentially correcting previous trees' residuals with `subsample=0.85`, `colsample_bytree=0.85`, and `learning_rate=0.03`.

### **Q10 (Hard): What are the top 3 Feature Importances (Value Drivers) in your best model?**
> **Answer:**
> 1. **`Power (bhp)`** (~20.5% weight) – Engine horsepower is the strongest value driver.
> 2. **`Transmission Type`** (~19.5% weight) – Automatic vehicles command a steep price premium over Manual.
> 3. **`Engine Displacement (CC)`** (~5.7% weight) – Engine displacement directly correlates with luxury status.

### **Q11 (Hard): How does your pipeline handle unseen categories (e.g. a new city or car brand) at test time?**
> **Answer:** Our `OneHotEncoder` uses `handle_unknown="ignore"`. Unseen categories are encoded as vector zeros (`0`) without breaking inference.

### **Q12 (Hard): What is the difference between $R^2$, MAE, RMSE, and MAPE?**
> **Answer:**
> - **$R^2$ (Goodness of Fit)**: Proportion of variance explained (**0.8890**).
> - **MAE (Mean Absolute Error)**: Average absolute magnitude of error (**1.28 Lakhs**).
> - **RMSE (Root Mean Squared Error)**: Penalizes large outlier errors (**3.70 Lakhs**).
> - **MAPE (Mean Absolute Percentage Error)**: Relative percentage error (**16.62%**).
