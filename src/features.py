import re

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "Year",
    "Car_Age",
    "Kilometers_Driven",
    "Km_per_Year",
    "Mileage",
    "Engine",
    "Power",
    "Power_per_CC",
    "Seats",
    "New_Price",
    "Has_New_Price",
]

CATEGORICAL_FEATURES = [
    "Brand",
    "Location",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _first_number(value):
    """Extract the first numeric value from values such as '18.9 kmpl'."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def _price_in_lakhs(value):
    """Normalize Indian new-car price strings to lakhs."""
    number = _first_number(value)
    if pd.isna(number):
        return np.nan
    return number * 100 if "crore" in str(value).lower() else number


def clean_dataframe(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Convert the raw Kaggle-style data into model-ready columns with engineered features."""
    df = frame.copy()
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

    required = {
        "Name", "Location", "Year", "Kilometers_Driven", "Fuel_Type",
        "Transmission", "Owner_Type", "Mileage", "Engine", "Power", "Seats",
        "New_Price",
    }
    if require_target:
        required.add("Price")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["Brand"] = df["Name"].astype(str).str.strip().str.split().str[0].str.title()
    for col in ["Mileage", "Engine", "Power"]:
        df[col] = df[col].map(_first_number)
    df["New_Price"] = df["New_Price"].map(_price_in_lakhs)
    for col in ["Year", "Kilometers_Driven", "Seats"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Domain Feature Engineering
    current_year = 2026
    df["Car_Age"] = current_year - df["Year"]
    df["Car_Age"] = df["Car_Age"].clip(lower=0)
    df["Km_per_Year"] = df["Kilometers_Driven"] / (df["Car_Age"] + 1)
    df["Power_per_CC"] = df["Power"] / (df["Engine"] + 1e-5)
    df["Has_New_Price"] = df["New_Price"].notna().astype(float)

    selected = MODEL_FEATURES + (["Price"] if require_target else [])
    return df[selected]


def form_to_frame(values: dict) -> pd.DataFrame:
    """Create a one-row raw frame compatible with clean_dataframe."""
    return pd.DataFrame([{
        "Name": values.get("brand", "Unknown"),
        "Location": values.get("location"),
        "Year": values.get("year"),
        "Kilometers_Driven": values.get("kilometers_driven"),
        "Fuel_Type": values.get("fuel_type"),
        "Transmission": values.get("transmission"),
        "Owner_Type": values.get("owner_type"),
        "Mileage": values.get("mileage"),
        "Engine": values.get("engine"),
        "Power": values.get("power"),
        "Seats": values.get("seats"),
        "New_Price": values.get("new_price"),
    }])

