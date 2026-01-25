from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -----------------------------
# ✅ Load model + feature columns
# -----------------------------
model = joblib.load("xgb_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# -----------------------------
# ✅ Feature Engineering
# -----------------------------
def create_features(df):
    df = df.copy()

    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["year"] = df.index.year
    df["season"] = df["month"] % 12 // 3 + 1
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day
    df["weekofyear"] = df.index.isocalendar().week.astype(int)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = (df["dayofmonth"] == 1).astype(int)
    df["is_month_end"] = (df["dayofmonth"] == df.index.days_in_month).astype(int)

    df["is_working_day"] = df["dayofweek"].isin([0, 1, 2, 3, 4]).astype(int)
    df["is_business_hours"] = df["hour"].between(9, 17).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 12, 18]).astype(int)

    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["minute_of_week"] = (df["dayofweek"] * 24 * 60) + df["minute_of_day"]

    return df

# -----------------------------
# ✅ Prediction function
# -----------------------------
def predict_load_api(payload: dict):
    df = pd.DataFrame([payload])

    if "Timestamp" not in df.columns:
        raise ValueError("Payload must contain 'Timestamp'")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")

    df = create_features(df)

    # ✅ align with training features
    df = df.reindex(columns=feature_cols, fill_value=0)

    pred = model.predict(df)[0]
    return float(pred)

# -----------------------------
# ✅ Request Schema
# -----------------------------
class PredictRequest(BaseModel):
    Timestamp: str

    Temperature: float
    Humidity: float
    WindSpeed: float
    Rainfall: float
    SolarIrradiance: float

    GDP: float
    PerCapitaEnergyUse: float
    ElectricityPrice: float

    DayOfWeek: int
    HourOfDay: int
    Month: int
    PublicEvent: int

    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float   

# -----------------------------
# ✅ FastAPI App
# -----------------------------
app = FastAPI(title="Load Forecast API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "message": "✅ API is running"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict 1 step ahead.
    """
    payload = {
        "Timestamp": req.Timestamp,
        "Temperature (°C)": req.Temperature,
        "Humidity (%)": req.Humidity,
        "Wind Speed (m/s)": req.WindSpeed,
        "Rainfall (mm)": req.Rainfall,
        "Solar Irradiance (W/m²)": req.SolarIrradiance,
        "GDP (LKR)": 925,
        "Per Capita Energy Use (kWh)": req.PerCapitaEnergyUse,
        "Electricity Price (LKR/kWh)": req.ElectricityPrice,
        "Day of Week": req.DayOfWeek,
        "Hour of Day": req.HourOfDay,
        "Month": req.Month,
        "Public Event": req.PublicEvent,
        "lag_1": req.lag_1,
        "lag_2": req.lag_2,
        "lag_3": req.lag_3,
        "lag_4": req.lag_4,
        "lag_5": req.lag_5,
    }

    out = predict_load_api(payload)
    return {"predicted_load_kw": out}
