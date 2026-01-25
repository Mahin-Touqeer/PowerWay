from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
import joblib
from day_wind_inference import WindDayAheadPredictor

from inference import SolarInference   # your LSTM solar model
from wind_inference import WindInference

# =====================================================
# 🔹 WIND MODEL (DAY-AHEAD LSTM)
# =====================================================
wind_predictor = WindDayAheadPredictor(
    model_path="day_ahead_model_wind.pth",
    scaler_path="day_ahead_scaler_wind.gz"
)

# ---- Wind ----
class WindPastPoint(BaseModel):
    Time: str
    Wind_speed: float
    Temperature: float
    Humidity: float

class WindFuturePoint(BaseModel):
    Time: str
    Wind_speed: float
    Temperature: float

class WindPredictRequest(BaseModel):
    past_5min_data: List[WindPastPoint]     # last 24h (288 points)
    future_hourly_data: List[WindFuturePoint]  # next 24h (24 points)

# =====================================================
# App
# =====================================================
app = FastAPI(title="Energy Forecast API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# 🔹 SOLAR MODEL (LSTM)
# =====================================================
solar_predictor = SolarInference(
    model_path="best_solar_model.pth",
    scaler_path="scaler.gz",
    ghi_map_path="max_ghi_map.gz"
)

wind_predictor = WindInference(
    model_path="best_wind_model.pth",
    scaler_path="scaler_wind.gz",
    # ghi_map_path="max_ghi_map.gz"
)


# =====================================================
# 🔹 LOAD MODEL (XGBoost)
# =====================================================
load_model = joblib.load("xgb_model.pkl")
load_feature_cols = joblib.load("feature_cols.pkl")

# =====================================================
# 🔹 Load Feature Engineering (from friend)
# =====================================================
def create_load_features(df):
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

def predict_load(payload: dict) -> float:
    df = pd.DataFrame([payload])

    if "Timestamp" not in df.columns:
        raise ValueError("Payload must contain Timestamp")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")

    df = create_load_features(df)

    # Align with training schema
    df = df.reindex(columns=load_feature_cols, fill_value=0)

    return float(load_model.predict(df)[0])

# =====================================================
# 🔹 Schemas
# =====================================================

# ---- Solar ----
class SolarDataPoint(BaseModel):
    timestamp: str
    GHI: float
    DNI: float
    DHI: float
    temp: float
    windSpeed: float
    humidity: float
    season: int
    day: int

class SolarPredictRequest(BaseModel):
    data: List[SolarDataPoint]

# ---- Load ----
class LoadPredictRequest(BaseModel):
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

# =====================================================
# 🔹 Routes
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": ["solar_lstm", "load_xgboost"]
    }

# ---------------- SOLAR ----------------
@app.post("/predict/solar")
def predict_solar(req: SolarPredictRequest):
    try:
        normalized = [
            {
                "Time": r.timestamp,
                "GHI": r.GHI,
                "DNI": r.DNI,
                "DHI": r.DHI,
                "Temperature": r.temp,
                "Wind_speed": r.windSpeed,
                "Humidity": r.humidity,
                "Season": r.season,
                "Day_of_the_week": r.day,
            }
            for r in req.data
        ]

        pred = solar_predictor.predict(normalized)

        return {
            "prediction": pred,
            "type": "solar_generation"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/wind")
def predict_solar(req: SolarPredictRequest):
    try:
        normalized = [
            {
                "Time": r.timestamp,
                # "GHI": r.GHI,
                # "DNI": r.DNI,
                # "DHI": r.DHI,
                "Temperature": r.temp,
                "Wind_speed": r.windSpeed,
                "Humidity": r.humidity,
                "Season": r.season,
                "Day_of_the_week": r.day,
            }
            for r in req.data
        ]

        pred = wind_predictor.predict(normalized)

        return {
            "prediction": pred,
            "type": "solar_generation"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------------- WIND ----------------
@app.post("/predict/wind/day-ahead")
def predict_wind_day_ahead(req: WindPredictRequest):
    try:
        print("hiii")
        print(req.future_hourly_data)
        # print(futu)
        result = wind_predictor.predict(
            past_json=[p.dict() for p in req.past_5min_data],
            future_json=[f.dict() for f in req.future_hourly_data],
        )

        return {
            "type": "wind_generation",
            "horizon": "24_hours",
            "hourly_forecast_mw": result["hourly_forecast"],
            "five_min_forecast_mw": result["high_res_5min_forecast"],
            "total_day_mwh": result["total_day_mwh"]
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))


# ---------------- LOAD ----------------
@app.post("/predict/load")
def predict_load_api(req: LoadPredictRequest):
    try:
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
        print(payload)

        pred = predict_load(payload)

        return {
            "prediction": pred,
            "type": "load_kw"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
