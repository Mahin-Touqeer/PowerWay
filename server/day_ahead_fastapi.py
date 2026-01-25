from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from inference_day import SolarDayAheadPredictor # Import the class we just made

app = FastAPI()

# Load model once on startup
predictor = SolarDayAheadPredictor("day_ahead_model.pth", "day_ahead_scaler.gz")

# Define Data Schemas (Validation)
class HistoryPoint(BaseModel):
    Time: str
    DHI: float
    DNI: float
    GHI: float
    Wind_speed: float
    Humidity: float
    Temperature: float
    PV_production: float

class FuturePoint(BaseModel):
    Time: str
    GHI: float
    Temperature: float
    Wind_speed: float

class ForecastRequest(BaseModel):
    past_data: List[HistoryPoint]
    future_weather: List[FuturePoint]

@app.post("/predict/day-ahead")
def predict_day_ahead(request: ForecastRequest):
    # Convert Pydantic models to list of dicts for Pandas
    past_data = [d.dict() for d in request.past_data]
    future_data = [d.dict() for d in request.future_weather]
    
    # Run Prediction
    try:
        results = predictor.predict(past_data, future_data)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}