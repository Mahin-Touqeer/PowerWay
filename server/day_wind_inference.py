import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import interp1d
import os

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cpu") # Use CPU for inference (cheaper/easier for web)
MAX_CAPACITY = 3000          # MW (Change this to your actual plant capacity)

# Feature Lists (Must match your training EXACTLY)
# Note: 'Wind_production' is the target here
PAST_FEATURE_COLS = [
    'Wind_speed', 'Humidity', 'Temperature', 
    'season_sin', 'season_cos', 'weekday_sin', 'weekday_cos', 
    'Wind_production' 
]

FUTURE_DRIVER_COLS = [
    'Temperature', 'Wind_speed', 
    'season_sin', 'season_cos', 'weekday_sin', 'weekday_cos'
]

# --- 2. MODEL ARCHITECTURE (Must match training) ---
class DayAheadNet(nn.Module):
    def __init__(self, past_features, future_features, hidden_size=64):
        super(DayAheadNet, self).__init__()
        
        # Encoder (Processes Past History)
        self.lstm = nn.LSTM(input_size=past_features, hidden_size=hidden_size, batch_first=True)
        
        # Processor (Processes Future Weather)
        self.future_fc = nn.Linear(24 * future_features, hidden_size)
        
        # Combiner
        self.combine_fc = nn.Linear(hidden_size + hidden_size, hidden_size)
        
        # Decoder (Output 24 hours of generation)
        self.output_fc = nn.Linear(hidden_size, 24) 
        
        # Note: In your notebook, you used MinMaxScaler for Wind_production target too.
        # So the output is likely 0-1 (Sigmoid) or just Linear if you scaled differently.
        # Assuming Sigmoid based on previous solar discussions, but if you used linear 
        # activation in training, remove the sigmoid below.
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, past, future):
        # past: (Batch, 24, past_features)
        # future: (Batch, 24, future_features)
        
        # 1. Encode Past
        _, (hidden, _) = self.lstm(past)
        context = hidden[-1] # (Batch, hidden_size)
        
        # 2. Process Future
        # Flatten future: (Batch, 24*future_features)
        future_flat = future.reshape(future.shape[0], -1) 
        future_context = torch.relu(self.future_fc(future_flat))
        
        # 3. Combine
        combined = torch.cat((context, future_context), dim=1)
        combined = torch.relu(self.combine_fc(combined))
        
        # 4. Predict
        out = self.output_fc(combined)
        return self.sigmoid(out)

# --- 3. HELPER: FEATURE ENGINEERING ---
def calculate_features(df):
    """Generates cyclic time features."""
    # Ensure Time is datetime
    if not np.issubdtype(df['Time'].dtype, np.datetime64):
        df['Time'] = pd.to_datetime(df['Time'])
        
    # Seasonality / Cyclical
    df["season_sin"] = np.sin(2 * np.pi * df["Time"].dt.quarter / 4)
    df["season_cos"] = np.cos(2 * np.pi * df["Time"].dt.quarter / 4)
    df["weekday_sin"] = np.sin(2 * np.pi * df["Time"].dt.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["Time"].dt.dayofweek / 7)
        
    return df

# --- 4. PREDICTOR CLASS ---
class WindDayAheadPredictor:
    def __init__(self, model_path, scaler_path):
        self.scaler = joblib.load(scaler_path)
        # Initialize Model
        self.model = DayAheadNet(
            past_features=len(PAST_FEATURE_COLS), 
            future_features=len(FUTURE_DRIVER_COLS)
        ).to(DEVICE)
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print("✅ Day-Ahead Wind Model Loaded")

    def preprocess_past(self, data_list):
        """
        Handles hourly past history (24h) -> feature engineering -> scaling
        """

        # -------------------------------------------------
        # 1. Load + basic cleanup
        # -------------------------------------------------
        df = pd.DataFrame(data_list)

        if "Time" not in df.columns:
            raise ValueError("Each data point must contain 'Time'")

        df["Time"] = pd.to_datetime(df["Time"])
        df = df.sort_values("Time")

        # -------------------------------------------------
        # 2. Mandatory fallback: Wind production
        # -------------------------------------------------
        if "Wind_production" not in df.columns:
            rated_speed = 12.0  # m/s

            df["Wind_production"] = (
                (df["Wind_speed"] / rated_speed) ** 3 * MAX_CAPACITY
            )

            # physical constraints
            df["Wind_production"] = df["Wind_production"].clip(0, MAX_CAPACITY)
            df.loc[df["Wind_speed"] < 3, "Wind_production"] = 0

            print(
                "⚠️ Warning: Wind_production missing. "
                "Generated theoretical fallback from wind speed."
            )

        # -------------------------------------------------
        # 3. Ensure raw meteorological columns exist
        # -------------------------------------------------
        required_raw_cols = ["Wind_speed", "Humidity", "Temperature"]

        for col in required_raw_cols:
            if col not in df.columns:
                df[col] = 0.0

        # -------------------------------------------------
        # 4. Feature engineering (cyclic time features)
        # -------------------------------------------------
        df = calculate_features(df)

        # -------------------------------------------------
        # 5. Index + numeric-only safety (CRITICAL)
        # -------------------------------------------------
        df = df.set_index("Time")

        # remove any string / object / pandas string dtype
        df = df.select_dtypes(include=["number"])

        # -------------------------------------------------
        # 6. Keep exactly last 24 hourly records
        # -------------------------------------------------
        df_hourly = df.tail(24)

        if len(df_hourly) < 24:
            raise ValueError(
                f"Not enough history. Needed 24h, got {len(df_hourly)}h"
            )

        # -------------------------------------------------
        # 7. Missing data handling (pandas 2.x safe)
        # -------------------------------------------------
        df_hourly = df_hourly.ffill()
        df_hourly = df_hourly.bfill()
        df_hourly = df_hourly.fillna(0)

        # -------------------------------------------------
        # 8. Final schema alignment with training
        # -------------------------------------------------
        for col in PAST_FEATURE_COLS:
            if col not in df_hourly.columns:
                df_hourly[col] = 0.0

        # keep column order EXACTLY as scaler expects
        df_hourly = df_hourly[PAST_FEATURE_COLS]

        # -------------------------------------------------
        # 9. Scaling
        # -------------------------------------------------
        scaled_values = self.scaler.transform(df_hourly.values)

        # -------------------------------------------------
        # 10. Convert to tensor
        # -------------------------------------------------
        return (
            torch.tensor(scaled_values, dtype=torch.float32)
            .unsqueeze(0)
            .to(DEVICE)
        )


    def preprocess_future(self, data_list):
        """
        Handles future forecast (Already Hourly) -> Features -> Scales
        """
        df = pd.DataFrame(data_list)
        df = calculate_features(df)
        
        # Create a dummy DF to satisfy the Scaler's expected shape (PAST_FEATURE_COLS)
        dummy_df = pd.DataFrame(0, index=df.index, columns=PAST_FEATURE_COLS)
        for col in FUTURE_DRIVER_COLS:
            if col in df.columns:
                dummy_df[col] = df[col]
        
        # Scale everything
        scaled_full = self.scaler.transform(dummy_df.values.astype(np.float32))
        
        # Extract ONLY the columns the Future-Branch of the model needs
        # Find indices of future cols in the full list
        future_indices = [PAST_FEATURE_COLS.index(c) for c in FUTURE_DRIVER_COLS]
        scaled_future = scaled_full[:, future_indices]
        
        return torch.tensor(scaled_future, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    def predict(self, past_json, future_json):
        # 1. Preprocess inputs
        past_tensor = self.preprocess_past(past_json)
        future_tensor = self.preprocess_future(future_json)
        
        # 2. Inference
        with torch.no_grad():
            # Output is (1, 24) -> Normalized 0-1
            pred_norm = self.model(past_tensor, future_tensor).numpy().flatten()
            
        # 3. Denormalize
        # Since target was MinMaxScaled (0-1), we multiply by MAX_CAPACITY
        pred_mw = pred_norm * MAX_CAPACITY
        
        # 4. Upscale (Interpolate 24 points -> 288 points for 5-min graph)
        hours_x = np.arange(24)
        minutes_x = np.linspace(0, 23, 24 * 12) # 5-min intervals
        
        f = interp1d(hours_x, pred_mw, kind='cubic')
        pred_5min = f(minutes_x)
        pred_5min = np.clip(pred_5min, 0, None) # No negative power
        
        return {
            "hourly_forecast": pred_mw.tolist(),
            "high_res_5min_forecast": pred_5min.tolist(),
            "total_day_mwh": float(pred_mw.sum())
        }

# --- SELF-TEST BLOCK (Run this file directly to test) ---
if __name__ == "__main__":
    print("🧪 Starting Self-Test for Wind Predictor...")

    # 1. GENERATE MOCK DATA (Simulating missing Wind_production)
    print("   Creating dummy past data (24h history) WITHOUT Wind_production...")
    past_dates = pd.date_range(end=pd.Timestamp.now(), periods=288, freq='5min')
    mock_past_json = []
    for date in past_dates:
        mock_past_json.append({
            "Time": str(date),
            "Wind_speed": np.random.uniform(2, 15),
            "Humidity": np.random.uniform(30, 80),
            "Temperature": np.random.uniform(20, 35)
            # NO Wind_production here!
        })

    print("   Creating dummy future data (24h forecast)...")
    future_dates = pd.date_range(start=past_dates[-1], periods=24, freq='h')
    mock_future_json = []
    for date in future_dates:
        mock_future_json.append({
            "Time": str(date),
            "Temperature": np.random.uniform(20, 35),
            "Wind_speed": np.random.uniform(2, 15)
        })

    # 2. LOAD & PREDICT
    try:
        # Check files exis
        if not os.path.exists("day_ahead_model_wind.pth") or not os.path.exists("day_ahead_scaler_wind.gz"):
            print("❌ Files missing! Please make sure 'day_ahead_model.pth' and 'day_ahead_scaler.gz' are in the folder.")
        else:
            print("   Loading model...")
            predictor = WindDayAheadPredictor("day_ahead_model_wind.pth", "day_ahead_scaler_wind.gz")
            
            print("   Running prediction...")
            result = predictor.predict(mock_past_json, mock_future_json)

            print("\n✅ SUCCESS! Inference pipeline works.")
            print(f"Total Day Generation: {result['total_day_mwh']:.2f} MWh")
            print("Sample 5-min output:", result['high_res_5min_forecast'][:5])

    except Exception as e:
        print(f"\n❌ CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()