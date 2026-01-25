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

# Feature Lists (Order must match your training scaler EXACTLY)
PAST_FEATURE_COLS = [
    'DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature', 
    'season_sin', 'season_cos', 'weekday_sin', 'weekday_cos', 
    'PV_potential', 'GHI_diff_5m', 'GHI_rolling_std_30m', 'Clearness_Index'
]

# Future features don't include target or historical rolling stats
FUTURE_DRIVER_COLS = [
    'GHI', 'Temperature', 'Wind_speed', 
    'season_sin', 'season_cos', 'weekday_sin', 'weekday_cos'
]

# --- 2. MODEL ARCHITECTURE (Must match training) ---
class DayAheadNet(nn.Module):
    def __init__(self, past_features, future_features, hidden_size=64):
        super(DayAheadNet, self).__init__()
        self.lstm = nn.LSTM(input_size=past_features, hidden_size=hidden_size, batch_first=True)
        self.future_fc = nn.Linear(24 * future_features, hidden_size)
        self.combine_fc = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, 24) 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, past, future):
        _, (hidden, _) = self.lstm(past)
        context = hidden[-1]
        future_flat = future.reshape(future.shape[0], -1) 
        future_context = torch.relu(self.future_fc(future_flat))
        combined = torch.cat((context, future_context), dim=1)
        combined = torch.relu(self.combine_fc(combined))
        return self.sigmoid(self.output_fc(combined))

# --- 3. HELPER: FEATURE ENGINEERING ---
def calculate_features(df):
    """Generates cyclic time features and physics interactions."""
    # Ensure Time is datetime
    if not np.issubdtype(df['Time'].dtype, np.datetime64):
        df['Time'] = pd.to_datetime(df['Time'])
        
    # Seasonality / Cyclical (Approximate using Quarter/DayOfWeek if exact DayOfYear not avail)
    df["season_sin"] = np.sin(2 * np.pi * df["Time"].dt.quarter / 4)
    df["season_cos"] = np.cos(2 * np.pi * df["Time"].dt.quarter / 4)
    df["weekday_sin"] = np.sin(2 * np.pi * df["Time"].dt.dayofweek / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["Time"].dt.dayofweek / 7)
    
    # Physics
    if 'GHI' in df.columns and 'Temperature' in df.columns:
        df["PV_potential"] = df["GHI"] * df["Temperature"]
        
    return df

# --- 4. PREDICTOR CLASS ---
class SolarDayAheadPredictor:
    def __init__(self, model_path, scaler_path):
        self.scaler = joblib.load(scaler_path)
        # Initialize Model
        self.model = DayAheadNet(
            past_features=len(PAST_FEATURE_COLS), 
            future_features=len(FUTURE_DRIVER_COLS)
        ).to(DEVICE)
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print("✅ Day-Ahead Model Loaded")

    def preprocess_past(self, data_list):
        """
        Handles raw 5-min history -> Resamples to Hourly -> Scales
        """
        df = pd.DataFrame(data_list)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time')
        
        # 1. Calc Rolling Stats on HIGH RES data (before resampling)
        df['GHI_diff_5m'] = df['GHI'].diff().fillna(0)
        df['GHI_rolling_std_30m'] = df['GHI'].rolling(window=6, min_periods=1).std().fillna(0)
        
        # 2. Clearness Index (Simplified Proxy for Inference)
        # Using a fixed 1000 W/m2 divisor if clear sky map isn't available
        df['Clearness_Index'] = df['GHI'] / (1000.0 + 1e-9) 

        # 3. Rename 'PV_production' to 'target' for the scaler
        # if 'PV_production' in df.columns:
        #     df['target'] = df['PV_production']
        # else:
        #     df['target'] = 0 # Fallback if missing
            
        # 4. Add Cyclical Features
        df = calculate_features(df)

        # 5. RESAMPLE TO HOURLY (Targeting 24 rows)
        df.set_index('Time', inplace=True)
        df_hourly = df.resample('1h').mean()
        
        # Handle Missing Data (Forward Fill)
        df_hourly = df_hourly.fillna(method='ffill').fillna(0)
        
        # Ensure we take the LAST 24 hours
        df_hourly = df_hourly.tail(24)
        
        if len(df_hourly) < 24:
            raise ValueError(f"Not enough history. Needed 24h, got {len(df_hourly)}h")

        # 6. Scale
        scaled_values = self.scaler.transform(df_hourly[PAST_FEATURE_COLS])
        
        return torch.tensor(scaled_values, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    def preprocess_future(self, data_list):
        """
        Handles future forecast (Already Hourly) -> Features -> Scales
        """
        df = pd.DataFrame(data_list)
        df = calculate_features(df)
        
        # Create a dummy DF to satisfy the Scaler's expected shape (PAST_FEATURE_COLS)
        # We fill missing columns with 0 because we only extract FUTURE_DRIVER_COLS later
        dummy_df = pd.DataFrame(0, index=df.index, columns=PAST_FEATURE_COLS)
        for col in FUTURE_DRIVER_COLS:
            if col in df.columns:
                dummy_df[col] = df[col]
        
        # Scale everything
        scaled_full = self.scaler.transform(dummy_df)
        
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
    


    # ... (End of your SolarDayAheadPredictor class) ...

# --- SELF-TEST BLOCK ---
if __name__ == "__main__":
    print("🧪 Starting Self-Test for Solar Predictor...")

    # 1. GENERATE MOCK DATA
    # We need to simulate exactly what the Web Developer will send you.
    # Past: 24 hours of 5-minute data = 288 data points
    print("   Creating dummy past data (24h history)...")
    past_dates = pd.date_range(end=pd.Timestamp.now(), periods=288, freq='5min')
    mock_past_json = []
    for date in past_dates:
        mock_past_json.append({
            "Time": str(date),
            "DHI": np.random.uniform(50, 200),
            "DNI": np.random.uniform(500, 900),
            "GHI": np.random.uniform(0, 1000),  # Some night (0), some day
            "Wind_speed": np.random.uniform(2, 10),
            "Humidity": np.random.uniform(30, 80),
            "Temperature": np.random.uniform(20, 35)
            # "PV_production": np.random.uniform(0, 2500) # The target history
        })

    # Future: 24 hours of 1-hour weather forecast = 24 data points
    print("   Creating dummy future data (24h forecast)...")
    future_dates = pd.date_range(start=past_dates[-1], periods=24, freq='h')
    mock_future_json = []
    for date in future_dates:
        mock_future_json.append({
            "Time": str(date),
            "GHI": np.random.uniform(0, 1000), # The "Oracle" weather
            "Temperature": np.random.uniform(20, 35),
            "Wind_speed": np.random.uniform(2, 10)
        })

    # 2. LOAD & PREDICT
    try:
        # Make sure these filenames match exactly what you saved!
        MODEL_FILE = "day_ahead_model.pth"
        SCALER_FILE = "day_ahead_scaler.gz"
        
        print(f"   Loading model from {MODEL_FILE}...")
        predictor = SolarDayAheadPredictor(MODEL_FILE, SCALER_FILE)
        
        print("   Running prediction logic...")
        result = predictor.predict(mock_past_json, mock_future_json)

        # 3. VERIFY OUTPUTS
        print("\n✅ SUCCESS! Inference pipeline works.")
        print("-" * 30)
        print(f"Total Day Generation: {result['total_day_mwh']:.2f} MWh")
        print(f"Hourly Points: {len(result['hourly_forecast'])} (Expected: 24)")
        print(f"5-Min Points:  {len(result['high_res_5min_forecast'])} (Expected: 288)")
        print("-" * 30)
        print("Sample 5-min output (First 5):", result['high_res_5min_forecast'][:5])

    except FileNotFoundError:
        print("\n❌ ERROR: Model/Scaler files missing!")
        print("   Please ensure 'day_ahead_model.pth' and 'day_ahead_scaler.gz' are in this folder.")
    except Exception as e:
        print(f"\n❌ CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()