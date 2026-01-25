import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# 1. MODEL ARCHITECTURE (Must match model2_wind.ipynb exactly)
class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :] 
        prediction = self.fc(last_step_out)
        return self.sigmoid(prediction)

class WindInference:
    def __init__(self, model_path='best_solar_model.pth', scaler_path='scaler.gz'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the artifacts saved in the notebook
        self.scaler = joblib.load(scaler_path)
        
        # Initialize model with 7 input features
        self.model = LSTMModel(input_size=7).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.lookback = 12
        self.feature_cols = [
            'Wind_speed', 'Humidity', 'Temperature',
            'season_sin', 'season_cos', 'weekday_sin', 'weekday_cos'
        ]
        self.minmax_cols = ['Wind_speed', 'Humidity', 'Temperature']

    def preprocess(self, data_list):
        """
        Processes a list of dictionaries into a scaled tensor sequence.
        """
        df = pd.DataFrame(data_list)
        
        # A. Create Cyclical Features
        df["season_sin"] = np.sin(2 * np.pi * df["Season"] / 4)
        df["season_cos"] = np.cos(2 * np.pi * df["Season"] / 4)
        df["weekday_sin"] = np.sin(2 * np.pi * df["Day_of_the_week"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["Day_of_the_week"] / 7)
        
        # B. Scaling (Only for the columns the scaler was trained on)
        df[self.minmax_cols] = self.scaler.transform(df[self.minmax_cols])
        
        # C. Prepare Sequence
        sequence = df[self.feature_cols].tail(self.lookback).values.astype(np.float32)
        return torch.tensor(sequence).unsqueeze(0).to(self.device)

    def predict(self, raw_data):
        if len(raw_data) < self.lookback:
            return {"error": f"Need at least {self.lookback} historical records."}
        
        input_tensor = self.preprocess(raw_data)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        return prediction.item()

# # Example usage for testing
# if __name__ == "__main__":
#     predictor = WindInference()
#     # Mock data would go here
#     # print(predictor.predict(sample_json_data))