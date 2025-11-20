import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuration
TIME_STEPS = 5                        
MODEL_PATH = "f1_lstm_model.h5"
SCALER_PATH = "scaler_target.pkl"

# Global objects (loaded once)
model = None
scaler = None


def load_artifacts():
    """Load model and scaler only once."""
    global model, scaler

    if model is None:
        model = load_model(MODEL_PATH)

    if scaler is None:
        scaler = joblib.load(SCALER_PATH)


def create_sequences(series, time_steps=TIME_STEPS):
    """Convert 1D series into LSTM-ready sequences."""
    X = []
    for i in range(len(series) - time_steps):
        X.append(series[i:i + time_steps])
    X = np.array(X).reshape(-1, time_steps, 1)
    return X


def predict_for_driver_race(df_driver_race):
    """Main prediction logic for a selected driver and race."""
    load_artifacts()

    # Prepare raw and scaled series
    series_raw = df_driver_race['milliseconds'].values.reshape(-1, 1).astype(float)
    series_scaled = scaler.transform(series_raw).flatten()

    # Create LSTM input sequences
    X = create_sequences(series_scaled)

    # True values for comparison
    y_true_scaled = np.array(series_scaled[TIME_STEPS:]).reshape(-1, 1)

    # Predict (scaled)
    y_pred_scaled = model.predict(X)

    # Inverse scaling
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_true_scaled).flatten()

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # Create results dataframe
    laps = df_driver_race['lap'].values[TIME_STEPS:]

    result_df = pd.DataFrame({
        'lap': laps,
        'actual_ms': y_true,
        'predicted_ms': y_pred
    })

    return result_df, {"mae": mae, "rmse": rmse}
