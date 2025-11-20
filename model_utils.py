import numpy as  np
import pandas as pd 
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error,mean_squared_error

TIME_STEPS=5
MODEL_PATH="f1_lstm_model.h5"
SCALER_PATH="scaler_target.pkl"

#time steps we have to match with that of the lstm we have created 

model=None
scaler=None

#So Streamlit does NOT reload the model on every button click.
#It loads just once → fast performance.

#now we have to write a funtion to load the mode and the scaler 

def load_artifacts():
    global model,scaler
    if model is None:
        model=load_model(MODEL_PATH)
    id scaler is None:
        scaler=load_scaler(SCALER_PATH)

#NO REPEATED LAODING HERE , EVERY time predictions used cached model/scaler

#next we have to create LSTM ready sequence.

def create_sequence(series,time_steps=TIME_STEPS):
    X=[]
    for i in range(len(series)-time_steps):
        X.append(series[i:i+ time_steps])
    X=np.array(X).reshape(-1,time_steps,1)
    return X 

# now we are into the main prediction function.

def predict_for_driver_race(df_driver_race):
    load_artifacts()
series_raw = df_driver_race['milliseconds'].values.reshape(-1, 1).astype(float)
    series_scaled = scaler.transform(series_raw).flatten()
X = create_sequences(series_scaled)
y_true_scaled = np.array(series_scaled[TIME_STEPS:]).reshape(-1, 1)
    y_pred_scaled = model.predict(X)
y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_true_scaled).flatten()
mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
laps = df_driver_race['lap'].values[TIME_STEPS:]
result_df = pd.DataFrame({
        'lap': laps,
        'actual_ms': y_true,
        'predicted_ms': y_pred
    })
return result_df, {"mae": mae, "rmse": rmse}



