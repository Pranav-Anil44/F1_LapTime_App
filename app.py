import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from model_utils import predict_for_driver_race, load_artifacts


# -------------------------------------------------------------
# PAGE CONFIG & TITLE
# -------------------------------------------------------------
st.set_page_config(page_title="F1 Lap-Time Predictor", layout="wide")
st.title("🏎️ F1 Lap-Time Predictor (LSTM Demo)")
st.write("This demo predicts lap times using a trained LSTM model.")


# -------------------------------------------------------------
# LOAD SAMPLE DATA
# -------------------------------------------------------------
@st.cache_data
def load_data(path="sample_data.csv"):
    df = pd.read_csv(path)
    df = df.sort_values(['year', 'name', 'driver_name', 'lap']).reset_index(drop=True)
    return df

df = load_data()


# -------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------
st.sidebar.header("Select Session")

drivers = sorted(df['driver_name'].unique().tolist())
driver = st.sidebar.selectbox("Driver", drivers)

races = sorted(df[df['driver_name'] == driver]['name'].unique().tolist())
race = st.sidebar.selectbox("Grand Prix", races)

years = sorted(df[(df['driver_name'] == driver) & (df['name'] == race)]['year'].unique().tolist())
year = st.sidebar.selectbox("Year", years)

run_btn = st.sidebar.button("Run Prediction")


# -------------------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------------------
if run_btn:
    # Filter selected driver-race-year
    df_sel = df[
        (df['driver_name'] == driver) &
        (df['name'] == race) &
        (df['year'] == year)
    ].sort_values("lap").reset_index(drop=True)

    if df_sel.shape[0] < 10:
        st.warning("⚠️ Not enough laps in this sample to run LSTM prediction.\nNeed at least 10 laps.")
    else:
        with st.spinner("🔄 Loading model and predicting..."):
            load_artifacts()
            result_df, metrics = predict_for_driver_race(df_sel)

        # -------------------------------------------------------------
        # DISPLAY METRICS
        # -------------------------------------------------------------
        st.subheader("📊 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("MAE (ms)", f"{metrics['mae']:.2f}")
        col2.metric("RMSE (ms)", f"{metrics['rmse']:.2f}")

        # -------------------------------------------------------------
        # PLOT (ACTUAL VS PREDICTED)
        # -------------------------------------------------------------
        st.subheader("📈 Actual vs Predicted Lap Times")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result_df['lap'], y=result_df['actual_ms'],
            mode='lines+markers', name='Actual'
        ))
        fig.add_trace(go.Scatter(
            x=result_df['lap'], y=result_df['predicted_ms'],
            mode='lines+markers', name='Predicted'
        ))

        fig.update_layout(
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (ms)",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------
        # TABLE OUTPUT
        # -------------------------------------------------------------
        st.subheader("🔍 Prediction Table (Top 10 Rows)")
        st.dataframe(result_df.head(10))

