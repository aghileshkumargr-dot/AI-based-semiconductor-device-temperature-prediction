import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.features import engineer_features, FEATURES

st.set_page_config(page_title="Semiconductor Temperature AI", page_icon="🌡️", layout="wide")
st.title("🌡️ AI-Based Semiconductor Device Temperature Prediction")
st.caption("Regression models trained on an illustrative synthetic thermal dataset.")
data_path = Path("data/semiconductor_temperature_dataset.csv")
model_path = Path("models/temperature_prediction_model.joblib")
metadata_path = Path("models/model_metadata.joblib")
df = pd.read_csv(data_path)
metadata = joblib.load(metadata_path)
model = joblib.load(model_path)

st.subheader("Model evaluation")
cols = st.columns(3)
m = metadata["metrics"][metadata["best_model"]]
cols[0].metric("MAE", f'{m["MAE_C"]:.2f} °C')
cols[1].metric("RMSE", f'{m["RMSE_C"]:.2f} °C')
cols[2].metric("R²", f'{m["R2"]:.3f}')
st.write(f'**Selected model:** {metadata["best_model"]}')

st.subheader("Upload operating conditions for a prediction")
uploaded = st.file_uploader("CSV with current_A, voltage_V, ambient_temperature_C, load_fraction, estimated_power_loss_W, thermal_resistance_C_per_W, heat_sink_factor", type="csv")
if uploaded:
    sample = pd.read_csv(uploaded)
    engineered = engineer_features(sample)
    sample["predicted_device_temperature_C"] = model.predict(engineered[FEATURES])
    st.dataframe(sample, use_container_width=True)
    st.download_button("Download prediction CSV", sample.to_csv(index=False), "temperature_predictions.csv", "text/csv")

st.subheader("Dataset preview")
st.dataframe(df.head(100), use_container_width=True)
st.subheader("Prediction analysis")
for image in ["actual_vs_predicted.png", "prediction_trend.png", "model_comparison.png"]:
    st.image(str(Path("outputs")/image), use_container_width=True)
st.info("Educational demonstration only. Synthetic thermal labels are generated from a simplified model, not measured silicon temperatures. Validate with a specific device datasheet, LTspice/thermal simulation, or calibrated sensor data before engineering use.")
