"""Streamlit demo for the Flight Fare Predictor."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Allow `from src...` imports when launched via `streamlit run app/streamlit_app.py`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import FEATURES, METADATA_PATH, MODEL_PATH


st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="centered")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)


model, metadata = load_artifacts()

st.title("Flight Fare Predictor")
st.caption("Predict a domestic flight ticket price from a few simple inputs.")

if model is None:
    st.error(
        "No trained model found. From the project root run:\n\n"
        "```bash\npython -m src.data_loader\npython -m src.train\n```"
    )
    st.stop()

options = metadata["category_options"]
metrics = metadata["metrics"]

with st.sidebar:
    st.header("Model performance")
    st.metric("R²", f"{metrics['r2']:.3f}")
    st.metric("RMSE", f"{metrics['rmse']:,.0f}")
    st.metric("MAE", f"{metrics['mae']:,.0f}")
    st.caption(f"Trained on {metadata['n_train_rows']:,} rows.")

    best_model = metadata.get("best_model")
    if best_model:
        st.caption(f"Best model: **{best_model}** (selected via {metadata.get('cv_folds', 5)}-fold CV).")

    comparison = metadata.get("model_comparison")
    if comparison:
        with st.expander("Model comparison"):
            st.dataframe(
                pd.DataFrame(comparison)[
                    ["name", "cv_rmse_mean", "cv_rmse_std", "test_rmse", "test_mae", "test_r2"]
                ].rename(
                    columns={
                        "name": "Model",
                        "cv_rmse_mean": "CV RMSE",
                        "cv_rmse_std": "CV RMSE std",
                        "test_rmse": "Test RMSE",
                        "test_mae": "Test MAE",
                        "test_r2": "Test R²",
                    }
                ).style.format(
                    {
                        "CV RMSE": "{:,.0f}",
                        "CV RMSE std": "{:,.0f}",
                        "Test RMSE": "{:,.0f}",
                        "Test MAE": "{:,.0f}",
                        "Test R²": "{:.4f}",
                    }
                ),
                hide_index=True,
            )

    tuning = metadata.get("tuning")
    if tuning and tuning.get("best_params"):
        with st.expander("Best hyperparameters"):
            st.json(tuning["best_params"])

with st.form("predict"):
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("Airline", options["airline"])
        source_city = st.selectbox("From", options["source_city"])
        departure_time = st.selectbox("Departure time", options["departure_time"])
        stops = st.selectbox("Stops", options["stops"])
    with col2:
        travel_class = st.selectbox("Class", options["class"])
        destination_city = st.selectbox(
            "To",
            options["destination_city"],
            index=min(1, len(options["destination_city"]) - 1),
        )
        arrival_time = st.selectbox("Arrival time", options["arrival_time"])
        duration = st.number_input(
            "Duration (hours)", min_value=0.5, max_value=50.0, value=2.5, step=0.25
        )

    days_left = st.slider("Days left until departure", 1, 49, 15)
    submitted = st.form_submit_button("Predict fare", type="primary")

if submitted:
    if source_city == destination_city:
        st.warning("Source and destination must be different cities.")
    else:
        row = pd.DataFrame(
            [
                {
                    "airline": airline,
                    "source_city": source_city,
                    "departure_time": departure_time,
                    "stops": stops,
                    "arrival_time": arrival_time,
                    "destination_city": destination_city,
                    "class": travel_class,
                    "duration": duration,
                    "days_left": days_left,
                }
            ],
            columns=FEATURES,
        )
        prediction = float(model.predict(row)[0])
        lo = max(0.0, prediction - metrics["mae"])
        hi = prediction + metrics["mae"]

        st.success(f"Estimated fare: **₹ {prediction:,.0f}**")
        st.caption(f"Typical range (± MAE): ₹ {lo:,.0f} – ₹ {hi:,.0f}")
        with st.expander("Input used"):
            st.dataframe(row.T.rename(columns={0: "value"}))
