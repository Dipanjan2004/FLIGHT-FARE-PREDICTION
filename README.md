# Flight Fare Predictor

A supervised-learning regression project that predicts the price of a flight ticket from features like airline, route, departure time, number of stops, travel class, and how many days are left until departure. Built end-to-end: Kaggle dataset → scikit-learn pipeline → trained Random Forest model → Streamlit web UI for live predictions.

## Project structure

```
ML_project/
├── app/streamlit_app.py     # Streamlit UI (the demo)
├── src/
│   ├── config.py            # Paths and constants
│   ├── data_loader.py       # Downloads dataset via Kaggle API
│   ├── preprocess.py        # Cleaning + feature engineering
│   └── train.py             # Trains & persists the model
├── data/                    # Raw + processed datasets (gitignored)
├── models/                  # Saved model artifacts (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Kaggle API credentials

1. Create a Kaggle account at https://www.kaggle.com if you don't have one.
2. Go to **Account → Settings → API → Create New Token**. This downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` and lock it down:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Run

```bash
# 1. Download dataset (~5 MB)
python -m src.data_loader

# 2. Train the model (saves to models/flight_fare_model.joblib)
python -m src.train

# 3. Launch the demo
streamlit run app/streamlit_app.py
```

## Dataset

[Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) by Shubham Bathwal — ~300k Indian domestic flight records with airline, route, timing, class, stops, and price.

## Model

A scikit-learn `Pipeline` combining `ColumnTransformer` (one-hot encoding for categoricals, passthrough for numerics) with a `RandomForestRegressor`. Persisted with `joblib` and reloaded by the Streamlit app at startup.
