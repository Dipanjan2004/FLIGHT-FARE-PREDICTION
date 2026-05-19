from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

KAGGLE_DATASET = "shubhambathwal/flight-price-prediction"
RAW_CSV = DATA_DIR / "Clean_Dataset.csv"

MODEL_PATH = MODELS_DIR / "flight_fare_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"

TARGET = "price"

CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]
NUMERIC_FEATURES = ["duration", "days_left"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
