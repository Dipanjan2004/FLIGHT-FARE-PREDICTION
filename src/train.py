"""Train a Random Forest regression model on the flight fare dataset and persist it."""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import METADATA_PATH, MODEL_PATH, MODELS_DIR
from src.data_loader import load_dataframe
from src.preprocess import (
    build_preprocessor,
    collect_category_options,
    split_features_target,
)


def train() -> None:
    df = load_dataframe()
    print(f"Loaded {len(df):,} rows")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training Random Forest...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Trained in {time.time() - start:.1f}s")

    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    print("\nTest-set performance")
    print(f"  RMSE : {rmse:,.0f}")
    print(f"  MAE  : {mae:,.0f}")
    print(f"  R^2  : {r2:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(
        {
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
            "category_options": collect_category_options(df),
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
        },
        METADATA_PATH,
    )
    print(f"\nSaved model to    {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    train()
