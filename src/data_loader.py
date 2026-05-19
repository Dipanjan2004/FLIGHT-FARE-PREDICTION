"""Download the Flight Price Prediction dataset from Kaggle and load it as a DataFrame."""
from __future__ import annotations

import subprocess
import sys

import pandas as pd

from src.config import DATA_DIR, KAGGLE_DATASET, RAW_CSV


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        print(f"Dataset already present at {RAW_CSV}")
        return

    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(DATA_DIR),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Kaggle download failed.", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    if not RAW_CSV.exists():
        print(
            f"Download finished but {RAW_CSV.name} was not found. "
            f"Inspect {DATA_DIR} and update src/config.py:RAW_CSV if needed.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Saved dataset to {RAW_CSV}")


def load_dataframe() -> pd.DataFrame:
    if not RAW_CSV.exists():
        download_dataset()
    df = pd.read_csv(RAW_CSV)
    # The Kaggle CSV ships with an unnamed index column — drop it if present.
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df


if __name__ == "__main__":
    download_dataset()
    df = load_dataframe()
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(df.head())
