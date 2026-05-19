"""Build the sklearn preprocessing ColumnTransformer for flight features."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()


def collect_category_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Returns sorted unique values per categorical column — used by the Streamlit form."""
    return {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}
