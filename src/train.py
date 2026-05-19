"""Train and compare multiple regression models on the flight-fare dataset.

Pipeline:
1. Load the dataset and split into train/test (80/20, fixed seed).
2. Evaluate three candidate models with 5-fold cross-validation
   (Linear Regression, Random Forest, Histogram Gradient Boosting).
3. Pick the best candidate by mean CV RMSE.
4. Tune its hyperparameters with RandomizedSearchCV.
5. Persist the best model + metadata + a model-comparison and
   feature-importance plot to disk.
"""
from __future__ import annotations

import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import METADATA_PATH, MODEL_PATH, MODELS_DIR, REPORTS_DIR
from src.data_loader import load_dataframe
from src.preprocess import (
    build_preprocessor,
    collect_category_options,
    split_features_target,
)


CV_FOLDS = 5
RANDOM_STATE = 42


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    """Three regression models spanning a complexity range."""
    return {
        "LinearRegression": make_pipeline(LinearRegression()),
        "RandomForest": make_pipeline(
            RandomForestRegressor(
                n_estimators=80,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "HistGradientBoosting": make_pipeline(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.1,
                max_depth=None,
                min_samples_leaf=20,
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate_candidate(name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test):
    print(f"\n— {name} —")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    start = time.time()
    cv_scores = -cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    cv_time = time.time() - start
    print(f"  CV RMSE : {cv_scores.mean():,.0f}  ± {cv_scores.std():,.0f}  ({cv_time:.1f}s)")

    start = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - start
    preds = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"  Test RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | R²: {r2:.4f}  (fit {fit_time:.1f}s)")

    return {
        "name": name,
        "cv_rmse_mean": float(cv_scores.mean()),
        "cv_rmse_std": float(cv_scores.std()),
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "fit_time_seconds": fit_time,
        "pipeline": pipeline,
    }


def param_grid_for(name: str) -> dict[str, list]:
    """Hyperparameter grid for RandomizedSearchCV — keys reference the 'model' step
    of the pipeline (e.g. 'model__n_estimators')."""
    if name == "RandomForest":
        return {
            "model__n_estimators": [80, 120, 200],
            "model__max_depth": [None, 15, 25, 35],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": [0.5, 0.7, 1.0],
        }
    if name == "HistGradientBoosting":
        return {
            "model__learning_rate": [0.05, 0.08, 0.1, 0.15],
            "model__max_iter": [200, 300, 500, 700],
            "model__max_depth": [None, 6, 8, 12],
            "model__min_samples_leaf": [10, 20, 40, 80],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        }
    if name == "LinearRegression":
        # Nothing meaningful to tune; return an empty grid (search will skip).
        return {}
    raise ValueError(name)


def tune_best(best_name: str, pipeline: Pipeline, X_train, y_train, X_test, y_test):
    grid = param_grid_for(best_name)
    if not grid:
        print(f"\nNo tuning grid for {best_name} — keeping default.")
        return pipeline, {}

    print(f"\nTuning {best_name} with RandomizedSearchCV (n_iter=10, cv=3)...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=grid,
        n_iter=10,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    start = time.time()
    search.fit(X_train, y_train)
    print(f"  search took {time.time() - start:.0f}s")
    print(f"  best CV RMSE: {-search.best_score_:,.0f}")
    print(f"  best params : {search.best_params_}")

    preds = search.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"  tuned test  : RMSE {rmse:,.0f} | MAE {mae:,.0f} | R² {r2:.4f}")
    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "tuned_test_rmse": rmse,
        "tuned_test_mae": mae,
        "tuned_test_r2": r2,
    }


def plot_model_comparison(results: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in results]
    test_rmse = [r["test_rmse"] for r in results]
    test_mae = [r["test_mae"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, test_rmse, width=0.4, label="Test RMSE", color="steelblue")
    ax.bar(x + 0.2, test_mae, width=0.4, label="Test MAE", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Error (Rs.)")
    ax.set_title("Model comparison on the held-out test set")
    ax.legend()
    for i, (rmse, mae) in enumerate(zip(test_rmse, test_mae)):
        ax.text(i - 0.2, rmse, f"{rmse:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + 0.2, mae, f"{mae:,.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = REPORTS_DIR / "07_model_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(REPORTS_DIR.parent)}")


def plot_feature_importance(pipeline: Pipeline, model_name: str, top_k: int = 15) -> None:
    """Generate a horizontal bar chart of the most important features.

    Tree models expose `feature_importances_`. Linear models expose `coef_`.
    Feature names come from the fitted ColumnTransformer.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preproc = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preproc.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
        label = "|Coefficient|"
    else:
        print(f"  {model_name} exposes no feature importance — skipping plot.")
        return

    k = min(top_k, len(importances))
    idx = np.argsort(importances)[-k:]
    fig, ax = plt.subplots(figsize=(9, max(5, k * 0.35)))
    ax.barh(range(k), importances[idx], color="steelblue")
    ax.set_yticks(range(k))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel(label)
    ax.set_title(f"Top {k} most important features — {model_name}")
    fig.tight_layout()
    out = REPORTS_DIR / "08_feature_importance.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.relative_to(REPORTS_DIR.parent)}")


def train() -> None:
    df = load_dataframe()
    print(f"Loaded {len(df):,} rows.")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

    print(f"\n=== Comparing candidate models with {CV_FOLDS}-fold CV ===")
    results = []
    for name, pipeline in candidate_models().items():
        results.append(
            evaluate_candidate(name, pipeline, X_train, y_train, X_test, y_test)
        )

    results.sort(key=lambda r: r["cv_rmse_mean"])
    best = results[0]
    print(f"\nBest by CV RMSE: {best['name']} ({best['cv_rmse_mean']:,.0f})")

    tuned_pipeline, tuning_info = tune_best(
        best["name"], best["pipeline"], X_train, y_train, X_test, y_test
    )

    print("\nGenerating diagnostic plots...")
    plot_model_comparison(results)
    plot_feature_importance(tuned_pipeline, best["name"])

    final_preds = tuned_pipeline.predict(X_test)
    final_rmse = float(np.sqrt(mean_squared_error(y_test, final_preds)))
    final_mae = float(mean_absolute_error(y_test, final_preds))
    final_r2 = float(r2_score(y_test, final_preds))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_pipeline, MODEL_PATH)
    joblib.dump(
        {
            "metrics": {"rmse": final_rmse, "mae": final_mae, "r2": final_r2},
            "category_options": collect_category_options(df),
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
            "best_model": best["name"],
            "cv_folds": CV_FOLDS,
            "model_comparison": [
                {k: v for k, v in r.items() if k != "pipeline"} for r in results
            ],
            "tuning": tuning_info,
        },
        METADATA_PATH,
    )
    print(f"\nFinal model: {best['name']} (tuned)")
    print(f"  RMSE {final_rmse:,.0f} | MAE {final_mae:,.0f} | R² {final_r2:.4f}")
    print(f"Saved model    -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")


if __name__ == "__main__":
    train()
