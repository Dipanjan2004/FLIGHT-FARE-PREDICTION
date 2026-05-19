# Flight Fare Predictor — Complete Project Guide

> A machine-learning **regression** project that predicts the price of a domestic Indian flight ticket from features like airline, route, timing, class, stops, and how many days are left until departure.
>
> This document is written for a Machine Learning course. It explains **every concept and every line of code** so a reader who is new to ML can follow along and so the author can defend the project in a viva.

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Why this is a regression problem](#2-why-this-is-a-regression-problem)
3. [Machine-learning concepts used (theory primer)](#3-machine-learning-concepts-used-theory-primer)
4. [Dataset](#4-dataset)
5. [Project architecture & file map](#5-project-architecture--file-map)
6. [End-to-end flow](#6-end-to-end-flow)
7. [Code walkthrough — file by file](#7-code-walkthrough--file-by-file)
8. [The preprocessing pipeline in depth](#8-the-preprocessing-pipeline-in-depth)
9. [The model: Random Forest Regressor](#9-the-model-random-forest-regressor)
10. [Train / test split — why we do it](#10-train--test-split--why-we-do-it)
11. [Evaluation metrics (RMSE, MAE, R²)](#11-evaluation-metrics-rmse-mae-r²)
12. [Results & interpretation](#12-results--interpretation)
13. [The Streamlit demo app](#13-the-streamlit-demo-app)
14. [How to run the project](#14-how-to-run-the-project)
15. [Limitations & possible extensions](#15-limitations--possible-extensions)
16. [Viva / FAQ — likely questions a teacher will ask](#16-viva--faq--likely-questions-a-teacher-will-ask)
17. [Glossary](#17-glossary)

---

## 1. Problem statement

**Goal:** Given a few simple inputs about a flight (airline, source city, destination city, departure time-of-day, arrival time-of-day, number of stops, travel class, total duration, and number of days remaining until departure), predict the **ticket price in Indian Rupees (₹)**.

**Why this is useful:** Customers want to know whether the price they see today is reasonable, and whether it is likely to be cheaper if they wait. Airlines themselves use very similar models for dynamic pricing.

**ML category:** Supervised learning → Regression.

---

## 2. Why this is a regression problem

Machine-learning tasks fall into two big families:

| Family | What it predicts | Example |
|---|---|---|
| **Classification** | A *category* (discrete label) | Spam vs. not spam |
| **Regression** | A *number* (continuous value) | Tomorrow's temperature, house price, **flight fare** |

The target variable here, `price`, is a continuous positive real number (e.g. ₹5,953, ₹12,400, ₹68,200). That makes this a **regression problem**, and the loss/metrics we use are regression metrics (RMSE, MAE, R²), not classification metrics (accuracy, F1).

---

## 3. Machine-learning concepts used (theory primer)

Before walking through the code, here are the concepts the project relies on. Every one of them appears later in this document with its concrete usage highlighted.

### 3.1 Supervised learning
The dataset gives us both the inputs (**features**, `X`) and the correct answer (**target**, `y`). The algorithm sees many `(X, y)` examples and learns a function `f` such that `f(X) ≈ y`. Supervision = "we know the right answer for the training rows."

### 3.2 Features vs. target
- **Features (X)** — the columns we are *allowed* to look at when predicting: airline, route, etc.
- **Target (y)** — the column we are trying to predict: `price`.

### 3.3 Categorical vs. numeric features
- **Numeric** features are already numbers and can be fed to a model directly: `duration`, `days_left`.
- **Categorical** features are text labels (`"SpiceJet"`, `"Mumbai"`, `"Evening"`). Most ML algorithms can't read text — we must convert them to numbers via **one-hot encoding** (Section 8).

### 3.4 Train / test split
We hold out a chunk of the data (here: 20%) that the model is never trained on. We measure performance only on this held-out part — that gives an honest estimate of how the model will behave on flights it has not seen.

### 3.5 The fit / predict pattern
Every scikit-learn estimator follows two steps:
- `.fit(X_train, y_train)` — learn from the training data.
- `.predict(X_new)` — produce a prediction for new rows.

### 3.6 Pipelines
A scikit-learn `Pipeline` chains preprocessing + model into a single object. Once trained, the pipeline takes raw inputs and outputs a prediction in one call. This is critical: it guarantees the **exact same preprocessing** is applied during training and during deployment in Streamlit. Without a pipeline, it is very easy to encode test/inference data differently from training data and silently get garbage predictions.

### 3.7 Ensemble learning — Random Forest
Instead of training one decision tree (which can overfit), we train **many** trees on random subsets of the data and features, then average their predictions. The Random Forest is robust, handles non-linearities and feature interactions, and works well out-of-the-box with little tuning. (Full explanation in Section 9.)

### 3.8 Evaluation metrics
- **RMSE** — Root Mean Squared Error. Big errors are punished more than small errors.
- **MAE** — Mean Absolute Error. The "average ₹ off" — easy to explain to a non-technical audience.
- **R²** — Coefficient of determination. Fraction of the variance in price that the model explains. 1.0 = perfect, 0 = no better than guessing the mean.

---

## 4. Dataset

**Source:** [Flight Price Prediction by Shubham Bathwal on Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

**Size:** 300,153 rows × 12 columns.

**Columns (after dropping the unnamed index column):**

| Column | Type | Meaning |
|---|---|---|
| `airline` | categorical | One of: AirAsia, Air_India, GO_FIRST, Indigo, SpiceJet, Vistara |
| `flight` | categorical | Flight code (e.g. SG-8709) — **not used** as a feature (too high cardinality, would overfit) |
| `source_city` | categorical | Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai |
| `departure_time` | categorical | Time-of-day bucket: Early_Morning, Morning, Afternoon, Evening, Night, Late_Night |
| `stops` | categorical | zero, one, two_or_more |
| `arrival_time` | categorical | Same six buckets as `departure_time` |
| `destination_city` | categorical | Same six cities |
| `class` | categorical | Economy, Business |
| `duration` | numeric | Total flight duration in hours (e.g. 2.17) |
| `days_left` | numeric | Days between booking date and flight date (1–49) |
| `price` | numeric | **Target.** Ticket price in ₹ |

The Kaggle archive also contains `business.csv` and `economy.csv` which are the unprocessed splits; we only use `Clean_Dataset.csv` because it is already merged and tidied by the dataset author.

---

## 5. Project architecture & file map

```
ML_project/
├── app/
│   └── streamlit_app.py        # Interactive web UI for live predictions
├── src/
│   ├── __init__.py             # Marks src/ as a Python package
│   ├── config.py               # Central constants (paths, feature lists, target)
│   ├── data_loader.py          # Downloads dataset from Kaggle, loads it to DataFrame
│   ├── preprocess.py           # Builds the sklearn preprocessing transformer
│   └── train.py                # Trains the model, evaluates it, saves it to disk
├── data/                       # Raw CSV (gitignored — regenerated by data_loader)
│   └── Clean_Dataset.csv
├── models/                     # Trained model + metadata (gitignored)
│   ├── flight_fare_model.joblib
│   └── model_metadata.joblib
├── requirements.txt            # Python dependencies
├── README.md                   # Short setup guide
└── PROJECT_GUIDE.md            # ← This document
```

**Design principle — separation of concerns.** Each module does one thing:
- `config.py` says **what** the project knows about (paths, column names).
- `data_loader.py` is **where** the data comes from.
- `preprocess.py` is **how** raw rows become model-ready numbers.
- `train.py` is **how** the model is built and measured.
- `streamlit_app.py` is **how** users interact with it.

This is a clean structure that scales beyond a notebook and is what an industry team would actually do.

---

## 6. End-to-end flow

```
┌─────────────────────────┐
│  Kaggle: flight dataset │
└───────────┬─────────────┘
            │ kaggle CLI
            ▼
┌─────────────────────────┐
│   data/Clean_Dataset.csv │
└───────────┬─────────────┘
            │ pandas.read_csv
            ▼
┌─────────────────────────┐
│  DataFrame (300k rows)  │
└───────────┬─────────────┘
            │ split_features_target
            ▼
┌─────────────────────────┐
│   X (features), y (price)│
└───────────┬─────────────┘
            │ train_test_split (80/20)
            ▼
┌─────────────────────────┐
│ X_train,y_train  X_test,y_test │
└───────────┬─────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│ Pipeline:                            │
│   1. ColumnTransformer               │
│      • OneHotEncoder on categoricals │
│      • passthrough on numerics       │
│   2. RandomForestRegressor           │
└───────────┬──────────────────────────┘
            │ .fit(X_train, y_train)
            ▼
┌─────────────────────────┐
│  Trained pipeline       │
└───────────┬─────────────┘
            │ joblib.dump
            ▼
┌─────────────────────────┐
│ models/flight_fare_model.joblib │
└───────────┬─────────────┘
            │ joblib.load (Streamlit startup)
            ▼
┌─────────────────────────┐
│  Streamlit UI form      │
└───────────┬─────────────┘
            │ user submits values
            ▼
┌─────────────────────────┐
│  Predicted fare in ₹    │
└─────────────────────────┘
```

---

## 7. Code walkthrough — file by file

### 7.1 `src/config.py` — central configuration

```python
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

KAGGLE_DATASET = "shubhambathwal/flight-price-prediction"
RAW_CSV = DATA_DIR / "Clean_Dataset.csv"

MODEL_PATH = MODELS_DIR / "flight_fare_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.joblib"

TARGET = "price"

CATEGORICAL_FEATURES = [
    "airline", "source_city", "departure_time",
    "stops", "arrival_time", "destination_city", "class",
]
NUMERIC_FEATURES = ["duration", "days_left"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
```

**What this does, line by line:**
- `Path(__file__).resolve().parent.parent` — computes the absolute project root regardless of where the user runs the script from. Avoids fragile relative paths.
- `DATA_DIR`, `MODELS_DIR` — folder constants. Everywhere else in the codebase imports from here instead of hard-coding strings. Change a path once, here, and the whole project follows.
- `KAGGLE_DATASET` — the Kaggle dataset slug used by the Kaggle CLI.
- `MODEL_PATH` / `METADATA_PATH` — `.joblib` files where the trained pipeline and a small metadata dictionary are persisted.
- `TARGET = "price"` — the column we predict.
- `CATEGORICAL_FEATURES`, `NUMERIC_FEATURES`, `FEATURES` — the canonical feature lists, used by both training and the Streamlit form so they cannot drift apart. Notice `flight` is **not** in the list — we deliberately exclude it because each flight code is almost unique and would just be memorised.

### 7.2 `src/data_loader.py` — fetching the dataset

```python
def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        print(f"Dataset already present at {RAW_CSV}")
        return

    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(DATA_DIR), "--unzip"],
        capture_output=True, text=True,
    )
    ...
```

**Concepts on display:**
- **Idempotent download.** `if RAW_CSV.exists(): return` — running the script twice doesn't re-download. This is good hygiene; networked operations should not be silently repeated.
- **Subprocess call to the `kaggle` CLI.** The Kaggle Python package ships a command-line tool; we shell out to it rather than re-implementing API auth. `--unzip` flag makes Kaggle expand the archive directly into `data/`.
- **Error handling.** If `result.returncode != 0` or the expected CSV doesn't appear, we `sys.exit(1)` with a clear message — failing loudly rather than silently producing a corrupt run.

```python
def load_dataframe() -> pd.DataFrame:
    if not RAW_CSV.exists():
        download_dataset()
    df = pd.read_csv(RAW_CSV)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])
    return df
```

- Auto-downloads if the file is missing, so any other module can just call `load_dataframe()` and trust it.
- Drops the leading "Unnamed: 0" index column that pandas exports when a CSV has an unnamed first column — a small but real data-cleaning step.

### 7.3 `src/preprocess.py` — turning rows into model-ready numbers

```python
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
```

- A **`ColumnTransformer`** applies different transformations to different columns and concatenates the results. Here:
  - the 7 categorical columns → one-hot encoded
  - the 2 numeric columns → passed through unchanged
- `handle_unknown="ignore"` — if at inference time we ever see a category the model wasn't trained on (a brand-new airline, say), don't crash; produce a row of zeros for that feature instead.
- `sparse_output=False` — return a dense NumPy array. Random Forest is fine with dense data; this trades a bit of memory for simpler downstream code.

```python
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()
```

- Splits the DataFrame into `X` (the features the model is allowed to see) and `y` (the target). `.copy()` defends against accidental in-place modification of the original DataFrame.

```python
def collect_category_options(df: pd.DataFrame) -> dict[str, list[str]]:
    return {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}
```

- Collects the unique values per categorical column. We persist this dictionary so the Streamlit dropdowns show *exactly* the categories the model knows about — no typos, no surprises.

### 7.4 `src/train.py` — training and evaluating

```python
df = load_dataframe()
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- Loads the data, splits it into features and target, then carves off a 20% test set. `random_state=42` makes the split reproducible — every time you run the script you get the same 60,031 test rows, so metrics are comparable across runs.

```python
pipeline = Pipeline(steps=[
    ("preprocess", build_preprocessor()),
    ("model", RandomForestRegressor(
        n_estimators=120,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )),
])
```

- The whole "raw row → prediction" recipe lives in one `Pipeline` object. Hyperparameters explained:
  - `n_estimators=120` — number of trees in the forest. More trees = lower variance but slower training/prediction.
  - `max_depth=None` — let each tree grow until splits no longer improve (controlled instead by `min_samples_leaf`).
  - `min_samples_leaf=2` — a leaf must represent ≥2 training samples. This regularises: prevents single-row leaves that memorise noise.
  - `n_jobs=-1` — use all CPU cores in parallel. Trees are independent, so this scales almost linearly.
  - `random_state=42` — same idea as in the split: makes randomness (bootstrap sampling, feature subsets) reproducible.

```python
pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
mae  = float(mean_absolute_error(y_test, preds))
r2   = float(r2_score(y_test, preds))
```

- Train, predict, score. All three metrics are computed on the **test set** (`y_test`) — the data the model has never seen.

```python
joblib.dump(pipeline, MODEL_PATH)
joblib.dump({
    "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
    "category_options": collect_category_options(df),
    "n_train_rows": int(len(X_train)),
    "n_test_rows": int(len(X_test)),
}, METADATA_PATH)
```

- `joblib.dump` serialises the trained pipeline (incl. the fitted OneHotEncoder) to disk. We **also** save metadata: the metrics for display, and the dropdown options. The Streamlit app reads both at startup.

### 7.5 `app/streamlit_app.py` — the user-facing demo

(Detailed walk-through in [Section 13](#13-the-streamlit-demo-app).)

---

## 8. The preprocessing pipeline in depth

### 8.1 One-hot encoding

A Random Forest cannot operate on the literal string `"SpiceJet"`. We must turn each category into numbers. The naive way (label-encode `SpiceJet → 0`, `Air_India → 1`, …) accidentally injects an ordering that doesn't exist — it would imply `Vistara > SpiceJet`, which is nonsense.

**One-hot encoding** instead creates one binary column per category:

| airline | airline_AirAsia | airline_Air_India | airline_GO_FIRST | airline_Indigo | airline_SpiceJet | airline_Vistara |
|---|---|---|---|---|---|---|
| SpiceJet | 0 | 0 | 0 | 0 | **1** | 0 |
| Vistara  | 0 | 0 | 0 | 0 | 0 | **1** |

After one-hot encoding all 7 categorical columns the model sees roughly:
`6 (airline) + 6 (source) + 6 (dep_time) + 3 (stops) + 6 (arr_time) + 6 (dest) + 2 (class) = 35` categorical binary features, plus the 2 numeric features = **37 input features** in total.

### 8.2 `ColumnTransformer` — different rules for different columns

`ColumnTransformer` is the sklearn primitive that lets us say "apply OneHotEncoder to columns A, B, C, and leave columns D, E unchanged." It outputs a single feature matrix with all transformed pieces glued together. Crucially, the *fitted* ColumnTransformer remembers which categories it saw during training, so at inference time it produces the same columns in the same order — even if the input only has one row.

### 8.3 `handle_unknown="ignore"`

A subtle but important flag. If a user (or future data) passes a category the encoder did not see during training, the encoder would normally raise. With `handle_unknown="ignore"` it instead emits a row of zeros — the model treats it as "an airline with no signal" and falls back on the other features. This makes the Streamlit app robust.

### 8.4 Why no scaling on the numerics?

Tree-based models (Random Forest, Gradient Boosting, XGBoost) are **scale-invariant** — they split on thresholds (`days_left ≤ 7?`), so multiplying `duration` by 1000 wouldn't change a single split. Scaling is essential for distance-based or gradient-based models (kNN, linear regression, neural nets), but harmless and unnecessary here. That's why `NUMERIC_FEATURES` are simply `"passthrough"`.

---

## 9. The model: Random Forest Regressor

### 9.1 Decision trees — the building block

A decision tree partitions the feature space with axis-aligned splits. For regression, each leaf stores the *mean* of the training targets that landed in it; prediction = walk the tree from root to a leaf and return that mean.

```
                  class == Business?
                  ┌──── yes ────┐
                  │             │
            days_left ≤ 7?     stops == zero?
            ┌──┴──┐            ┌──┴──┐
        ₹54,000  ₹38,000    ₹6,200  ₹4,900
```

A single tree is fast and interpretable but **overfits** wildly — left unconstrained it memorises every training row.

### 9.2 Random Forest — bagging many trees

A Random Forest fixes overfitting by combining many trees:

1. **Bootstrap sampling.** For each tree, draw `n` rows *with replacement* from the training set. Each tree therefore sees a slightly different ~63% of the data.
2. **Feature subsampling.** At each split, the tree may only consider a random subset of features (defaulting to all features for regression in sklearn, but with row randomness still doing the de-correlation).
3. **Average the predictions.** For a new row, run it through all 120 trees and average their outputs.

Why this works: individual trees are high-variance (small data changes → very different tree). Averaging many decorrelated high-variance estimators dramatically reduces variance without increasing bias much — the bias–variance trade-off, exploited.

### 9.3 Hyperparameters chosen and why

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` | 120 | More trees → lower variance, slower. 120 is a good sweet spot for this dataset. |
| `max_depth` | None | Trees grow until other stopping criteria fire — RF doesn't usually overfit because of averaging. |
| `min_samples_leaf` | 2 | A leaf must contain ≥2 rows. Stops the tree from creating a leaf for every single training example. |
| `n_jobs` | -1 | Use all CPU cores in parallel. Each tree is independent → near-linear speedup. |
| `random_state` | 42 | Reproducibility. |

### 9.4 Strengths and limitations of Random Forest here

**Strengths**
- Handles non-linear relationships (e.g. price spikes when `days_left < 3`) and feature interactions (`class=Business` × `airline=Vistara`) automatically.
- Robust to outliers — splits are based on order, not magnitude.
- Almost no preprocessing required (no scaling, no log transform needed for tree-based models).
- Provides feature-importance scores for free.

**Limitations**
- Larger to store and slower to predict than a linear model.
- Predictions cannot extrapolate beyond the range of the training data (e.g. a `days_left=200` value would be clipped to behaviour seen near the max of 49).
- Can be biased on imbalanced features.

---

## 10. Train / test split — why we do it

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

If we evaluate the model on the same data it was trained on, we are measuring **memorisation**, not learning. A sufficiently flexible model can score perfectly on training data and still fail on real flights.

By holding out 20% of the rows (≈60,000), we get an honest estimate of **generalisation error** — how the model performs on unseen flights. `random_state=42` ensures the split is the same every run, so today's metrics are comparable to tomorrow's.

**Could we do better?** Yes — *k-fold cross-validation* would average performance over k different splits. We don't here because the dataset is large (300k rows) and the held-out 60k test set is already statistically very stable. For smaller datasets, cross-validation would be the right call.

---

## 11. Evaluation metrics (RMSE, MAE, R²)

Let `y_i` = true price for row `i`, `ŷ_i` = predicted price, `ȳ` = mean of true prices, `n` = number of rows.

### 11.1 MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Interpretation:** "On average, our prediction is off by **₹MAE**." Easy to explain. Treats every rupee of error equally.

### 11.2 RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }$$

**Interpretation:** Like MAE but *squares* the errors before averaging. This means a single big miss (₹10,000 off on a business-class flight) hurts much more than ten ₹1,000 misses. Same units as the target (₹). RMSE is always ≥ MAE; the gap between them tells you how skewed your errors are.

### 11.3 R² — Coefficient of Determination

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

The numerator is the sum of squared errors from our model; the denominator is the sum of squared errors from a "trivial" model that always predicts the mean price. So:

- **R² = 1.0** → perfect predictions.
- **R² = 0** → the model is no better than guessing the average price for everyone.
- **R² < 0** → the model is *worse* than the trivial mean predictor (yes, this can happen on test data).

R² is unitless, so it's the right metric for comparing models *across* problems.

---

## 12. Results & interpretation

After training on **240,122 rows** and testing on **60,031 rows**, the model achieves:

| Metric | Value | What it means |
|---|---|---|
| **R²** | **0.986** | The model explains 98.6% of the variation in flight prices. |
| **MAE** | **₹1,092** | On average, predictions are within ~₹1,092 of the true fare. |
| **RMSE** | **₹2,698** | Squared-error magnitude; larger than MAE because of occasional bigger misses (business-class outliers). |

### Sanity check
Ticket prices in the dataset range roughly from ₹1,000 (cheap economy) to ₹100,000+ (last-minute business). A MAE of ₹1,092 is essentially "spot on" for economy and a small fraction of price for business. R² ≈ 0.986 is excellent — but we should *not* claim "98.6% accuracy," which is a classification term. The correct phrasing is "the model explains 98.6% of the variance."

### Why the model is this good
Most price variation in this dataset is driven by:
1. **Travel class** — economy vs. business — which alone splits the dataset cleanly into two price clusters.
2. **Days left** — last-minute fares are systematically higher.
3. **Airline** — Vistara/Air_India business cabins are pricier than budget carriers.

A Random Forest captures these effects and their interactions natively, which is why we hit R² in the 0.98 range even with a fairly small feature set.

---

## 13. The Streamlit demo app

**File:** `app/streamlit_app.py`. Launched with `streamlit run app/streamlit_app.py`, it serves a local web page on `http://localhost:8501`.

### 13.1 Loading the trained model

```python
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(METADATA_PATH)
```

- `@st.cache_resource` tells Streamlit to load the model **once** and reuse it across user interactions — without this, the model would re-load on every form submit, costing seconds.
- Loading is gated on the files existing; if they don't, the app shows a friendly error explaining how to train the model. (Good UX = predictable failure.)

### 13.2 Sidebar: surfacing the model's quality

```python
with st.sidebar:
    st.metric("R²", f"{metrics['r2']:.3f}")
    st.metric("RMSE", f"{metrics['rmse']:,.0f}")
    st.metric("MAE", f"{metrics['mae']:,.0f}")
```

The app does not just predict — it tells the user **how trustworthy the prediction is** by displaying the test-set metrics from the saved metadata. This is honesty by design.

### 13.3 The form: collecting inputs

The dropdowns are populated from `metadata["category_options"]`, which means the UI can never offer a category the model doesn't understand. Numeric inputs use sensible ranges (`duration` clamped to 0.5–50 hours, `days_left` slider 1–49).

A small validation check prevents the user from picking the same source and destination:

```python
if source_city == destination_city:
    st.warning("Source and destination must be different cities.")
```

### 13.4 Predicting

```python
row = pd.DataFrame([{ ... }], columns=FEATURES)
prediction = float(model.predict(row)[0])
```

The form values are packed into a single-row DataFrame with the **exact column order** of `FEATURES`. Because we saved a full `Pipeline` (not just the model), `model.predict(row)` automatically:
1. One-hot encodes the categoricals with the very same encoder used at training time.
2. Concatenates the numeric features in the same order.
3. Runs the Random Forest forward pass.

This is the payoff of using a `Pipeline` instead of doing the encoding manually.

### 13.5 Showing uncertainty

```python
lo = max(0.0, prediction - metrics["mae"])
hi = prediction + metrics["mae"]
st.success(f"Estimated fare: **₹ {prediction:,.0f}**")
st.caption(f"Typical range (± MAE): ₹ {lo:,.0f} – ₹ {hi:,.0f}")
```

A single point estimate can be misleading. Showing `prediction ± MAE` communicates the typical error and is far more honest than a single bold number.

---

## 14. How to run the project

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Kaggle API credentials (one-time)
#    Get kaggle.json from https://www.kaggle.com → Account → Settings → API
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Download the dataset (~5 MB)
python -m src.data_loader

# 5. Train the model (≈30s–2min on a laptop)
python -m src.train

# 6. Launch the demo
streamlit run app/streamlit_app.py
```

After step 6 the browser opens automatically at `http://localhost:8501`.

---

## 15. Limitations & possible extensions

**Limitations of the current project**
- The dataset is from a single time window (snapshot from Kaggle). Real airline pricing changes seasonally — the model would drift if deployed.
- We don't try a hold-out by *date*; a future flight may be more correlated with another future flight than with a past one (data leakage risk).
- The `flight` column is dropped — we lose per-flight effects (some specific flights might be systematically pricier).
- No hyperparameter search — `n_estimators` and `min_samples_leaf` were chosen by hand.

**Concrete extensions to push the project further**
1. **Model comparison.** Train Linear Regression, Decision Tree, Random Forest, Gradient Boosting (XGBoost / LightGBM) and compare metrics in a table. Random Forest is usually competitive but XGBoost often wins by a hair.
2. **Hyperparameter tuning.** Add `GridSearchCV` or `RandomizedSearchCV` on `n_estimators`, `max_depth`, `min_samples_leaf`.
3. **Feature engineering.** Add `route = source_city + "→" + destination_city` as a single categorical, log-transform `duration`, bucket `days_left` into "last-minute / 1-week / 1-month / advance".
4. **Cross-validation.** Replace the single train/test split with 5-fold CV and report mean ± std.
5. **Feature importance plot.** Use `pipeline.named_steps["model"].feature_importances_` and SHAP values to explain *why* the model predicts a given fare.
6. **Confidence intervals.** Replace `± MAE` with proper quantile regression or RF prediction intervals.
7. **Deployment.** Push the Streamlit app to Streamlit Community Cloud or Hugging Face Spaces so the teacher can click a link and try it.

---

## 16. Viva / FAQ — likely questions a teacher will ask

**Q: Is this supervised or unsupervised learning?**
A: Supervised. We have labelled examples — for each flight in the training set we know both the features and the actual price.

**Q: Why regression and not classification?**
A: The target `price` is a continuous numeric value, not a discrete category. Regression is the family of supervised algorithms that predict numbers.

**Q: Why Random Forest and not Linear Regression?**
A: Flight prices are strongly non-linear in `days_left`, depend on interactions (e.g. `class × airline`), and have heteroskedastic noise. A linear model can't capture these patterns. Random Forest handles them natively without any feature engineering, and trees are scale-invariant so we don't need to standardise inputs.

**Q: What is one-hot encoding and why do you use it?**
A: It converts each categorical column into a set of binary columns — one per category. We need it because the model only understands numbers, and using a single integer label (`SpiceJet=0, Vistara=1`) would falsely imply an ordering between airlines.

**Q: What is a Pipeline and why is it important?**
A: A `Pipeline` chains preprocessing steps and the model into one object. It guarantees that the **exact same encoding** is applied at training and at prediction time. Without it, you can easily get a mismatch between training and inference, which silently produces wrong predictions.

**Q: Why a 80/20 train/test split with `random_state=42`?**
A: 80/20 is a common rule of thumb that leaves enough data for both learning and evaluation. `random_state=42` makes the split reproducible — same rows every run, so metrics are comparable.

**Q: How do you know the model isn't overfitting?**
A: All reported metrics (RMSE, MAE, R²) are computed on the held-out test set the model never trained on. R² of 0.986 on unseen data demonstrates strong generalisation. We could go further with cross-validation.

**Q: What does R² = 0.986 mean exactly?**
A: The model explains 98.6% of the variance in price compared to a trivial baseline that always predicts the mean. It is **not** accuracy; saying "98.6% accuracy" would be wrong terminology.

**Q: What does MAE = ₹1,092 mean?**
A: On average, the absolute difference between the predicted price and the real price is about ₹1,092.

**Q: Why is RMSE higher than MAE?**
A: Because RMSE squares the errors before averaging, so big misses (mostly business-class outliers) pull RMSE up. The gap (~₹2,700 vs. ~₹1,100) tells us the error distribution has a heavy tail.

**Q: What hyperparameters did you tune?**
A: `n_estimators=120`, `min_samples_leaf=2`, `max_depth=None`, `random_state=42`. They were chosen by reasoning about bias/variance; a future improvement is to run a proper grid search.

**Q: Why did you drop the `flight` column?**
A: It has very high cardinality (essentially one value per flight) — including it would let the model memorise specific flights without generalising.

**Q: How would you deploy this to production?**
A: Containerise the Streamlit app with Docker, host on Streamlit Community Cloud or a small VPS, schedule monthly re-training on fresh Kaggle data, monitor MAE drift on the live distribution.

---

## 17. Glossary

- **Bagging** — Bootstrap Aggregating; training many models on different random subsets and averaging.
- **Bias / variance** — Bias = systematic error from over-simplifying. Variance = sensitivity to the particular training set.
- **Categorical feature** — A feature whose values are labels with no inherent order or arithmetic.
- **ColumnTransformer** — sklearn utility that applies different preprocessing to different columns.
- **Dense / sparse array** — Sparse arrays only store non-zero entries. One-hot data is mostly zeros, so sparse can save memory.
- **Estimator** — Any object in sklearn that implements `.fit` (and usually `.predict`).
- **Feature** — A column used as input to the model.
- **Generalisation** — How well a model performs on data it didn't see during training.
- **Hyperparameter** — A setting *of the model* (e.g. number of trees) chosen before training, as opposed to weights learned during training.
- **MAE / RMSE / R²** — See Section 11.
- **One-hot encoding** — See Section 8.1.
- **Overfitting** — When a model fits training noise rather than signal; great training score, poor test score.
- **Pipeline** — sklearn object that chains preprocessing + model into a single fit/predict-able object.
- **Regression** — Supervised learning where the target is a continuous number.
- **Reproducibility** — Same code + same `random_state` → same results.
- **Supervised learning** — Learning from labelled examples `(X, y)`.
- **Target** — The variable being predicted (here, `price`).
- **Train/test split** — Holding out a subset of the data to evaluate generalisation.

---

*Author: Dipanjan*
*Course: Machine Learning*
*Project: Flight Fare Predictor (regression, Random Forest, Streamlit demo)*
