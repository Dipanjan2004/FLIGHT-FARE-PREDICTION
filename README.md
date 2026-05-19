# Flight Fare Prediction

A supervised regression project that predicts the price of an Indian domestic
flight ticket from nine features (airline, source and destination city,
departure and arrival time-of-day, number of stops, travel class, flight
duration, and days remaining until departure).

The project compares three regression algorithms — Linear Regression, Random
Forest, and Histogram Gradient Boosting — with five-fold cross-validation,
tunes the best candidate with `RandomizedSearchCV`, and ships the trained
pipeline behind a Streamlit web application for interactive prediction.

**Course:** Introduction to Machine Learning (24ETAEC49)
**Institute:** Ramaiah Institute of Technology, Department of Electronics & Telecommunication Engineering
**Authors:** Dipanjan Chowdhury (1MS24ET015), Devansh Gupta (1MS24ET012)

---

## Repository layout

```
ML_project/
├── app/
│   └── streamlit_app.py          Streamlit web application
├── src/
│   ├── config.py                 Paths, feature lists, target column
│   ├── data_loader.py            Kaggle dataset download + cached load
│   ├── preprocess.py             ColumnTransformer (one-hot + passthrough)
│   ├── eda.py                    Exploratory data analysis plots
│   └── train.py                  Multi-model comparison, tuning, persistence
├── data/                         Raw dataset (downloaded, gitignored)
├── models/                       Trained pipeline + metadata (gitignored)
├── reports/                      Generated PNG plots (EDA + diagnostics)
├── requirements.txt
├── ML_Report.docx                Project report (formal submission)
└── README.md
```

---

## Dataset

[Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
by Shubham Bathwal (Kaggle).

- **Rows:** 300,153 Indian domestic flight records
- **Features used:** 7 categorical (airline, source\_city, departure\_time, stops, arrival\_time, destination\_city, class) + 2 numeric (duration, days\_left)
- **Target:** `price` (continuous, in Rupees)
- **Excluded:** `flight` (per-flight code, high cardinality, would cause memorisation)

---

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Kaggle API credentials

The dataset is downloaded automatically via the Kaggle CLI. A one-time
credential setup is needed:

1. Sign in to <https://www.kaggle.com>, go to **Account → Settings → API → Create New Token**. A `kaggle.json` file is downloaded.
2. Move it to `~/.kaggle/` and restrict its permissions:

   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Reproducing the results

Run from the project root:

```bash
# 1. Download the dataset (idempotent — skipped if already present)
python -m src.data_loader

# 2. Generate exploratory plots (saved to reports/01_… 06_…)
python -m src.eda

# 3. Train and compare the three models, tune the best, persist artefacts
python -m src.train

# 4. Launch the interactive demo
streamlit run app/streamlit_app.py
```

Step 3 also writes the model-comparison and feature-importance plots
(`reports/07_model_comparison.png`, `reports/08_feature_importance.png`) and
saves the trained pipeline to `models/flight_fare_model.joblib` together with a
metadata sidecar (`models/model_metadata.joblib`).

---

## Methodology

1. **Train / test split** — 80 / 20, fixed random seed (42), 240,122 train rows and 60,031 test rows.
2. **Preprocessing** — `sklearn.compose.ColumnTransformer` performs one-hot encoding on the seven categorical columns (`handle_unknown="ignore"`) and passes the two numeric columns through unchanged. Preprocessing and the regressor are bundled into a single `sklearn.pipeline.Pipeline`, so the same transformations are applied during training, cross-validation, hyperparameter search, and live prediction.
3. **Model comparison** — Linear Regression, Random Forest (`n_estimators=80, min_samples_leaf=2`), and Histogram Gradient Boosting (`max_iter=300, learning_rate=0.1, min_samples_leaf=20`) are evaluated with 5-fold `KFold` cross-validation, scored by `neg_root_mean_squared_error`.
4. **Hyperparameter tuning** — the best model (Random Forest) is tuned with `RandomizedSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features` (10 random combinations, 3-fold CV).
5. **Evaluation** — final metrics are reported on the held-out test set.

---

## Results

Test-set performance (held-out 60,031 rows):

| Model                            | CV RMSE | Test RMSE | Test MAE | Test R² |
| -------------------------------- | ------: | --------: | -------: | ------: |
| Linear Regression                |   6,753 |     6,762 |    4,553 |  0.9113 |
| Random Forest                    |   2,699 |     2,700 |    1,094 |  0.9859 |
| Histogram Gradient Boosting      |   3,447 |     3,449 |    2,002 |  0.9769 |
| **Random Forest (tuned, final)** |   2,718 | **2,693** |    1,113 |  0.9859 |

The final tuned Random Forest explains 98.6 % of the variance in flight fares.
Feature-importance analysis (`reports/08_feature_importance.png`) confirms
travel class, days left until departure, and flight duration as the dominant
predictors, which is consistent with airline-pricing intuition.

Tuned hyperparameters: `n_estimators=200, max_depth=35, min_samples_leaf=2, max_features=0.7`.

---

## Streamlit demo

The web application loads the persisted pipeline at startup and provides a
form for selecting the nine input features. It returns the predicted fare
together with a ±MAE uncertainty band, and exposes the model-comparison table
and tuned hyperparameters in the sidebar for transparency.

```bash
streamlit run app/streamlit_app.py
# Opens http://localhost:8501
```

---

## Dependencies

Pinned (minimum) versions in `requirements.txt`:

- `pandas >= 2.0`
- `numpy >= 1.24`
- `scikit-learn >= 1.4`
- `joblib >= 1.3`
- `kaggle >= 1.6`
- `streamlit >= 1.32`
- `matplotlib >= 3.8`
- `seaborn >= 0.13`

---

## References

- Bathwal, S. *Flight Price Prediction.* Kaggle dataset.
  <https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction>
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45 (1), 5–32.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.*
  Journal of Machine Learning Research, 12, 2825–2830.
