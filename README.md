# EV Charging Demand Prediction and Usage Pattern Analysis

Electric-vehicle charging demand prediction with usage-pattern analysis:
regression (demand), classification (high-demand sessions), time-series
forecasting, SHAP/LIME explainability, and a Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce (in order — do not skip or reorder)

```bash
python src/data_cleaning.py        # Phase 1 -> data/processed/charging_cleaned.csv
# Phase 2: open notebooks/02_eda.ipynb (figures in reports/figures/)
python src/feature_engineering.py  # Phase 3 -> modeling_train/test.csv
python src/train_regression.py     # Phase 4 -> models/*_regressor.pkl, reports/regression_results.csv
python src/train_classification.py # Phase 5 -> models/*_classifier.pkl, reports/classification_results.csv
python src/time_series.py          # Phase 6 -> reports/time_series_results.csv
python src/explainability.py       # Phase 7 -> reports/figures/shap_*.png, lime_*.png
streamlit run dashboard/app.py     # Phase 8: dashboard
```

Notebooks `notebooks/01_...07_....ipynb` call into `src/` rather than
duplicating logic; run them top to bottom after the matching `src/` script.

## Key results

- Regression best: **LightGBM** — MAE 3.09, RMSE 6.04, R² 0.81.
- Classification best: **XGBoost** — acc 0.926, F1 0.927, ROC-AUC 0.973
  (threshold = train median Demand 14.20; classes ~50/50, no resampling needed).
- Time series: daily series non-stationary (ADF p=0.34); 7-day moving
  average (MAE 581) beat Holt-Winters (MAE 721) on the last-30-day window;
  held-out demand was rising; peak day 2022-08-10.
- Drivers: **Duration** + user/charger past-average demand dominate both
  tasks; charger type/company secondary. See `reports/key_findings.md`.

## Structure

```
data/raw/ChargingRecords.csv  data/processed/  notebooks/01_..07_
src/data_cleaning|feature_engineering|train_regression|train_classification|time_series|explainability.py
models/  dashboard/app.py  reports/  requirements.txt  README.md
```
