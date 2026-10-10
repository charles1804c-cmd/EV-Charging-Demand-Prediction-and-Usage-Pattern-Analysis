# Key Findings — EV Charging Demand Prediction

## 1. Data & cleaning
- 72,856 rows -> **72,826** after dropping 13 exact duplicates and 17
  negative-Duration rows (EndDatetime earlier than StartDatetime; dropped,
  not silently kept — recomputing would just echo the erroneous datetimes).
- No missing values (re-verified in code). Date/time columns parsed to pandas datetime.
- Demand right-skewed: mean 17.44, median 14.10, max 97, skew 1.19.
  Log-target experiment run (RF R² 0.758 < baseline 0.792) — primary metrics
  reported on the original scale.
- Imbalanced categoricals: ChargerType 0/1 = 57,896/14,930; Location
  "bus garage" 29 rows, "accommodation" 187 rows -> grouped into
  `other_rare` (< 500-row threshold); one-hot encoded (train-fit only).

## 2. EDA
- Duration-Demand correlation **0.66** (strongest numeric signal).
- ChargerType 1 mean 18.95 > Type 0 mean 17.05; Company 1 (17.89) > Company 0 (16.69).
- Highest mean demand: accommodation (29.1), bus garage (29.6, n=29),
  camping (25.5); lowest: hotel (14.7), sightseeing (15.2).
- One year of data: 2021-09-30 -> 2022-09-30. Evening peak hours; daily
  trend with weekly seasonality. Figures in `reports/figures/`.

## 3. Features & split
- Time: hour, day, month, day_of_week, is_weekend, time_of_day bucket.
- History (leakage-safe, past-only expanding means with shift(1)):
  user_past_avg_demand, user_session_count, charger_past_avg_demand,
  charger_session_count; first-seen fall back to global mean.
- **Time-aware 80/20 split** (train <= 2022-07-30, test = recent period):
  chronological order prevents future-pattern leakage. 33 features.

## 4. Regression (target: Demand)
| model | MAE | RMSE | R² |
|---|---|---|---|
| LightGBM | 3.09 | 6.04 | 0.812 |
| XGBoost | 3.20 | 6.15 | 0.805 |
| Random Forest | 3.49 | 6.34 | 0.792 |
| Gradient Boosting | 3.62 | 6.59 | 0.776 |
| Decision Tree | 3.61 | 7.31 | 0.724 |
| Linear Regression | 6.05 | 9.82 | 0.502 |
| RF log-target (exp) | 3.94 | 6.85 | 0.758 |

Best: **LightGBM** (lowest RMSE, highest R²). Top features: Duration,
UserID, ChargerID, user/charger past-avg demand, session counts.

## 5. Classification (High = Demand >= train median 14.20)
- Train balance 50.0/50.0, test 51.2/48.8 -> balanced, **no resampling
  applied** (criterion was >60/40; RandomOver/SMOTE/ADASYN correctly skipped).
| model | acc | F1 | ROC-AUC |
|---|---|---|---|
| XGBoost | 0.926 | 0.927 | 0.973 |
| LightGBM | 0.924 | 0.925 | 0.973 |
| Random Forest | 0.923 | 0.923 | 0.971 |
| Decision Tree | 0.900 | 0.899 | 0.904 |
| SVM | 0.865 | 0.865 | 0.927 |
| Logistic | 0.857 | 0.859 | 0.909 |
| KNN | 0.784 | 0.785 | 0.866 |

Best: **XGBoost**. High-demand associated with long Duration,
high user past-avg demand, ChargerType/company dummies.

## 6. Time series
- Hourly + daily total-demand series built. Daily ADF p=0.34 ->
  **non-stationary** (trend/seasonality); hourly ADF p<1e-9 -> stationary.
- Last-30-day holdout: Moving-Average-7d MAE 581/RMSE 892 beats
  Holt-Winters MAE 721/RMSE 973 (simple baseline wins on this window).
- Held-out demand **rising** (first-half mean 3518, second-half 3895);
  peak day 2022-08-10 (5879.9).

## 7. Explainability
- SHAP global + local (waterfall) for both best models; LIME cross-check
  agrees: Duration >> past-usage averages >> charger type/company.
- Example test row: actual 49.02, regressor 51.69, classifier High (correct).
- Plain language: **long sessions by historically heavy users/chargers
  drive high demand**; where/when matters less.
