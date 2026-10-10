"""Phase 4 — Regression: charging demand prediction.

Compares Linear Regression, Decision Tree, Random Forest, Gradient
Boosting, XGBoost, LightGBM on the time-aware split. Metrics MAE/MSE/
RMSE/R2 on the ORIGINAL demand scale. Includes a log-target experiment
(Demand is right-skewed) with inverse-transformed reporting.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "processed" / "modeling_train.csv"
TEST_PATH = BASE / "data" / "processed" / "modeling_test.csv"
MODEL_DIR = BASE / "models"
REPORT_PATH = BASE / "reports" / "regression_results.csv"


def load_xy():
    """Load train/test matrices (drops helper datetime col)."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    drop = ["Demand", "_StartDatetime"]
    X_train = train.drop(columns=drop)
    y_train = train["Demand"]
    X_test = test.drop(columns=drop)
    y_test = test["Demand"]
    return X_train, y_train, X_test, y_test


def get_models():
    """Return the six required regressors with fixed seeds."""
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42, max_depth=12),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "xgboost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=42, verbose=-1,
        ),
    }


def evaluate(y_true, y_pred):
    """Compute MAE, MSE, RMSE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    """Train, evaluate, save models + comparison table + best model."""
    X_train, y_train, X_test, y_test = load_xy()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    fitted = {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.clip(pred, 0, None)  # demand cannot be negative
        metrics = evaluate(y_test, pred)
        rows.append({"model": name, **metrics})
        fitted[name] = model
        with open(MODEL_DIR / f"{name}_regressor.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"{name}: MAE={metrics['MAE']:.3f} RMSE={metrics['RMSE']:.3f} R2={metrics['R2']:.4f}")

    # Log-transform experiment (random forest on log1p target, expm1 back).
    log_model = RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=42
    )
    log_model.fit(X_train, np.log1p(y_train))
    log_pred = np.expm1(log_model.predict(X_test)).clip(min=0)
    log_metrics = evaluate(y_test, log_pred)
    rows.append({"model": "random_forest_log_target", **log_metrics})
    print(f"random_forest_log_target: MAE={log_metrics['MAE']:.3f} "
          f"RMSE={log_metrics['RMSE']:.3f} R2={log_metrics['R2']:.4f}")

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)

    # Best model = lowest RMSE among the six primary (excl. log experiment
    # unless it wins — here selection is documented either way).
    primary = results[results["model"] != "random_forest_log_target"].iloc[0]
    best_name = primary["model"]
    print(f"\nBest regressor: {best_name} (RMSE={primary['RMSE']:.3f}, R2={primary['R2']:.4f})")
    with open(MODEL_DIR / "best_regressor.pkl", "wb") as f:
        pickle.dump(fitted[best_name], f)
    # Feature importance for best tree-based model.
    best = fitted[best_name]
    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X_train.columns)
        print("\nTop 10 features:")
        print(imp.sort_values(ascending=False).head(10).to_string())
        imp.sort_values(ascending=False).to_csv(
            BASE / "reports" / "regression_feature_importance.csv"
        )


if __name__ == "__main__":
    main()
