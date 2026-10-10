"""Phase 5 — Classification: high-demand session prediction.

Target rule (exact, from brief):
    High Demand   = Demand >= median(Demand_train)
    Normal Demand = Demand <  median(Demand_train)
Threshold is computed on TRAIN only, then applied to test (no leakage).
Compares Logistic Regression, Decision Tree, Random Forest, KNN, SVM,
XGBoost, LightGBM. If class ratio worse than ~60/40, compares Random
Oversampling, SMOTE, ADASYN on TRAIN only.
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "processed" / "modeling_train.csv"
TEST_PATH = BASE / "data" / "processed" / "modeling_test.csv"
MODEL_DIR = BASE / "models"
REPORT_PATH = BASE / "reports" / "classification_results.csv"
THRESHOLD_PATH = BASE / "data" / "processed" / "classification_threshold.pkl"


def load_xy():
    """Load features + Demand series for both splits."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    drop = ["Demand", "_StartDatetime"]
    return (
        train.drop(columns=drop), train["Demand"],
        test.drop(columns=drop), test["Demand"],
    )


def get_models():
    """Seven required classifiers (scaled where needed)."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=12),
        "random_forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        # RBF SVM on a large set is slow; linear-ish SVC with scaled data.
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True)),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            eval_metric="logloss", random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=42, verbose=-1,
        ),
    }


def clf_metrics(y_true, y_pred, y_proba):
    """Accuracy, precision, recall, F1, ROC-AUC."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def main():
    """Label via train median, train/compare, resample if imbalanced."""
    X_train, d_train, X_test, d_test = load_xy()

    # Median from TRAINING set only — leakage guard.
    threshold = float(d_train.median())
    y_train = (d_train >= threshold).astype(int)
    y_test = (d_test >= threshold).astype(int)
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump({"median_threshold": threshold}, f)

    train_balance = y_train.value_counts(normalize=True).sort_index()
    test_balance = y_test.value_counts(normalize=True).sort_index()
    print(f"Threshold (train median Demand): {threshold:.2f}")
    print(f"Train balance (0=Normal,1=High): {train_balance.to_dict()}")
    print(f"Test balance: {test_balance.to_dict()}")
    worst_ratio = max(train_balance.max(), test_balance.max())
    needs_resampling = worst_ratio > 0.60
    print(f"Needs resampling (>60/40)? {needs_resampling}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows, fitted, probas = [], {}, {}
    for name, model in get_models().items():
        # SVM on 58k rows is very slow — subsample 15k for fit only.
        if name == "svm":
            sub = X_train.sample(n=15000, random_state=42)
            model.fit(sub, y_train.loc[sub.index])
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        m = clf_metrics(y_test, pred, proba)
        rows.append({"model": name, "sampling": "none", **m})
        fitted[name] = model
        probas[name] = proba
        with open(MODEL_DIR / f"{name}_classifier.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"{name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    # Resampling comparison on TRAIN only (using a fast RF probe).
    if needs_resampling:
        samplers = {
            "random_oversample": RandomOverSampler(random_state=42),
            "smote": SMOTE(random_state=42),
            "adasyn": ADASYN(random_state=42),
        }
        for sname, sampler in samplers.items():
            Xr, yr = sampler.fit_resample(X_train, y_train)
            probe = RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=42
            )
            probe.fit(Xr, yr)
            pred = probe.predict(X_test)
            proba = probe.predict_proba(X_test)[:, 1]
            m = clf_metrics(y_test, pred, proba)
            rows.append({"model": f"random_forest+{sname}", "sampling": sname, **m})
            print(f"RF+{sname}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    else:
        print("Classes ~balanced (median split); no resampling applied.")

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(REPORT_PATH, index=False)
    # Best = highest ROC-AUC among the seven primary models.
    primary = results[results["sampling"] == "none"].iloc[0]
    print(f"\nBest classifier: {primary['model']} "
          f"(acc={primary['accuracy']:.4f}, f1={primary['f1']:.4f}, "
          f"auc={primary['roc_auc']:.4f})")
    with open(MODEL_DIR / "best_classifier.pkl", "wb") as f:
        pickle.dump(fitted[primary["model"]], f)

    best = fitted[primary["model"]]
    # Feature association for high-demand sessions.
    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X_train.columns)
        print("\nTop 10 high-demand features:")
        print(imp.sort_values(ascending=False).head(10).to_string())
        imp.sort_values(ascending=False).to_csv(
            BASE / "reports" / "classification_feature_importance.csv"
        )
    elif hasattr(best, "named_steps"):
        # logistic pipeline: expose coefficients.
        lr = best.named_steps["logisticregression"]
        coef = pd.Series(np.abs(lr.coef_[0]), index=X_train.columns)
        print("\nTop 10 |coefficients| (logistic):")
        print(coef.sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
