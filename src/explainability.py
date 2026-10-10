"""Phase 7 — Model explainability with SHAP and LIME.

SHAP global summary + one local explanation for the best regressor
(LightGBM) and best classifier (XGBoost); LIME as a secondary
cross-check on one prediction from each. Figures saved to
reports/figures/ for embedding in the dashboard.
"""

from pathlib import Path
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

BASE = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE / "data" / "processed" / "modeling_train.csv"
TEST_PATH = BASE / "data" / "processed" / "modeling_test.csv"
FIG_DIR = BASE / "reports" / "figures"

warnings.filterwarnings("ignore")


def load():
    """Load matrices, models, and classification threshold."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    drop = ["Demand", "_StartDatetime"]
    X_train, X_test = train.drop(columns=drop), test.drop(columns=drop)
    y_reg = test["Demand"]
    with open(BASE / "models" / "best_regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(BASE / "models" / "best_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(BASE / "data" / "processed" / "classification_threshold.pkl", "rb") as f:
        threshold = pickle.load(f)["median_threshold"]
    y_clf = (test["Demand"] >= threshold).astype(int)
    return X_train, X_test, y_reg, y_clf, reg, clf


def shap_global_local(model, X_bg, X_sample, prefix: str, model_type: str):
    """Save SHAP beeswarm (global) + waterfall (local) plots."""
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X_sample)
    if isinstance(vals, list):  # binary classifier returns per-class list
        vals = vals[1]
    vals = np.asarray(vals)
    # Global summary (beeswarm).
    plt.figure()
    shap.summary_plot(vals, X_sample, show=False, max_display=15)
    plt.title(f"SHAP summary — {prefix}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_summary_{prefix}.png", bbox_inches="tight")
    plt.close()
    # Local waterfall for the first sampled row.
    exp = shap.Explanation(
        values=vals[0], base_values=explainer.expected_value
        if not isinstance(explainer.expected_value, (list, np.ndarray))
        else (explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value),
        data=X_sample.iloc[0].values, feature_names=X_sample.columns.tolist(),
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=12)
    plt.title(f"SHAP local explanation — {prefix} (row 0)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_local_{prefix}.png", bbox_inches="tight")
    plt.close()
    top = pd.Series(np.abs(vals).mean(axis=0), index=X_sample.columns)
    print(f"\nTop SHAP features — {prefix}:")
    print(top.sort_values(ascending=False).head(8).to_string())
    return top


def lime_check(X_train, model, instance, prefix: str, mode: str):
    """LIME explanation for a single instance; save figure + print weights."""
    explainer = LimeTabularExplainer(
        X_train.values, feature_names=X_train.columns.tolist(),
        mode=mode, random_state=42, discretize_continuous=True,
    )
    fn = model.predict_proba if mode == "classification" else model.predict
    exp = explainer.explain_instance(instance.values, fn, num_features=8)
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"lime_{prefix}.png", bbox_inches="tight")
    plt.close("all")
    print(f"\nLIME weights — {prefix} ({mode}):")
    for feat, w in exp.as_list():
        print(f"  {feat}: {w:+.4f}")


def main():
    """Run SHAP + LIME for both best models and summarize in plain language."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_reg, y_clf, reg, clf = load()
    # Subsamples keep TreeExplainer fast yet representative.
    bg = X_train.sample(n=500, random_state=42)
    sample = X_test.sample(n=500, random_state=42)

    shap_global_local(reg, bg, sample, "regression", "regression")
    shap_global_local(clf, bg, sample, "classification", "classification")

    lime_check(X_train, reg, X_test.iloc[0], "regression", "regression")
    lime_check(X_train, clf, X_test.iloc[0], "classification", "classification")

    pred_r = float(reg.predict(X_test.iloc[[0]])[0])
    pred_c = clf.predict(X_test.iloc[[0]])[0]
    print(f"\nExample row 0: actual Demand={float(y_reg.iloc[0]):.2f}, "
          f"regressor predicts {pred_r:.2f}; classifier predicts "
          f"{'High' if pred_c == 1 else 'Normal'} (actual "
          f"{'High' if int(y_clf.iloc[0]) == 1 else 'Normal'})")
    print("\nPlain-language summary: Duration and past-usage history "
          "(user/charger average demand) dominate both demand level and "
          "high-demand classification; charger type and company add signal.")


if __name__ == "__main__":
    main()
