"""Streamlit dashboard: EV Charging Demand Prediction & Usage Analysis.

Tabs: Overview | Usage Patterns | Regression | Classification |
Time Series | Explainability. Trained models are loaded via pickle;
SHAP plots are precomputed images (never computed per-request).
"""

from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
CLEANED_PATH = BASE / "data" / "processed" / "charging_cleaned.csv"
TEST_PATH = BASE / "data" / "processed" / "modeling_test.csv"
FIG_DIR = BASE / "reports" / "figures"

st.set_page_config(page_title="EV Charging Demand", layout="wide")
st.title("EV Charging Demand Prediction & Usage Pattern Analysis")


@st.cache_data
def load_cleaned() -> pd.DataFrame:
    """Load cleaned sessions with parsed datetimes."""
    return pd.read_csv(CLEANED_PATH, parse_dates=["StartDatetime", "EndDatetime"])


@st.cache_resource
def load_models():
    """Load pickled best models + artifacts + threshold (cached)."""
    with open(BASE / "models" / "best_regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    with open(BASE / "models" / "best_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(BASE / "data" / "processed" / "feature_artifacts.pkl", "rb") as f:
        arts = pickle.load(f)
    with open(BASE / "data" / "processed" / "classification_threshold.pkl", "rb") as f:
        thresh = pickle.load(f)["median_threshold"]
    return reg, clf, arts, thresh


df = load_cleaned()
reg, clf, arts, threshold = load_models()
FEATURES: list = arts["feature_columns"]

# ---- Shared sidebar filters ----
st.sidebar.header("Filters")
locs = sorted(df["Location"].unique().tolist())
sel_locs = st.sidebar.multiselect("Location", locs, default=locs)
sel_type = st.sidebar.multiselect("ChargerType", [0, 1], default=[0, 1])
sel_comp = st.sidebar.multiselect("ChargerCompany", [0, 1], default=[0, 1])
dmin, dmax = df["StartDatetime"].min().date(), df["StartDatetime"].max().date()
date_range = st.sidebar.date_input("Date range", [dmin, dmax])
mask = (
    df["Location"].isin(sel_locs)
    & df["ChargerType"].isin(sel_type)
    & df["ChargerCompany"].isin(sel_comp)
)
if len(date_range) == 2:
    mask &= (df["StartDatetime"].dt.date >= date_range[0]) & (
        df["StartDatetime"].dt.date <= date_range[1]
    )
fdf = df[mask]

tabs = st.tabs([
    "Overview", "Usage Patterns", "Regression",
    "Classification", "Time Series", "Explainability",
])

# 1. Overview
with tabs[0]:
    st.header("Overview — usage KPIs")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sessions", f"{len(fdf):,}")
    c2.metric("Avg demand", f"{fdf['Demand'].mean():.2f}")
    c3.metric("Avg duration (min)", f"{fdf['Duration'].mean():.1f}")
    c4.metric("Unique users", f"{fdf['UserID'].nunique():,}")
    c5.metric("Unique chargers", f"{fdf['ChargerID'].nunique():,}")
    st.dataframe(fdf.describe(numeric_only=True))

# 2. Usage Patterns
with tabs[1]:
    st.header("Usage Patterns")
    st.subheader("Mean demand by location")
    st.bar_chart(fdf.groupby("Location")["Demand"].mean())
    st.subheader("Mean demand by charger type / company")
    c1, c2 = st.columns(2)
    c1.bar_chart(fdf.groupby("ChargerType")["Demand"].mean())
    c2.bar_chart(fdf.groupby("ChargerCompany")["Demand"].mean())
    st.subheader("Mean demand by hour of day")
    st.bar_chart(fdf.assign(hour=fdf["StartDatetime"].dt.hour).groupby("hour")["Demand"].mean())
    st.subheader("Daily mean demand trend")
    st.line_chart(fdf.set_index("StartDatetime").resample("D")["Demand"].mean())

# Helper: build one model-ready row from user inputs.
def build_feature_row(duration, hour, month, dow, userid, chargerid,
                      user_avg, charger_avg, location, company, ctype):
    """Assemble a single row aligned to training feature columns."""
    loc = location if location not in arts["rare_locations"] else "other_rare"
    tod = ("morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17
           else "evening" if 17 <= hour < 22 else "night")
    row = {
        "Duration": duration, "hour": hour, "day": 15, "month": month,
        "day_of_week": dow, "is_weekend": int(dow >= 5),
        "UserID": userid, "ChargerID": chargerid,
        "user_past_avg_demand": user_avg, "user_session_count": 5,
        "charger_past_avg_demand": charger_avg, "charger_session_count": 10,
    }
    for col in FEATURES:
        if col not in row:
            row[col] = 0.0
    for prefix, val in [("Location_grouped", loc), ("ChargerCompany", str(company)),
                        ("ChargerType", str(ctype)), ("time_of_day", tod)]:
        key = f"{prefix}_{val}"
        if key in row:
            row[key] = 1.0
    return pd.DataFrame([row])[FEATURES]


# 3. Regression
with tabs[2]:
    st.header("Regression — demand prediction")
    st.image(str(FIG_DIR / "pred_vs_actual.png"), caption="Predicted vs actual (test set)")
    st.table(pd.read_csv(BASE / "reports" / "regression_results.csv").round(4))
    st.subheader("Live demand prediction")
    duration = st.number_input("Duration (min)", 0, 1500, 120)
    hour = st.slider("Hour of day", 0, 23, 18)
    month = st.slider("Month", 1, 12, 7)
    dow = st.selectbox("Day of week (Mon=0)", list(range(7)), index=2)
    location = st.selectbox("Location", locs)
    company = st.selectbox("Company", [0, 1])
    ctype = st.selectbox("Charger type", [0, 1])
    user_avg = st.number_input("User past avg demand", 0.0, 100.0, 14.0)
    charger_avg = st.number_input("Charger past avg demand", 0.0, 100.0, 17.0)
    if st.button("Predict demand"):
        row = build_feature_row(duration, hour, month, dow, 0, 1,
                                user_avg, charger_avg, location, company, ctype)
        st.success(f"Predicted demand: {float(reg.predict(row)[0]):.2f}")

# 4. Classification
with tabs[3]:
    st.header("Classification — high/normal demand")
    st.write(f"Threshold (train median): **{threshold:.2f}** — "
             "High = Demand ≥ threshold.")
    st.table(pd.read_csv(BASE / "reports" / "classification_results.csv").round(4))
    st.subheader("Live high-demand prediction (same inputs as Regression tab)")
    if st.button("Predict high/normal"):
        row = build_feature_row(duration, hour, month, dow, 0, 1,
                                user_avg, charger_avg, location, company, ctype)
        proba = float(clf.predict_proba(row)[0, 1])
        label = "High Demand" if proba >= 0.5 else "Normal Demand"
        st.success(f"{label} (P(high)={proba:.3f})")

# 5. Time Series
with tabs[4]:
    st.header("Time Series — trend & forecast")
    st.image(str(FIG_DIR / "ts_daily_trend.png"))
    st.image(str(FIG_DIR / "ts_forecast.png"))
    st.table(pd.read_csv(BASE / "reports" / "time_series_results.csv").round(2))
    st.dataframe(pd.read_csv(BASE / "data" / "processed" / "demand_daily_forecast.csv"))

# 6. Explainability
with tabs[5]:
    st.header("Model Explainability (precomputed SHAP)")
    st.subheader("Regression — global")
    st.image(str(FIG_DIR / "shap_summary_regression.png"))
    st.subheader("Regression — local (single prediction)")
    st.image(str(FIG_DIR / "shap_local_regression.png"))
    st.subheader("Classification — global")
    st.image(str(FIG_DIR / "shap_summary_classification.png"))
    st.subheader("Classification — local (single prediction)")
    st.image(str(FIG_DIR / "shap_local_classification.png"))
    st.info("Duration and user/charger past-average demand drive both demand "
            "level and high-demand classification; charger type and company "
            "add secondary signal (see reports/key_findings.md).")
