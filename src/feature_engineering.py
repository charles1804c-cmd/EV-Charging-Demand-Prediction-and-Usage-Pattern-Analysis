"""Phase 3 — Feature Engineering (leakage-safe, time-aware).

Builds time features, grouped categoricals, historical usage features
using only past information, one-hot encoding fitted on TRAIN only,
and a chronological 80/20 train/test split.
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
CLEANED_PATH = BASE / "data" / "processed" / "charging_cleaned.csv"
TRAIN_PATH = BASE / "data" / "processed" / "modeling_train.csv"
TEST_PATH = BASE / "data" / "processed" / "modeling_test.csv"
ARTIFACT_PATH = BASE / "data" / "processed" / "feature_artifacts.pkl"

# Locations with < 500 sessions are grouped to handle extreme imbalance
# (bus garage=29, accommodation=187). Threshold documented in EDA.
RARE_THRESHOLD = 500


def time_of_day_bucket(hour: int) -> str:
    """Bucket hour into morning/afternoon/evening/night."""
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day, month, day_of_week, is_weekend, time_of_day."""
    df = df.copy()
    df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
    df["hour"] = df["StartDatetime"].dt.hour
    df["day"] = df["StartDatetime"].dt.day
    df["month"] = df["StartDatetime"].dt.month
    df["day_of_week"] = df["StartDatetime"].dt.dayofweek  # Mon=0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["time_of_day"] = df["hour"].apply(time_of_day_bucket)
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe past-only aggregates (sorted by StartDatetime).

    Uses expanding means shifted by 1 so each row only sees strictly
    earlier sessions for that user/charger — never its own or future rows.
    """
    df = df.copy().sort_values("StartDatetime").reset_index(drop=True)
    global_mean = df["Demand"].mean()
    # Per-user past average demand + session count so far.
    df["user_past_avg_demand"] = (
        df.groupby("UserID")["Demand"].transform(lambda s: s.expanding().mean().shift(1))
    )
    df["user_session_count"] = df.groupby("UserID").cumcount()  # 0 for first session
    # Per-charger historical utilization (past mean demand).
    df["charger_past_avg_demand"] = (
        df.groupby("ChargerID")["Demand"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    df["charger_session_count"] = df.groupby("ChargerID").cumcount()
    # First-seen users/chargers have no history: fall back to global mean / 0.
    df["user_past_avg_demand"] = df["user_past_avg_demand"].fillna(global_mean)
    df["charger_past_avg_demand"] = df["charger_past_avg_demand"].fillna(global_mean)
    df["_global_mean"] = global_mean  # stored for reference, dropped before modeling
    return df


def group_rare_locations(df: pd.DataFrame, threshold: int = RARE_THRESHOLD):
    """Group locations with fewer than `threshold` rows into 'other_rare'."""
    counts = df["Location"].value_counts()
    rare = set(counts[counts < threshold].index.tolist())
    df = df.copy()
    df["Location_grouped"] = df["Location"].apply(
        lambda x: "other_rare" if x in rare else x
    )
    return df, sorted(rare)


def build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline up to (but excluding) one-hot encoding."""
    df = add_time_features(cleaned)
    df = add_history_features(df)
    df, rare = group_rare_locations(df)
    df.attrs["rare_locations"] = rare
    return df


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2):
    """Chronological split, fit one-hot encoder on train only, transform both.

    Time-aware split: train on earliest (1-test_size) of sessions ordered by
    StartDatetime, test on the most recent period. Correct for sequential
    charging data — random splits would leak future patterns into training.
    """
    df = df.copy().sort_values("StartDatetime").reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    split_date = df.loc[split_idx, "StartDatetime"]

    cat_cols = ["Location_grouped", "ChargerCompany", "ChargerType", "time_of_day"]
    for col in ["ChargerCompany", "ChargerType"]:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

    # One-hot fitted on train categories only; unseen test cats ignored.
    train_dummies = pd.get_dummies(train[cat_cols], prefix=cat_cols, dtype=float)
    train_columns = train_dummies.columns.tolist()
    test_dummies = pd.get_dummies(test[cat_cols], prefix=cat_cols, dtype=float)
    test_dummies = test_dummies.reindex(columns=train_columns, fill_value=0.0)

    num_cols = [
        "Duration", "hour", "day", "month", "day_of_week", "is_weekend",
        "UserID", "ChargerID", "user_past_avg_demand", "user_session_count",
        "charger_past_avg_demand", "charger_session_count",
    ]
    X_train = pd.concat(
        [train[num_cols].astype(float), train_dummies], axis=1
    )
    X_test = pd.concat([test[num_cols].astype(float), test_dummies], axis=1)
    y_train = train["Demand"].astype(float)
    y_test = test["Demand"].astype(float)

    artifacts = {
        "feature_columns": X_train.columns.tolist(),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "train_columns_dummies": train_columns,
        "rare_locations": df.attrs.get("rare_locations", []),
        "global_demand_mean": float(df["_global_mean"].iloc[0]),
        "split_datetime": str(split_date),
        "test_size": test_size,
    }
    # Attach keys needed by downstream phases (ids + timestamps for TS).
    train_out = X_train.copy()
    train_out["Demand"] = y_train.values
    train_out["_StartDatetime"] = train["StartDatetime"].values
    test_out = X_test.copy()
    test_out["Demand"] = y_test.values
    test_out["_StartDatetime"] = test["StartDatetime"].values
    return train_out, test_out, artifacts


def main() -> None:
    """Build modeling datasets and save train/test + artifacts."""
    cleaned = pd.read_csv(CLEANED_PATH, parse_dates=["StartDatetime", "EndDatetime"])
    feat = build_features(cleaned)
    train_out, test_out, artifacts = encode_and_split(feat)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(TRAIN_PATH, index=False)
    test_out.to_csv(TEST_PATH, index=False)
    with open(ARTIFACT_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"Rare locations grouped: {artifacts['rare_locations']}")
    print(f"Split datetime: {artifacts['split_datetime']}")
    print(f"Train shape: {train_out.shape}, Test shape: {test_out.shape}")
    print(f"n features: {len(artifacts['feature_columns'])}")
    print(f"Saved train -> {TRAIN_PATH}, test -> {TEST_PATH}")


if __name__ == "__main__":
    main()
