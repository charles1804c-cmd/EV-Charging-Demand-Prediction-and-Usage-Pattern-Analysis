"""Phase 1 — Data Understanding & Cleaning.

Loads data/raw/ChargingRecords.csv, profiles every column,
handles the known quality issues (13 exact duplicates, 17 negative
Duration rows, skewed Demand, imbalanced categoricals, missing check),
converts date/time columns to proper pandas datetime objects,
and saves the cleaned frame to data/processed/charging_cleaned.csv.
"""

from pathlib import Path
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "ChargingRecords.csv"
PROCESSED_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "charging_cleaned.csv"
)

DATETIME_COLS = ["StartDay", "EndDay"]
TIME_COLS = ["StartTime", "EndTime"]
DATETIME_FULL_COLS = ["StartDatetime", "EndDatetime"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw charging records CSV."""
    df = pd.read_csv(path)
    return df


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a profiling summary: dtypes, ranges, uniques, missing."""
    profile = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "missing_per_col": df.isna().sum().to_dict(),
        "exact_duplicates": int(df.duplicated().sum()),
        "negative_duration": int((df["Duration"] < 0).sum()),
        "demand_describe": df["Demand"].describe().to_dict(),
        "location_counts": df["Location"].value_counts().to_dict(),
        "chargertype_counts": df["ChargerType"].value_counts().to_dict(),
        "company_counts": df["ChargerCompany"].value_counts().to_dict(),
    }
    return profile


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the frame and return (cleaned_df, cleaning_report)."""
    report = {}
    report["rows_raw"] = len(df)

    # 1. Missing values — re-verify in code (brief says assume nothing).
    report["missing_before"] = int(df.isna().sum().sum())
    report["missing_per_col"] = df.isna().sum().to_dict()

    # 2. Exact duplicates — drop (confirmed 13 rows).
    n_dupes = int(df.duplicated().sum())
    report["exact_duplicates_found"] = n_dupes
    df = df.drop_duplicates().reset_index(drop=True)
    report["rows_after_dedup"] = len(df)

    # 3. Convert date/time columns to proper datetime objects.
    # StartDay/EndDay -> datetime (date), StartDatetime/EndDatetime -> datetime.
    for col in DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in DATETIME_FULL_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # StartTime/EndTime are times-of-day; keep as timedelta-safe strings
    # but also store parsed time objects for completeness.
    for col in TIME_COLS:
        df[col] = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce").dt.time

    n_bad_parse = int(df[DATETIME_COLS + DATETIME_FULL_COLS].isna().sum().sum())
    report["datetime_parse_failures"] = n_bad_parse

    # 4. Negative Duration — data entry error (EndDatetime < StartDatetime).
    neg_mask = df["Duration"] < 0
    report["negative_duration_found"] = int(neg_mask.sum())
    neg_examples = df.loc[neg_mask, ["StartDatetime", "EndDatetime", "Duration"]].head()
    report["negative_examples"] = neg_examples.to_dict(orient="records")
    # Decision: DROP these rows (cannot reliably correct without ground truth;
    # recomputing from datetimes would still reflect the erroneous datetimes).
    # Flagged explicitly per brief — never silently ignored.
    df = df.loc[~neg_mask].reset_index(drop=True)
    report["rows_after_neg_drop"] = len(df)

    # 5. Enforce dtypes.
    df["UserID"] = df["UserID"].astype(int)
    df["ChargerID"] = df["ChargerID"].astype(int)
    df["ChargerCompany"] = df["ChargerCompany"].astype(int)
    df["ChargerType"] = df["ChargerType"].astype(int)
    df["Duration"] = df["Duration"].astype(int)
    df["Demand"] = df["Demand"].astype(float)

    # 6. Demand skew note (right-skewed; log-transform is a Phase 4 experiment).
    report["demand_mean"] = float(df["Demand"].mean())
    report["demand_median"] = float(df["Demand"].median())
    report["demand_max"] = float(df["Demand"].max())
    report["demand_skew"] = float(df["Demand"].skew())

    # 7. Imbalanced categoricals note.
    report["location_counts_clean"] = df["Location"].value_counts().to_dict()
    report["bus_garage_rows"] = int((df["Location"] == "bus garage").sum())

    report["rows_final"] = len(df)
    return df, report


def main() -> None:
    """CLI entry point: load, profile, clean, save."""
    df = load_raw()
    print("=== RAW PROFILE ===")
    prof = profile_dataframe(df)
    for key, val in prof.items():
        print(f"{key}: {val}")

    cleaned, report = clean_dataframe(df)
    print("\n=== CLEANING REPORT ===")
    for key, val in report.items():
        print(f"{key}: {val}")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSaved cleaned data to {PROCESSED_PATH} ({cleaned.shape})")


if __name__ == "__main__":
    main()
