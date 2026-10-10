"""Phase 6 — Time series: future charging demand forecasting.

Aggregates Demand by hour and by day, tests stationarity (ADF),
forecasts with Moving Average and Exponential Smoothing (Holt-Winters),
evaluates MAE/RMSE on the most recent held-out window, and flags
rising/falling demand periods.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parents[1]
CLEANED_PATH = BASE / "data" / "processed" / "charging_cleaned.csv"
FIG_DIR = BASE / "reports" / "figures"
REPORT_PATH = BASE / "reports" / "time_series_results.csv"
FORECAST_PATH = BASE / "data" / "processed" / "demand_daily_forecast.csv"


def load_series():
    """Return hourly and daily total-demand series."""
    df = pd.read_csv(CLEANED_PATH, parse_dates=["StartDatetime"])
    df = df.sort_values("StartDatetime")
    hourly = df.set_index("StartDatetime").resample("h")["Demand"].sum().fillna(0)
    daily = df.set_index("StartDatetime").resample("D")["Demand"].sum()
    return hourly, daily


def adf_report(series: pd.Series, name: str) -> dict:
    """Run ADF test and return plain-language stationarity verdict."""
    stat, pval, *_ = adfuller(series.dropna())[:2]
    stationary = bool(pval < 0.05)
    verdict = (
        "stationary (no unit root; mean/variance roughly stable)"
        if stationary else "non-stationary (trend/seasonality present)"
    )
    print(f"ADF {name}: stat={stat:.3f}, p={pval:.4g} -> {verdict}")
    return {"series": name, "adf_stat": float(stat), "p_value": float(pval),
            "stationary": stationary, "verdict": verdict}


def moving_average_forecast(train: pd.Series, steps: int, window: int = 7) -> pd.Series:
    """Naive moving-average: repeat the last `window` mean for all steps."""
    level = train.iloc[-window:].mean()
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return pd.Series([level] * steps, index=idx)


def exp_smoothing_forecast(train: pd.Series, steps: int) -> pd.Series:
    """Holt-Winters additive trend+weekly seasonality forecast."""
    fit = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=7
    ).fit(optimized=True)
    return fit.forecast(steps)


def main():
    """Aggregate, test stationarity, forecast, evaluate, plot."""
    hourly, daily = load_series()
    print(f"Hourly points: {len(hourly)}, Daily points: {len(daily)}")
    print(f"Daily range: {daily.index.min()} -> {daily.index.max()}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Trend/seasonality/noise decomposition view (weekly rolling).
    plt.figure(figsize=(10, 3))
    daily.plot(label="daily total")
    daily.rolling(7).mean().plot(label="7-day MA (trend)")
    plt.legend()
    plt.title("Daily charging demand: trend vs noise")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ts_daily_trend.png")
    plt.close()
    plt.figure(figsize=(10, 3))
    hourly.groupby(hourly.index.hour).mean().plot(kind="bar")
    plt.title("Mean hourly demand by hour of day (daily seasonality)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ts_hourly_seasonality.png")
    plt.close()

    adf_daily = adf_report(daily, "daily")
    adf_hourly = adf_report(hourly, "hourly")

    # Hold out the most recent 30 days for evaluation.
    HOLDOUT = 30
    train, test = daily.iloc[:-HOLDOUT], daily.iloc[-HOLDOUT:]
    ma_fc = moving_average_forecast(train, HOLDOUT, window=7)
    es_fc = exp_smoothing_forecast(train, HOLDOUT)

    def scores(fc):
        return (float(mean_absolute_error(test, fc)),
                float(np.sqrt(mean_squared_error(test, fc))))

    ma_mae, ma_rmse = scores(ma_fc)
    es_mae, es_rmse = scores(es_fc)
    print(f"MA  (7-day) -> MAE={ma_mae:.2f} RMSE={ma_rmse:.2f}")
    print(f"ExpSmooth   -> MAE={es_mae:.2f} RMSE={es_rmse:.2f}")

    plt.figure(figsize=(10, 3))
    train.iloc[-90:].plot(label="train (last 90d)")
    test.plot(label="actual (held-out 30d)")
    ma_fc.plot(label="moving-average forecast")
    es_fc.plot(label="exp-smoothing forecast")
    plt.legend()
    plt.title("30-day demand forecast vs actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ts_forecast.png")
    plt.close()

    # Rising/falling demand: compare first vs second half of held-out actuals.
    first, second = test.iloc[:15].mean(), test.iloc[15:].mean()
    direction = "rising" if second > first else "falling"
    print(f"Held-out demand {direction}: first-half mean={first:.1f}, "
          f"second-half mean={second:.1f}")
    # Peak/valley days in full daily series.
    print(f"Peak demand day: {daily.idxmax()} ({daily.max():.1f})")
    print(f"Lowest demand day: {daily.idxmin()} ({daily.min():.1f})")

    pd.DataFrame([
        {"method": "moving_average_7d", "MAE": ma_mae, "RMSE": ma_rmse},
        {"method": "exponential_smoothing", "MAE": es_mae, "RMSE": es_rmse},
    ]).to_csv(REPORT_PATH, index=False)
    pd.DataFrame({"date": es_fc.index, "forecast_demand": es_fc.values}).to_csv(
        FORECAST_PATH, index=False
    )
    print(f"Saved {REPORT_PATH} and {FORECAST_PATH}")


if __name__ == "__main__":
    main()
