"""
Data Pipeline — Load, clean, and engineer features for time series forecasting.
"""

import pandas as pd
import numpy as np
import holidays
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "Forecasting Case- Study (1).xlsx")


def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """Load the Excel file and return a clean DataFrame."""
    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "State", "Total"])
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    df = df.dropna(subset=["Total"])
    # Aggregate by State + Date (sum across categories if multiple)
    df = df.groupby(["State", "Date"], as_index=False)["Total"].sum()
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each state, create a complete weekly date range and fill gaps
    using linear interpolation.
    """
    all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="W-SAT")
    states = df["State"].unique()
    records = []
    for state in states:
        state_df = df[df["State"] == state].set_index("Date")["Total"]
        state_df = state_df.reindex(all_dates)
        state_df = state_df.interpolate(method="linear").bfill().ffill()
        tmp = pd.DataFrame({"Date": all_dates, "Total": state_df.values, "State": state})
        records.append(tmp)
    full_df = pd.concat(records, ignore_index=True)
    return full_df.sort_values(["State", "Date"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features."""
    df = df.copy()
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year
    df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun

    # US Federal Holidays flag
    us_holidays = holidays.US(years=range(2018, 2025))
    df["is_holiday"] = df["Date"].dt.date.apply(lambda d: int(d in us_holidays))
    # Near-holiday flag (within 1 week of a US holiday)
    holiday_dates = pd.to_datetime(list(us_holidays.keys()))
    df["near_holiday"] = df["Date"].apply(
        lambda d: int(any(abs((d - h).days) <= 7 for h in holiday_dates))
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features per state."""
    df = df.copy()
    df = df.sort_values(["State", "Date"])
    for state_grp, grp in df.groupby("State"):
        idx = grp.index
        # Lag features
        df.loc[idx, "lag_1"] = grp["Total"].shift(1)
        df.loc[idx, "lag_7"] = grp["Total"].shift(7)
        df.loc[idx, "lag_30"] = grp["Total"].shift(30)
        # Rolling mean and std (4-week and 12-week)
        df.loc[idx, "rolling_mean_4"] = grp["Total"].shift(1).rolling(4, min_periods=1).mean()
        df.loc[idx, "rolling_std_4"] = grp["Total"].shift(1).rolling(4, min_periods=1).std().fillna(0)
        df.loc[idx, "rolling_mean_12"] = grp["Total"].shift(1).rolling(12, min_periods=1).mean()
        df.loc[idx, "rolling_std_12"] = grp["Total"].shift(1).rolling(12, min_periods=1).std().fillna(0)
        # Trend: difference from previous period
        df.loc[idx, "trend_1"] = grp["Total"].diff(1)
        df.loc[idx, "trend_4"] = grp["Total"].diff(4)
    return df


def build_features(filepath: str = DATA_FILE) -> pd.DataFrame:
    """
    Full pipeline: load → fill missing → calendar → lag features.
    Returns a DataFrame ready for model training.
    """
    df = load_data(filepath)
    df = fill_missing_weeks(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def train_val_split(state_df: pd.DataFrame, val_weeks: int = 16):
    """
    Time-series aware split — last `val_weeks` rows as validation.
    NO data leakage.
    """
    state_df = state_df.sort_values("Date")
    train = state_df.iloc[:-val_weeks]
    val = state_df.iloc[-val_weeks:]
    return train, val


def get_state_series(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Extract a single state's time series, sorted by date."""
    return df[df["State"] == state].sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    print("Running data pipeline...")
    df = build_features()
    print(f"Feature dataframe shape: {df.shape}")
    print(df.head())
    print("\nFeature columns:", df.columns.tolist())
    print("\nNull counts:\n", df.isnull().sum())
    df.to_csv("data/features.csv", index=False)
    print("Saved to data/features.csv")
