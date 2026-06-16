"""
Feature engineering utilities for the cross-asset VaR backtesting project.

This module reads raw Yahoo Finance data and creates return, volatility,
drawdown, and volume-based features for later volatility forecasting and VaR
backtesting.
"""

from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def compute_max_drawdown(window: pd.Series) -> float:
    """
    Compute maximum drawdown within a rolling price window.

    Parameters
    ----------
    window:
        Rolling window of close prices.

    Returns
    -------
    float
        Maximum drawdown as a negative number.
    """
    running_max = window.cummax()
    drawdowns = window / running_max - 1.0
    return drawdowns.min()


def build_features(
    input_path: str = "data/raw/yfinance_prices.csv",
    output_path: str = "data/processed/features.csv",
) -> pd.DataFrame:
    """
    Build a clean feature dataset from raw Yahoo Finance data.

    Parameters
    ----------
    input_path:
        Path to raw yfinance price data.
    output_path:
        Path to save processed features.

    Returns
    -------
    pd.DataFrame
        Feature dataset.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file, parse_dates=["date"])

    required_cols = [
        "date",
        "asset",
        "asset_name",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    feature_frames = []

    for asset, group in df.groupby("asset", sort=False):
        group = group.copy()
        group = group.sort_values("date").reset_index(drop=True)

        # Use adjusted close for returns where available.
        price = group["adj_close"].astype(float)
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)

        group["log_return"] = np.log(price / price.shift(1))
        group["abs_return"] = group["log_return"].abs()
        group["squared_return"] = group["log_return"] ** 2

        # Rolling cumulative log returns.
        group["rolling_return_5d"] = group["log_return"].rolling(5).sum()
        group["rolling_return_20d"] = group["log_return"].rolling(20).sum()

        # Realised volatility as daily standard deviation of returns.
        group["realised_vol_5d"] = group["log_return"].rolling(5).std()
        group["realised_vol_20d"] = group["log_return"].rolling(20).std()
        group["realised_vol_60d"] = group["log_return"].rolling(60).std()

        # Annualised realised volatility, useful for plots and descriptive tables.
        group["realised_vol_20d_annualised"] = (
            group["realised_vol_20d"] * np.sqrt(TRADING_DAYS)
        )

        # Higher moments of returns.
        group["rolling_skew_20d"] = group["log_return"].rolling(20).skew()
        group["rolling_kurt_20d"] = group["log_return"].rolling(20).kurt()

        # Rolling max drawdown based on close prices.
        group["rolling_max_drawdown_20d"] = (
            close.rolling(20).apply(compute_max_drawdown, raw=False)
        )

        # Volume feature. Some indices may have zero or missing volume from Yahoo.
        group["volume_change_5d"] = np.log(volume.replace(0, np.nan) / volume.replace(0, np.nan).shift(5))

        feature_frames.append(group)

    features = pd.concat(feature_frames, ignore_index=True)

    # Remove rows where return-based features cannot be computed.
    features = features.dropna(subset=["log_return", "realised_vol_20d"]).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file, index=False)

    print(f"Saved features to: {output_file}")
    print(f"Rows: {len(features):,}")
    print(f"Assets: {features['asset'].nunique()}")

    summary = (
        features.groupby("asset")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            observations=("date", "count"),
            mean_daily_return=("log_return", "mean"),
            daily_volatility=("log_return", "std"),
            annualised_volatility=("log_return", lambda x: x.std() * np.sqrt(TRADING_DAYS)),
            worst_daily_return=("log_return", "min"),
        )
        .reset_index()
    )

    print("\nAsset-level summary:")
    print(summary.to_string(index=False))

    return features


if __name__ == "__main__":
    build_features()