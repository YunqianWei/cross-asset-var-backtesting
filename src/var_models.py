"""
Baseline VaR models for the cross-asset VaR backtesting project.

This module implements:
1. Historical Simulation VaR
2. EWMA Normal VaR

The forecasts are one-day-ahead VaR estimates and are shifted to avoid
look-ahead bias.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(
    returns: pd.Series,
    window: int = 250,
    alpha: float = 0.05,
) -> pd.Series:
    """
    Compute one-day-ahead Historical Simulation VaR.

    The rolling quantile is shifted by one day so that VaR at date t only uses
    returns available up to date t-1.

    Parameters
    ----------
    returns:
        Return series.
    window:
        Rolling window length.
    alpha:
        Tail probability. For 95% VaR use alpha=0.05; for 99% VaR use alpha=0.01.

    Returns
    -------
    pd.Series
        One-day-ahead VaR threshold.
    """
    return returns.rolling(window=window).quantile(alpha).shift(1)


def ewma_volatility(
    returns: pd.Series,
    lambda_: float = 0.94,
) -> pd.Series:
    """
    Compute one-day-ahead EWMA volatility forecast.

    The variance estimate is based on exponentially weighted squared returns.
    It is shifted by one day to avoid look-ahead bias.

    Parameters
    ----------
    returns:
        Return series.
    lambda_:
        EWMA decay factor. 0.94 is a common daily-data choice.

    Returns
    -------
    pd.Series
        One-day-ahead EWMA volatility forecast.
    """
    squared_returns = returns.pow(2)
    ewma_variance = squared_returns.ewm(alpha=1.0 - lambda_, adjust=False).mean()
    ewma_variance_forecast = ewma_variance.shift(1)
    return np.sqrt(ewma_variance_forecast)


def ewma_normal_var(
    returns: pd.Series,
    lambda_: float = 0.94,
    alpha: float = 0.05,
) -> pd.Series:
    """
    Compute one-day-ahead EWMA Normal VaR.

    A zero conditional mean is used, which is standard for short-horizon daily
    VaR applications.

    Parameters
    ----------
    returns:
        Return series.
    lambda_:
        EWMA decay factor.
    alpha:
        Tail probability.

    Returns
    -------
    pd.Series
        One-day-ahead VaR threshold.
    """
    sigma = ewma_volatility(returns, lambda_=lambda_)
    z_alpha = norm.ppf(alpha)
    return z_alpha * sigma


def build_baseline_var_forecasts(
    input_path: str = "data/processed/regime_features.csv",
    output_path: str = "data/processed/var_forecasts_baseline.csv",
    historical_window: int = 250,
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """
    Build baseline VaR forecasts for all assets.

    Parameters
    ----------
    input_path:
        Path to regime-enhanced feature dataset.
    output_path:
        Path to save baseline VaR forecasts.
    historical_window:
        Rolling window used for Historical Simulation VaR.
    ewma_lambda:
        EWMA decay parameter.

    Returns
    -------
    pd.DataFrame
        Long-format VaR forecast dataset.
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
        "market_group",
        "log_return",
        "asset_high_vol_regime",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    forecast_frames = []

    for asset, group in df.groupby("asset", sort=False):
        group = group.copy()
        group = group.sort_values("date").reset_index(drop=True)

        r = group["log_return"].astype(float)

        # Historical Simulation VaR
        hist = group[
            [
                "date",
                "asset",
                "asset_name",
                "market_group",
                "log_return",
                "asset_high_vol_regime",
            ]
        ].copy()

        hist["model"] = f"historical_{historical_window}d"
        hist["var_95"] = historical_var(r, window=historical_window, alpha=0.05)
        hist["var_99"] = historical_var(r, window=historical_window, alpha=0.01)

        forecast_frames.append(hist)

        # EWMA Normal VaR
        ewma = group[
            [
                "date",
                "asset",
                "asset_name",
                "market_group",
                "log_return",
                "asset_high_vol_regime",
            ]
        ].copy()

        ewma["model"] = f"ewma_{ewma_lambda:.2f}"
        ewma["var_95"] = ewma_normal_var(r, lambda_=ewma_lambda, alpha=0.05)
        ewma["var_99"] = ewma_normal_var(r, lambda_=ewma_lambda, alpha=0.01)

        forecast_frames.append(ewma)

    forecasts = pd.concat(forecast_frames, ignore_index=True)

    # Remove rows where VaR cannot be computed because of insufficient history.
    forecasts = forecasts.dropna(subset=["var_95", "var_99"]).reset_index(drop=True)

    # Violation indicators. A violation occurs when realised return is below VaR.
    forecasts["violation_95"] = (forecasts["log_return"] < forecasts["var_95"]).astype(int)
    forecasts["violation_99"] = (forecasts["log_return"] < forecasts["var_99"]).astype(int)

    # Exceedance size: how far below the VaR threshold the realised return was.
    forecasts["exceedance_95"] = np.where(
        forecasts["violation_95"] == 1,
        forecasts["var_95"] - forecasts["log_return"],
        0.0,
    )

    forecasts["exceedance_99"] = np.where(
        forecasts["violation_99"] == 1,
        forecasts["var_99"] - forecasts["log_return"],
        0.0,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_file, index=False)

    print(f"Saved baseline VaR forecasts to: {output_file}")
    print(f"Rows: {len(forecasts):,}")
    print(f"Assets: {forecasts['asset'].nunique()}")
    print(f"Models: {forecasts['model'].nunique()}")

    summary = (
        forecasts.groupby(["market_group", "asset", "model"])
        .agg(
            observations=("date", "count"),
            violation_rate_95=("violation_95", "mean"),
            violation_rate_99=("violation_99", "mean"),
            avg_exceedance_95=("exceedance_95", "mean"),
            avg_exceedance_99=("exceedance_99", "mean"),
            mean_var_95=("var_95", "mean"),
            mean_var_99=("var_99", "mean"),
        )
        .reset_index()
    )

    print("\nBaseline VaR summary:")
    print(summary.to_string(index=False))

    return forecasts


if __name__ == "__main__":
    build_baseline_var_forecasts()