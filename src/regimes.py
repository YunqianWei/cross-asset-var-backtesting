"""
Regime classification utilities for the cross-asset VaR backtesting project.

This module defines high-volatility regimes using asset-specific realised
volatility percentiles and assigns each asset to a market group.
"""

from pathlib import Path

import pandas as pd


US_ETFS = {"SPY", "QQQ", "IWM", "TLT", "GLD"}
EUROPE_INDICES = {"^FTSE", "^GDAXI", "^FCHI", "^STOXX50E"}
CRYPTO_ASSETS = {"BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD"}


def assign_market_group(asset: str) -> str:
    """
    Assign an asset to a broad market group.
    """
    if asset in US_ETFS:
        return "US_ETF"
    if asset in EUROPE_INDICES:
        return "European_Index"
    if asset in CRYPTO_ASSETS:
        return "Crypto"
    return "Other"


def build_regime_features(
    input_path: str = "data/processed/features.csv",
    output_path: str = "data/processed/regime_features.csv",
    vol_column: str = "realised_vol_20d",
    percentile: float = 0.80,
) -> pd.DataFrame:
    """
    Add market group labels and asset-specific high-volatility regime labels.

    Parameters
    ----------
    input_path:
        Path to processed feature dataset.
    output_path:
        Path to save regime-enhanced feature dataset.
    vol_column:
        Volatility column used to define high-volatility regime.
    percentile:
        Percentile threshold for high-volatility classification.

    Returns
    -------
    pd.DataFrame
        Dataset with additional regime columns.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file, parse_dates=["date"])

    if vol_column not in df.columns:
        raise ValueError(f"Column not found: {vol_column}")

    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    df["market_group"] = df["asset"].apply(assign_market_group)

    # Asset-specific volatility threshold.
    thresholds = (
        df.groupby("asset")[vol_column]
        .quantile(percentile)
        .rename("asset_vol_threshold")
        .reset_index()
    )

    df = df.merge(thresholds, on="asset", how="left")

    df["asset_high_vol_regime"] = (
        df[vol_column] >= df["asset_vol_threshold"]
    ).astype(int)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Saved regime features to: {output_file}")
    print(f"Rows: {len(df):,}")
    print(f"Assets: {df['asset'].nunique()}")

    regime_summary = (
        df.groupby(["market_group", "asset"])
        .agg(
            observations=("date", "count"),
            high_vol_days=("asset_high_vol_regime", "sum"),
            high_vol_share=("asset_high_vol_regime", "mean"),
            vol_threshold=("asset_vol_threshold", "first"),
            mean_vol=(vol_column, "mean"),
        )
        .reset_index()
    )

    print("\nRegime summary:")
    print(regime_summary.to_string(index=False))

    return df


if __name__ == "__main__":
    build_regime_features()