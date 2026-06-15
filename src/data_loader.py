"""
Data loading utilities for the cross-asset VaR backtesting project.

This module downloads daily market data from Yahoo Finance using yfinance.
The output is saved locally under data/raw/ and is not intended to be committed
to GitHub.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf


US_ETFS: Dict[str, str] = {
    "SPY": "US Large-Cap Equity ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "US Small-Cap Equity ETF",
    "TLT": "Long-Term US Treasury ETF",
    "GLD": "Gold ETF",
}

EUROPE_INDICES: Dict[str, str] = {
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX",
    "^FCHI": "CAC 40",
    "^STOXX50E": "Euro Stoxx 50",
}

CRYPTO_ASSETS: Dict[str, str] = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "DOGE-USD": "Dogecoin",
    "XRP-USD": "XRP",
}

ALL_ASSETS: Dict[str, str] = {
    **US_ETFS,
    **EUROPE_INDICES,
    **CRYPTO_ASSETS,
}


def download_yfinance_data(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance tickers.
    start:
        Start date in YYYY-MM-DD format.
    end:
        Optional end date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        date, asset, open, high, low, close, adj_close, volume.
    """

    all_frames = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            print(f"Warning: no data returned for {ticker}")
            continue

        # Handle possible multi-index columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }

        data = data.rename(columns=rename_map)

        required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValueError(f"{ticker} is missing columns: {missing_cols}")

        data = data[required_cols]
        data["asset"] = ticker
        data["asset_name"] = ALL_ASSETS.get(ticker, ticker)

        all_frames.append(data)

    if not all_frames:
        raise RuntimeError("No data was downloaded. Check tickers or internet connection.")

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["asset", "date"]).reset_index(drop=True)

    return result


def save_raw_data(
    output_path: str = "data/raw/yfinance_prices.csv",
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Download and save raw market data.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tickers = list(ALL_ASSETS.keys())

    data = download_yfinance_data(
        tickers=tickers,
        start=start,
    )

    data.to_csv(output, index=False)

    print(f"Saved raw data to: {output}")
    print(f"Rows: {len(data):,}")
    print(f"Assets: {data['asset'].nunique()}")

    return data


if __name__ == "__main__":
    save_raw_data()