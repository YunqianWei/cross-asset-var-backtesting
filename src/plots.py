"""
Plotting utilities for the cross-asset VaR backtesting project.

This script creates a first set of figures for GitHub/report use.

Generated figures:
1. realised_volatility_by_asset.png
2. full_sample_violation_rate_95.png
3. full_sample_violation_rate_99.png
4. high_vol_violation_rate_95.png
5. high_vol_violation_rate_99.png
6. full_sample_kupiec_rejection_rate.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "historical_250d",
    "ewma_0.94",
    "garch_normal",
    "garch_student_t",
    "gb_quantile",
]

MARKET_GROUP_ORDER = [
    "Crypto",
    "US_ETF",
    "European_Index",
]


def ensure_figure_dir(path: str = "figures") -> Path:
    figure_dir = Path(path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def save_bar_plot(
    labels,
    values,
    title: str,
    y_label: str,
    output_path: Path,
    ref_line: float | None = None,
):
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)

    if ref_line is not None:
        plt.axhline(ref_line, linestyle="--", linewidth=1, label=f"Expected = {ref_line:.2f}")
        plt.legend()

    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def build_realised_volatility_plot(
    features_path: str = "data/processed/features.csv",
    figure_dir: str = "figures",
):
    features_file = Path(features_path)
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")

    df = pd.read_csv(features_file, parse_dates=["date"])

    summary = (
        df.groupby("asset")
        .agg(
            annualised_volatility=("log_return", lambda x: x.std() * np.sqrt(252))
        )
        .reset_index()
        .sort_values("annualised_volatility", ascending=False)
    )

    figure_path = ensure_figure_dir(figure_dir) / "realised_volatility_by_asset.png"

    save_bar_plot(
        labels=summary["asset"],
        values=summary["annualised_volatility"],
        title="Annualised Volatility by Asset",
        y_label="Annualised Volatility",
        output_path=figure_path,
    )

    print(f"Saved: {figure_path}")


def prepare_market_group_model_plot_data(
    results: pd.DataFrame,
    sample: str,
    confidence_level: str,
):
    plot_df = results[
        (results["aggregation_level"] == "market_group_model")
        & (results["sample"] == sample)
        & (results["confidence_level"] == confidence_level)
    ].copy()

    plot_df["model_order"] = plot_df["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999
    )
    plot_df["group_order"] = plot_df["market_group"].apply(
        lambda x: MARKET_GROUP_ORDER.index(x) if x in MARKET_GROUP_ORDER else 999
    )

    plot_df = plot_df.sort_values(["group_order", "model_order"]).reset_index(drop=True)
    plot_df["label"] = plot_df["market_group"] + "\n" + plot_df["model"]

    return plot_df


def build_violation_rate_plots(
    backtest_path: str = "data/processed/backtest_results_with_ml.csv",
    figure_dir: str = "figures",
):
    backtest_file = Path(backtest_path)
    if not backtest_file.exists():
        raise FileNotFoundError(f"Backtest results file not found: {backtest_file}")

    df = pd.read_csv(backtest_file)

    figure_dir_path = ensure_figure_dir(figure_dir)

    plot_specs = [
        ("full_sample", "95%", 0.05, "full_sample_violation_rate_95.png"),
        ("full_sample", "99%", 0.01, "full_sample_violation_rate_99.png"),
        ("high_vol", "95%", 0.05, "high_vol_violation_rate_95.png"),
        ("high_vol", "99%", 0.01, "high_vol_violation_rate_99.png"),
    ]

    for sample, confidence_level, ref_line, filename in plot_specs:
        plot_df = prepare_market_group_model_plot_data(
            results=df,
            sample=sample,
            confidence_level=confidence_level,
        )

        figure_path = figure_dir_path / filename

        save_bar_plot(
            labels=plot_df["label"],
            values=plot_df["violation_rate"],
            title=f"Violation Rates ({sample}, VaR {confidence_level})",
            y_label="Violation Rate",
            output_path=figure_path,
            ref_line=ref_line,
        )

        print(f"Saved: {figure_path}")


def build_kupiec_rejection_plot(
    backtest_path: str = "data/processed/backtest_results_with_ml.csv",
    figure_dir: str = "figures",
):
    backtest_file = Path(backtest_path)
    if not backtest_file.exists():
        raise FileNotFoundError(f"Backtest results file not found: {backtest_file}")

    df = pd.read_csv(backtest_file)

    asset_level = df[
        (df["aggregation_level"] == "asset_model")
        & (df["sample"] == "full_sample")
    ].copy()

    summary = (
        asset_level.groupby(["model", "confidence_level"])
        .agg(
            rejection_rate=("kupiec_reject_5pct", "mean")
        )
        .reset_index()
    )

    summary["model_order"] = summary["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999
    )
    summary["confidence_order"] = summary["confidence_level"].apply(
        lambda x: 0 if x == "95%" else 1
    )

    summary = summary.sort_values(["confidence_order", "model_order"]).reset_index(drop=True)
    summary["label"] = summary["confidence_level"] + "\n" + summary["model"]

    figure_path = ensure_figure_dir(figure_dir) / "full_sample_kupiec_rejection_rate.png"

    save_bar_plot(
        labels=summary["label"],
        values=summary["rejection_rate"],
        title="Share of Assets Rejected by Kupiec Test (Full Sample)",
        y_label="Rejection Rate Across Assets",
        output_path=figure_path,
    )

    print(f"Saved: {figure_path}")


def main():
    build_realised_volatility_plot()
    build_violation_rate_plots()
    build_kupiec_rejection_plot()
    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()