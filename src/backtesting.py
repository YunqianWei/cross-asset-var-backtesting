"""
VaR backtesting utilities for the cross-asset VaR backtesting project.

This module evaluates VaR forecasts using:
1. Violation rates
2. Kupiec unconditional coverage test
3. Average and maximum exceedance size
4. Full-sample and high-volatility-regime performance

The module is designed to work with any VaR forecast file that contains:
date, asset, market_group, model, log_return, var_95, var_99,
asset_high_vol_regime.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2


def safe_log_likelihood_binomial(
    x: int,
    n: int,
    p: float,
) -> float:
    """
    Compute binomial log-likelihood safely.

    Parameters
    ----------
    x:
        Number of VaR violations.
    n:
        Number of observations.
    p:
        Violation probability.

    Returns
    -------
    float
        Binomial log-likelihood.
    """

    if n <= 0:
        return np.nan

    if p <= 0.0:
        return 0.0 if x == 0 else -np.inf

    if p >= 1.0:
        return 0.0 if x == n else -np.inf

    return x * np.log(p) + (n - x) * np.log(1.0 - p)


def kupiec_test(
    violations: pd.Series,
    alpha: float,
) -> Dict[str, float]:
    """
    Kupiec unconditional coverage test for VaR violations.

    Null hypothesis:
        The observed violation probability equals alpha.

    Parameters
    ----------
    violations:
        Series of 0/1 VaR violation indicators.
    alpha:
        Expected violation probability. For 95% VaR, alpha=0.05.
        For 99% VaR, alpha=0.01.

    Returns
    -------
    dict
        Test statistics and p-value.
    """

    clean = violations.dropna().astype(int)

    n = int(clean.shape[0])
    x = int(clean.sum())

    if n == 0:
        return {
            "n_obs": 0,
            "exceptions": 0,
            "expected_exceptions": np.nan,
            "violation_rate": np.nan,
            "expected_violation_rate": alpha,
            "kupiec_lr_uc": np.nan,
            "kupiec_p_value": np.nan,
            "kupiec_reject_5pct": np.nan,
        }

    p_hat = x / n

    log_likelihood_null = safe_log_likelihood_binomial(x=x, n=n, p=alpha)
    log_likelihood_alt = safe_log_likelihood_binomial(x=x, n=n, p=p_hat)

    lr_uc = -2.0 * (log_likelihood_null - log_likelihood_alt)

    if np.isfinite(lr_uc):
        p_value = 1.0 - chi2.cdf(lr_uc, df=1)
    else:
        p_value = np.nan

    return {
        "n_obs": n,
        "exceptions": x,
        "expected_exceptions": n * alpha,
        "violation_rate": p_hat,
        "expected_violation_rate": alpha,
        "kupiec_lr_uc": lr_uc,
        "kupiec_p_value": p_value,
        "kupiec_reject_5pct": bool(p_value < 0.05) if np.isfinite(p_value) else np.nan,
    }


def apply_common_evaluation_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only dates where all models have VaR forecasts for the same asset.

    This makes model comparison fair by ensuring that different models are
    evaluated over the same dates within each asset.

    Parameters
    ----------
    df:
        Long-format VaR forecast dataset.

    Returns
    -------
    pd.DataFrame
        Filtered dataset using common evaluation windows.
    """

    filtered_frames = []

    for asset, group in df.groupby("asset", sort=False):
        group = group.copy()
        models = sorted(group["model"].unique())
        n_models = len(models)

        model_count_by_date = (
            group.groupby("date")["model"]
            .nunique()
            .rename("model_count")
            .reset_index()
        )

        common_dates = model_count_by_date.loc[
            model_count_by_date["model_count"] == n_models,
            "date",
        ]

        filtered = group[group["date"].isin(common_dates)].copy()
        filtered_frames.append(filtered)

    result = pd.concat(filtered_frames, ignore_index=True)
    result = result.sort_values(["asset", "date", "model"]).reset_index(drop=True)

    return result


def summarise_var_backtest(
    df: pd.DataFrame,
    group_cols: List[str],
    sample_label: str,
    aggregation_level: str,
) -> pd.DataFrame:
    """
    Summarise VaR backtesting results for 95% and 99% VaR.

    Parameters
    ----------
    df:
        VaR forecast dataset.
    group_cols:
        Columns used for grouping.
    sample_label:
        Label describing the sample, e.g. full_sample, normal_vol, high_vol.
    aggregation_level:
        Label describing aggregation level, e.g. asset_model, group_model.

    Returns
    -------
    pd.DataFrame
        Backtesting summary.
    """

    result_rows = []

    confidence_specs = [
        {
            "confidence_level": "95%",
            "alpha": 0.05,
            "var_col": "var_95",
            "violation_col": "violation_95",
        },
        {
            "confidence_level": "99%",
            "alpha": 0.01,
            "var_col": "var_99",
            "violation_col": "violation_99",
        },
    ]

    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(group_cols, keys))

        for spec in confidence_specs:
            alpha = spec["alpha"]
            var_col = spec["var_col"]
            violation_col = spec["violation_col"]

            test_result = kupiec_test(
                violations=group[violation_col],
                alpha=alpha,
            )

            violations = group[group[violation_col] == 1].copy()

            if len(violations) > 0:
                exceedance = violations[var_col] - violations["log_return"]
                avg_exceedance_conditional = exceedance.mean()
                max_exceedance = exceedance.max()
            else:
                avg_exceedance_conditional = 0.0
                max_exceedance = 0.0

            row = {
                "aggregation_level": aggregation_level,
                "sample": sample_label,
                "confidence_level": spec["confidence_level"],
                **key_dict,
                **test_result,
                "mean_var": group[var_col].mean(),
                "mean_return": group["log_return"].mean(),
                "avg_exceedance_conditional": avg_exceedance_conditional,
                "max_exceedance": max_exceedance,
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
            }

            result_rows.append(row)

    return pd.DataFrame(result_rows)


def build_backtest_results(
    input_path: str = "data/processed/var_forecasts_baseline.csv",
    output_path: str = "data/processed/backtest_results_baseline.csv",
    use_common_window: bool = True,
) -> pd.DataFrame:
    """
    Build VaR backtesting result tables.

    Parameters
    ----------
    input_path:
        Path to VaR forecast dataset.
    output_path:
        Path to save backtesting results.
    use_common_window:
        Whether to restrict each asset to dates where all models have forecasts.

    Returns
    -------
    pd.DataFrame
        Backtesting results.
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
        "model",
        "log_return",
        "asset_high_vol_regime",
        "var_95",
        "var_99",
        "violation_95",
        "violation_99",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values(["asset", "date", "model"]).reset_index(drop=True)

    original_rows = len(df)

    if use_common_window:
        df = apply_common_evaluation_window(df)

    print(f"Original rows: {original_rows:,}")
    print(f"Rows after common-window filter: {len(df):,}")
    print(f"Assets: {df['asset'].nunique()}")
    print(f"Models: {df['model'].nunique()}")

    result_tables = []

    # 1. Asset-level full-sample results.
    result_tables.append(
        summarise_var_backtest(
            df=df,
            group_cols=["market_group", "asset", "model"],
            sample_label="full_sample",
            aggregation_level="asset_model",
        )
    )

    # 2. Market-group-level full-sample results.
    result_tables.append(
        summarise_var_backtest(
            df=df,
            group_cols=["market_group", "model"],
            sample_label="full_sample",
            aggregation_level="market_group_model",
        )
    )

    # 3. Overall model-level full-sample results.
    result_tables.append(
        summarise_var_backtest(
            df=df,
            group_cols=["model"],
            sample_label="full_sample",
            aggregation_level="overall_model",
        )
    )

    # 4. Asset-level regime results.
    normal_df = df[df["asset_high_vol_regime"] == 0].copy()
    high_vol_df = df[df["asset_high_vol_regime"] == 1].copy()

    result_tables.append(
        summarise_var_backtest(
            df=normal_df,
            group_cols=["market_group", "asset", "model"],
            sample_label="normal_vol",
            aggregation_level="asset_model",
        )
    )

    result_tables.append(
        summarise_var_backtest(
            df=high_vol_df,
            group_cols=["market_group", "asset", "model"],
            sample_label="high_vol",
            aggregation_level="asset_model",
        )
    )

    # 5. Market-group-level regime results.
    result_tables.append(
        summarise_var_backtest(
            df=normal_df,
            group_cols=["market_group", "model"],
            sample_label="normal_vol",
            aggregation_level="market_group_model",
        )
    )

    result_tables.append(
        summarise_var_backtest(
            df=high_vol_df,
            group_cols=["market_group", "model"],
            sample_label="high_vol",
            aggregation_level="market_group_model",
        )
    )

    results = pd.concat(result_tables, ignore_index=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print(f"\nSaved backtest results to: {output_file}")
    print(f"Rows: {len(results):,}")

    print("\nFull-sample market-group summary:")
    display_cols = [
        "sample",
        "aggregation_level",
        "market_group",
        "model",
        "confidence_level",
        "n_obs",
        "exceptions",
        "expected_exceptions",
        "violation_rate",
        "expected_violation_rate",
        "kupiec_p_value",
        "kupiec_reject_5pct",
    ]

    group_summary = results[
        (results["sample"] == "full_sample")
        & (results["aggregation_level"] == "market_group_model")
    ][display_cols]

    print(group_summary.to_string(index=False))

    print("\nHigh-volatility market-group summary:")
    high_vol_summary = results[
        (results["sample"] == "high_vol")
        & (results["aggregation_level"] == "market_group_model")
    ][display_cols]

    print(high_vol_summary.to_string(index=False))

    return results


if __name__ == "__main__":
    build_backtest_results()