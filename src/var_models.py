"""
VaR models for the cross-asset VaR backtesting project.

This module implements:
1. Historical Simulation VaR
2. EWMA Normal VaR
3. GARCH(1,1)-Normal VaR
4. GARCH(1,1)-Student-t VaR

All forecasts are one-day-ahead VaR estimates and are constructed to avoid
look-ahead bias.
"""

from pathlib import Path
from typing import Dict, Tuple

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t


def historical_var(
    returns: pd.Series,
    window: int = 250,
    alpha: float = 0.05,
) -> pd.Series:
    """
    Compute one-day-ahead Historical Simulation VaR.

    The rolling quantile is shifted by one day so that VaR at date t only uses
    returns available up to date t-1.
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
    """
    sigma = ewma_volatility(returns, lambda_=lambda_)
    z_alpha = norm.ppf(alpha)
    return z_alpha * sigma


def extract_garch_parameters(result, distribution: str) -> Dict[str, float]:
    """
    Extract GARCH(1,1) parameters from an arch model result.
    """

    params = result.params

    output = {
        "omega": float(params["omega"]),
        "alpha": float(params["alpha[1]"]),
        "beta": float(params["beta[1]"]),
    }

    if distribution == "t":
        output["nu"] = float(params["nu"])

    return output


def fit_garch_model(
    returns_pct: pd.Series,
    distribution: str,
):
    """
    Fit a zero-mean GARCH(1,1) model to percentage returns.

    Parameters
    ----------
    returns_pct:
        Returns multiplied by 100.
    distribution:
        Either "normal" or "t".
    """

    model = arch_model(
        returns_pct,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist=distribution,
        rescale=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp="off", show_warning=False)

    return result


def garch_var_forecast(
    returns: pd.Series,
    distribution: str = "normal",
    min_window: int = 500,
    refit_frequency: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute one-day-ahead GARCH VaR forecasts.

    The GARCH parameters are re-estimated every `refit_frequency` days using
    only information available before the forecast date. Between refits, the
    conditional variance is updated recursively using observed returns.

    Parameters
    ----------
    returns:
        Daily log-return series in decimal form.
    distribution:
        "normal" or "t".
    min_window:
        Minimum number of historical observations required before the first
        GARCH fit.
    refit_frequency:
        Number of days between parameter re-estimations.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        95% and 99% VaR forecast series.
    """

    if distribution not in {"normal", "t"}:
        raise ValueError("distribution must be either 'normal' or 't'")

    r = returns.astype(float).reset_index(drop=True)
    r_pct = r * 100.0

    n = len(r)

    var_95 = pd.Series(np.nan, index=r.index)
    var_99 = pd.Series(np.nan, index=r.index)

    current_params = None
    sigma_prev = None

    for i in range(min_window, n):
        should_refit = (
            current_params is None
            or (i - min_window) % refit_frequency == 0
        )

        if should_refit:
            train = r_pct.iloc[:i].dropna()

            if len(train) < min_window:
                continue

            try:
                result = fit_garch_model(
                    returns_pct=train,
                    distribution=distribution,
                )

                current_params = extract_garch_parameters(
                    result=result,
                    distribution=distribution,
                )

                # Last fitted conditional volatility corresponds to date i-1.
                sigma_prev = float(result.conditional_volatility.iloc[-1])

            except Exception as exc:
                print(
                    f"Warning: GARCH fit failed at index {i} "
                    f"for distribution={distribution}: {exc}"
                )
                continue

        if current_params is None or sigma_prev is None:
            continue

        omega = current_params["omega"]
        alpha = current_params["alpha"]
        beta = current_params["beta"]

        previous_return = float(r_pct.iloc[i - 1])

        sigma2_forecast = omega + alpha * previous_return**2 + beta * sigma_prev**2

        if sigma2_forecast <= 0 or not np.isfinite(sigma2_forecast):
            continue

        sigma_forecast = float(np.sqrt(sigma2_forecast))

        if distribution == "normal":
            q_95 = norm.ppf(0.05)
            q_99 = norm.ppf(0.01)

        else:
            nu = current_params["nu"]

            if nu <= 2:
                continue

            # arch uses a standardised Student-t distribution with unit variance.
            scale = np.sqrt((nu - 2.0) / nu)
            q_95 = t.ppf(0.05, df=nu) * scale
            q_99 = t.ppf(0.01, df=nu) * scale

        # Convert percentage-return VaR back to decimal-return VaR.
        var_95.iloc[i] = q_95 * sigma_forecast / 100.0
        var_99.iloc[i] = q_99 * sigma_forecast / 100.0

        # After observing return i, update sigma_prev for the next day's forecast.
        sigma_prev = sigma_forecast

    return var_95, var_99


def make_forecast_frame(
    group: pd.DataFrame,
    model_name: str,
    var_95: pd.Series,
    var_99: pd.Series,
) -> pd.DataFrame:
    """
    Create a standard long-format forecast frame.
    """

    frame = group[
        [
            "date",
            "asset",
            "asset_name",
            "market_group",
            "log_return",
            "asset_high_vol_regime",
        ]
    ].copy()

    frame["model"] = model_name
    frame["var_95"] = var_95.values
    frame["var_99"] = var_99.values

    return frame


def add_violation_columns(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Add VaR violation and exceedance columns.
    """

    forecasts = forecasts.copy()

    forecasts["violation_95"] = (
        forecasts["log_return"] < forecasts["var_95"]
    ).astype(int)

    forecasts["violation_99"] = (
        forecasts["log_return"] < forecasts["var_99"]
    ).astype(int)

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

    return forecasts


def build_var_forecasts(
    input_path: str = "data/processed/regime_features.csv",
    output_path: str = "data/processed/var_forecasts_all.csv",
    historical_window: int = 250,
    ewma_lambda: float = 0.94,
    garch_min_window: int = 500,
    garch_refit_frequency: int = 20,
    include_garch: bool = True,
) -> pd.DataFrame:
    """
    Build VaR forecasts for all assets and all selected models.
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
        print(f"\nBuilding VaR forecasts for {asset}...")

        group = group.copy()
        group = group.sort_values("date").reset_index(drop=True)

        r = group["log_return"].astype(float)

        # Historical Simulation VaR.
        hist_var_95 = historical_var(
            returns=r,
            window=historical_window,
            alpha=0.05,
        )

        hist_var_99 = historical_var(
            returns=r,
            window=historical_window,
            alpha=0.01,
        )

        forecast_frames.append(
            make_forecast_frame(
                group=group,
                model_name=f"historical_{historical_window}d",
                var_95=hist_var_95,
                var_99=hist_var_99,
            )
        )

        # EWMA Normal VaR.
        ewma_var_95 = ewma_normal_var(
            returns=r,
            lambda_=ewma_lambda,
            alpha=0.05,
        )

        ewma_var_99 = ewma_normal_var(
            returns=r,
            lambda_=ewma_lambda,
            alpha=0.01,
        )

        forecast_frames.append(
            make_forecast_frame(
                group=group,
                model_name=f"ewma_{ewma_lambda:.2f}",
                var_95=ewma_var_95,
                var_99=ewma_var_99,
            )
        )

        if include_garch:
            # GARCH-Normal VaR.
            print(f"  Fitting GARCH-Normal for {asset}...")

            garch_normal_95, garch_normal_99 = garch_var_forecast(
                returns=r,
                distribution="normal",
                min_window=garch_min_window,
                refit_frequency=garch_refit_frequency,
            )

            forecast_frames.append(
                make_forecast_frame(
                    group=group,
                    model_name="garch_normal",
                    var_95=garch_normal_95,
                    var_99=garch_normal_99,
                )
            )

            # GARCH-Student-t VaR.
            print(f"  Fitting GARCH-Student-t for {asset}...")

            garch_t_95, garch_t_99 = garch_var_forecast(
                returns=r,
                distribution="t",
                min_window=garch_min_window,
                refit_frequency=garch_refit_frequency,
            )

            forecast_frames.append(
                make_forecast_frame(
                    group=group,
                    model_name="garch_student_t",
                    var_95=garch_t_95,
                    var_99=garch_t_99,
                )
            )

    forecasts = pd.concat(forecast_frames, ignore_index=True)

    # Remove rows where VaR cannot be computed because of insufficient history.
    forecasts = forecasts.dropna(subset=["var_95", "var_99"]).reset_index(drop=True)

    forecasts = add_violation_columns(forecasts)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_file, index=False)

    print(f"\nSaved VaR forecasts to: {output_file}")
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

    print("\nVaR forecast summary:")
    print(summary.to_string(index=False))

    return forecasts


if __name__ == "__main__":
    build_var_forecasts()