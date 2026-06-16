"""
Machine learning VaR models for the cross-asset VaR backtesting project.

This module implements Gradient Boosting Quantile Regression for direct
one-day-ahead VaR forecasting.

The model predicts:
- 5% conditional return quantile as 95% VaR
- 1% conditional return quantile as 99% VaR

Features are shifted by one day to avoid look-ahead bias.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


FEATURE_COLUMNS: List[str] = [
    "lag_1_return",
    "lag_2_return",
    "lag_5_return",
    "rolling_return_5d_lag1",
    "rolling_return_20d_lag1",
    "realised_vol_5d_lag1",
    "realised_vol_20d_lag1",
    "realised_vol_60d_lag1",
    "rolling_skew_20d_lag1",
    "rolling_kurt_20d_lag1",
    "rolling_max_drawdown_20d_lag1",
    "volume_change_5d_lag1",
]


def make_quantile_model(alpha: float) -> GradientBoostingRegressor:
    """
    Create a Gradient Boosting quantile regression model.

    Parameters
    ----------
    alpha:
        Quantile level. Use 0.05 for 95% VaR and 0.01 for 99% VaR.

    Returns
    -------
    GradientBoostingRegressor
        Quantile regression model.
    """

    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=120,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=20,
        random_state=42,
    )


def prepare_asset_supervised_data(group: pd.DataFrame) -> pd.DataFrame:
    """
    Construct supervised learning data for one asset.

    The target is today's log return.
    Features are shifted so that predictions for date t only use information
    available up to date t-1.
    """

    group = group.copy()
    group = group.sort_values("date").reset_index(drop=True)

    df = group[
        [
            "date",
            "asset",
            "asset_name",
            "market_group",
            "log_return",
            "asset_high_vol_regime",
        ]
    ].copy()

    # Target: realised return on date t.
    df["target_return"] = group["log_return"].astype(float)

    # Direct lagged returns.
    df["lag_1_return"] = group["log_return"].shift(1)
    df["lag_2_return"] = group["log_return"].shift(2)
    df["lag_5_return"] = group["log_return"].shift(5)

    # Shift engineered features by one day to avoid look-ahead bias.
    shifted_feature_map = {
        "rolling_return_5d": "rolling_return_5d_lag1",
        "rolling_return_20d": "rolling_return_20d_lag1",
        "realised_vol_5d": "realised_vol_5d_lag1",
        "realised_vol_20d": "realised_vol_20d_lag1",
        "realised_vol_60d": "realised_vol_60d_lag1",
        "rolling_skew_20d": "rolling_skew_20d_lag1",
        "rolling_kurt_20d": "rolling_kurt_20d_lag1",
        "rolling_max_drawdown_20d": "rolling_max_drawdown_20d_lag1",
        "volume_change_5d": "volume_change_5d_lag1",
    }

    for source_col, target_col in shifted_feature_map.items():
        if source_col in group.columns:
            df[target_col] = group[source_col].shift(1)
        else:
            df[target_col] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def fit_predict_quantile_var_for_asset(
    asset_df: pd.DataFrame,
    min_train_size: int = 500,
    refit_frequency: int = 40,
) -> pd.DataFrame:
    """
    Fit rolling Gradient Boosting quantile models for one asset.

    Parameters
    ----------
    asset_df:
        Supervised asset-level dataframe.
    min_train_size:
        Minimum number of observations before first prediction.
    refit_frequency:
        Number of days between model refits.

    Returns
    -------
    pd.DataFrame
        Forecast dataframe with var_95 and var_99.
    """

    asset_df = asset_df.copy().reset_index(drop=True)

    n = len(asset_df)

    var_95 = pd.Series(np.nan, index=asset_df.index)
    var_99 = pd.Series(np.nan, index=asset_df.index)

    model_05 = None
    model_01 = None
    feature_medians = None

    for i in range(min_train_size, n):
        should_refit = (
            model_05 is None
            or model_01 is None
            or (i - min_train_size) % refit_frequency == 0
        )

        if should_refit:
            train = asset_df.iloc[:i].copy()
            train = train.dropna(subset=["target_return"])

            if len(train) < min_train_size:
                continue

            X_train = train[FEATURE_COLUMNS].copy()
            y_train = train["target_return"].copy()

            # Median imputation using training data only.
            feature_medians = X_train.median(numeric_only=True).fillna(0.0)
            X_train = X_train.fillna(feature_medians)

            model_05 = make_quantile_model(alpha=0.05)
            model_01 = make_quantile_model(alpha=0.01)

            model_05.fit(X_train, y_train)
            model_01.fit(X_train, y_train)

        if model_05 is None or model_01 is None or feature_medians is None:
            continue

        X_test = asset_df.loc[[i], FEATURE_COLUMNS].copy()
        X_test = X_test.fillna(feature_medians)

        pred_05 = float(model_05.predict(X_test)[0])
        pred_01 = float(model_01.predict(X_test)[0])

        # Enforce quantile ordering: 1% VaR should not be above 5% VaR.
        var_95.iloc[i] = pred_05
        var_99.iloc[i] = min(pred_01, pred_05)

    forecast = asset_df[
        [
            "date",
            "asset",
            "asset_name",
            "market_group",
            "log_return",
            "asset_high_vol_regime",
        ]
    ].copy()

    forecast["model"] = "gb_quantile"
    forecast["var_95"] = var_95
    forecast["var_99"] = var_99

    forecast = forecast.dropna(subset=["var_95", "var_99"]).reset_index(drop=True)

    forecast["violation_95"] = (
        forecast["log_return"] < forecast["var_95"]
    ).astype(int)

    forecast["violation_99"] = (
        forecast["log_return"] < forecast["var_99"]
    ).astype(int)

    forecast["exceedance_95"] = np.where(
        forecast["violation_95"] == 1,
        forecast["var_95"] - forecast["log_return"],
        0.0,
    )

    forecast["exceedance_99"] = np.where(
        forecast["violation_99"] == 1,
        forecast["var_99"] - forecast["log_return"],
        0.0,
    )

    return forecast


def build_ml_var_forecasts(
    input_path: str = "data/processed/regime_features.csv",
    statistical_var_path: str = "data/processed/var_forecasts_all.csv",
    ml_output_path: str = "data/processed/var_forecasts_ml.csv",
    combined_output_path: str = "data/processed/var_forecasts_with_ml.csv",
    min_train_size: int = 500,
    refit_frequency: int = 40,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build ML VaR forecasts and append them to existing statistical VaR forecasts.

    Parameters
    ----------
    input_path:
        Path to regime-enhanced feature dataset.
    statistical_var_path:
        Existing VaR forecast file containing historical, EWMA and GARCH models.
    ml_output_path:
        Path to save ML-only VaR forecasts.
    combined_output_path:
        Path to save statistical + ML VaR forecasts.
    min_train_size:
        Minimum training size for ML models.
    refit_frequency:
        Refit frequency for ML models.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ML-only forecasts and combined forecasts.
    """

    input_file = Path(input_path)
    statistical_file = Path(statistical_var_path)
    ml_output_file = Path(ml_output_path)
    combined_output_file = Path(combined_output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not statistical_file.exists():
        raise FileNotFoundError(
            f"Statistical VaR file not found: {statistical_file}. "
            "Run python src\\var_models.py first."
        )

    df = pd.read_csv(input_file, parse_dates=["date"])

    required_cols = [
        "date",
        "asset",
        "asset_name",
        "market_group",
        "log_return",
        "asset_high_vol_regime",
        "rolling_return_5d",
        "rolling_return_20d",
        "realised_vol_5d",
        "realised_vol_20d",
        "realised_vol_60d",
        "rolling_skew_20d",
        "rolling_kurt_20d",
        "rolling_max_drawdown_20d",
        "volume_change_5d",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values(["asset", "date"]).reset_index(drop=True)

    ml_forecast_frames = []

    for asset, group in df.groupby("asset", sort=False):
        print(f"\nBuilding ML quantile VaR forecasts for {asset}...")

        supervised = prepare_asset_supervised_data(group)

        forecast = fit_predict_quantile_var_for_asset(
            asset_df=supervised,
            min_train_size=min_train_size,
            refit_frequency=refit_frequency,
        )

        print(
            f"  Forecast rows: {len(forecast):,}, "
            f"95% violation rate: {forecast['violation_95'].mean():.4f}, "
            f"99% violation rate: {forecast['violation_99'].mean():.4f}"
        )

        ml_forecast_frames.append(forecast)

    ml_forecasts = pd.concat(ml_forecast_frames, ignore_index=True)

    ml_output_file.parent.mkdir(parents=True, exist_ok=True)
    ml_forecasts.to_csv(ml_output_file, index=False)

    statistical_forecasts = pd.read_csv(statistical_file, parse_dates=["date"])

    combined = pd.concat(
        [statistical_forecasts, ml_forecasts],
        ignore_index=True,
    )

    combined = combined.sort_values(["asset", "date", "model"]).reset_index(drop=True)

    combined_output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_output_file, index=False)

    print(f"\nSaved ML VaR forecasts to: {ml_output_file}")
    print(f"ML rows: {len(ml_forecasts):,}")
    print(f"ML assets: {ml_forecasts['asset'].nunique()}")

    print(f"\nSaved combined VaR forecasts to: {combined_output_file}")
    print(f"Combined rows: {len(combined):,}")
    print(f"Combined models: {combined['model'].nunique()}")

    summary = (
        combined.groupby(["market_group", "model"])
        .agg(
            observations=("date", "count"),
            violation_rate_95=("violation_95", "mean"),
            violation_rate_99=("violation_99", "mean"),
            mean_var_95=("var_95", "mean"),
            mean_var_99=("var_99", "mean"),
        )
        .reset_index()
    )

    print("\nCombined forecast summary:")
    print(summary.to_string(index=False))

    return ml_forecasts, combined


if __name__ == "__main__":
    build_ml_var_forecasts()