#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Forecasting module with:
- Holdout metrics (MAPE, sMAPE, MASE)
- Confidence intervals
- Time series diagnostics
- Model ensemble
- Export and versioning
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


class ForecastMetrics:
    """Calculate various forecast accuracy metrics."""

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
        """Mean Absolute Scaled Error."""
        mae_forecast = mean_absolute_error(y_true, y_pred)

        # Naive seasonal forecast error on training data
        if len(y_train) > seasonality:
            mae_naive = mean_absolute_error(
                y_train[seasonality:],
                y_train[:-seasonality]
            )
        else:
            mae_naive = np.mean(np.abs(np.diff(y_train)))

        return mae_forecast / mae_naive if mae_naive > 0 else np.inf

    @staticmethod
    def coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
        """Prediction interval coverage."""
        in_interval = (y_true >= y_lower) & (y_true <= y_upper)
        return np.mean(in_interval) * 100


class TimeSeriesDiagnostics:
    """Diagnostic tools for time series analysis."""

    @staticmethod
    def stl_decompose(
            ts: pd.Series,
            period: int = 7,
            seasonal: int = 13
    ) -> Dict[str, pd.Series]:
        """STL decomposition into trend, seasonal, and residual components."""
        try:
            stl = STL(ts, period=period, seasonal=seasonal)
            result = stl.fit()

            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid,
                "strength_trend": 1 - np.var(result.resid) / np.var(result.trend + result.resid),
                "strength_seasonal": 1 - np.var(result.resid) / np.var(result.seasonal + result.resid)
            }
        except Exception as e:
            print(f"[WARN] STL decomposition failed: {e}")
            return {}

    @staticmethod
    def detect_anomalies(
            ts: pd.Series,
            threshold: float = 3.0
    ) -> pd.Series:
        """Detect anomalies using IQR method."""
        q1 = ts.quantile(0.25)
        q3 = ts.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        anomalies = (ts < lower_bound) | (ts > upper_bound)
        return anomalies

    @staticmethod
    def ljung_box_test(residuals: pd.Series, lags: int = 10) -> Dict:
        """Ljung-Box test for autocorrelation in residuals."""
        try:
            result = acorr_ljungbox(residuals, lags=lags, return_df=True)
            return {
                "statistic": result["lb_stat"].values[-1],
                "p_value": result["lb_pvalue"].values[-1],
                "autocorrelated": result["lb_pvalue"].values[-1] < 0.05
            }
        except Exception as e:
            print(f"[WARN] Ljung-Box test failed: {e}")
            return {}


class EnhancedForecaster:
    """Enhanced forecasting with multiple models and diagnostics."""

    def __init__(self, models_dir: str = "./ml"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = ForecastMetrics()
        self.diagnostics = TimeSeriesDiagnostics()

    def prepare_data(
            self,
            df: pd.DataFrame,
            date_col: str = "date",
            value_col: str = "volume",
            holdout_days: int = 14
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare train and holdout datasets."""
        # Ensure date column is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Split into train and holdout
        cutoff_date = df[date_col].max() - timedelta(days=holdout_days)

        train = df[df[date_col] <= cutoff_date].copy()
        holdout = df[df[date_col] > cutoff_date].copy()

        return train, holdout

    def seasonal_naive_forecast(
            self,
            train: pd.DataFrame,
            horizon: int,
            seasonality: int = 7,
            value_col: str = "volume"
    ) -> pd.DataFrame:
        """Simple seasonal naive forecast as baseline."""
        # Clean training data
        train_clean = train.dropna(subset=[value_col])

        if len(train_clean) < seasonality:
            # Not enough data for seasonal pattern
            print(f"[WARN] Not enough data for seasonality={seasonality}, using mean")
            mean_value = train_clean[value_col].mean()
            forecast_values = np.full(horizon, mean_value)
        else:
            last_season = train_clean[value_col].tail(seasonality).values

            # Check for NaN in last season
            if np.any(np.isnan(last_season)):
                last_season = np.nan_to_num(last_season, nan=np.nanmean(last_season))

            # Repeat pattern for horizon
            n_periods = (horizon // seasonality) + 1
            forecast = np.tile(last_season, n_periods)[:horizon]
            forecast_values = forecast

        # Create forecast dataframe
        last_date = train["date"].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq="D"
        )

        return pd.DataFrame({
            "date": forecast_dates,
            "yhat": forecast_values,
            "yhat_lower": forecast_values * 0.8,  # Simple 20% interval
            "yhat_upper": forecast_values * 1.2
        })

    def prophet_forecast(
            self,
            train: pd.DataFrame,
            horizon: int,
            value_col: str = "volume"
    ) -> Optional[pd.DataFrame]:
        """Prophet forecast with uncertainty intervals."""
        try:
            from prophet import Prophet

            # Prepare data for Prophet
            prophet_df = train.rename(columns={"date": "ds", value_col: "y"})

            # Fit model
            model = Prophet(
                interval_width=0.95,
                seasonality_mode="multiplicative",
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df[["ds", "y"]])

            # Make forecast
            future = model.make_future_dataframe(periods=horizon, freq="D")
            forecast = model.predict(future)

            # Get only future predictions
            forecast_future = forecast[forecast["ds"] > train["date"].max()]

            return pd.DataFrame({
                "date": forecast_future["ds"],
                "yhat": forecast_future["yhat"],
                "yhat_lower": forecast_future["yhat_lower"],
                "yhat_upper": forecast_future["yhat_upper"]
            })

        except ImportError:
            print("[WARN] Prophet not installed")
            return None
        except Exception as e:
            print(f"[WARN] Prophet forecast failed: {e}")
            return None

    def ensemble_forecast(
            self,
            forecasts: List[pd.DataFrame],
            weights: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Combine multiple forecasts using weighted average."""
        if not forecasts:
            raise ValueError("No forecasts to ensemble")

        if weights is None:
            weights = [1.0 / len(forecasts)] * len(forecasts)

        # Initialize with first forecast structure
        ensemble = forecasts[0].copy()
        ensemble["yhat"] = 0
        ensemble["yhat_lower"] = 0
        ensemble["yhat_upper"] = 0

        # Weighted average
        for forecast, weight in zip(forecasts, weights):
            ensemble["yhat"] += forecast["yhat"] * weight
            ensemble["yhat_lower"] += forecast["yhat_lower"] * weight
            ensemble["yhat_upper"] += forecast["yhat_upper"] * weight

        return ensemble

    def evaluate_forecast(
            self,
            forecast: pd.DataFrame,
            holdout: pd.DataFrame,
            train: pd.DataFrame,
            value_col: str = "volume"
    ) -> Dict:
        """Evaluate forecast on holdout data."""
        # Ensure date columns have the same type
        if 'date' in forecast.columns:
            forecast['date'] = pd.to_datetime(forecast['date'])
        if 'date' in holdout.columns:
            holdout['date'] = pd.to_datetime(holdout['date'])

        # Align forecast with holdout
        merged = holdout.merge(
            forecast,
            on="date",
            how="inner"
        )

        if len(merged) == 0:
            return {"error": "No overlapping dates"}

        y_true = merged[value_col].values
        y_pred = merged["yhat"].values

        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"error": "No valid data after removing NaN"}

        # Calculate metrics
        metrics = {}

        try:
            metrics["mape"] = self.metrics.mape(y_true, y_pred)
        except Exception as e:
            print(f"[WARN] MAPE calculation failed: {e}")
            metrics["mape"] = np.nan

        try:
            metrics["smape"] = self.metrics.smape(y_true, y_pred)
        except Exception as e:
            print(f"[WARN] sMAPE calculation failed: {e}")
            metrics["smape"] = np.nan

        try:
            # Clean train data for MASE
            train_values = train[value_col].values
            train_values = train_values[~np.isnan(train_values)]
            if len(train_values) > 0:
                metrics["mase"] = self.metrics.mase(y_true, y_pred, train_values)
            else:
                metrics["mase"] = np.nan
        except Exception as e:
            print(f"[WARN] MASE calculation failed: {e}")
            metrics["mase"] = np.nan

        try:
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
        except Exception as e:
            print(f"[WARN] MAE calculation failed: {e}")
            metrics["mae"] = np.nan

        try:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception as e:
            print(f"[WARN] RMSE calculation failed: {e}")
            metrics["rmse"] = np.nan

        # Add coverage if intervals exist
        if "yhat_lower" in merged.columns and "yhat_upper" in merged.columns:
            try:
                y_lower = merged["yhat_lower"].values[mask]
                y_upper = merged["yhat_upper"].values[mask]
                metrics["coverage_95"] = self.metrics.coverage(y_true, y_lower, y_upper)
            except Exception as e:
                print(f"[WARN] Coverage calculation failed: {e}")
                metrics["coverage_95"] = np.nan

        return metrics

    def rolling_origin_validation(
            self,
            df: pd.DataFrame,
            n_splits: int = 3,
            test_size: int = 14,
            value_col: str = "volume"
    ) -> pd.DataFrame:
        """Perform rolling origin cross-validation."""
        results = []

        total_days = len(df)
        step_size = test_size

        for i in range(n_splits):
            # Define train/test split
            test_end = total_days - i * step_size
            test_start = test_end - test_size

            if test_start < 30:  # Minimum training size
                break

            train = df.iloc[:test_start].copy()
            test = df.iloc[test_start:test_end].copy()

            # Generate forecasts
            naive_fc = self.seasonal_naive_forecast(train, len(test))
            prophet_fc = self.prophet_forecast(train, len(test), value_col)

            # Evaluate
            for name, fc in [("seasonal_naive", naive_fc), ("prophet", prophet_fc)]:
                if fc is not None:
                    metrics = self.evaluate_forecast(fc, test, train, value_col)
                    metrics["model"] = name
                    metrics["split"] = i + 1
                    results.append(metrics)

        return pd.DataFrame(results)

    def diagnose_series(
            self,
            ts: pd.Series,
            name: str = "series"
    ) -> Dict:
        """Comprehensive time series diagnostics."""
        diagnostics = {
            "name": name,
            "length": len(ts),
            "missing": ts.isna().sum(),
            "mean": ts.mean(),
            "std": ts.std(),
            "cv": ts.std() / ts.mean() if ts.mean() != 0 else np.inf
        }

        # STL decomposition
        stl_result = self.diagnostics.stl_decompose(ts)
        if stl_result:
            diagnostics["trend_strength"] = stl_result.get("strength_trend", 0)
            diagnostics["seasonal_strength"] = stl_result.get("strength_seasonal", 0)

        # Anomalies
        anomalies = self.diagnostics.detect_anomalies(ts)
        diagnostics["anomaly_rate"] = anomalies.mean() * 100
        diagnostics["anomaly_dates"] = ts.index[anomalies].tolist()

        return diagnostics

    def export_forecast(
            self,
            forecast: pd.DataFrame,
            metrics: Dict,
            diagnostics: Dict,
            name: str = "forecast"
    ) -> None:
        """Export forecast with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save forecast
        forecast_file = self.models_dir / f"{name}_{timestamp}.csv"
        forecast.to_csv(forecast_file, index=False)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "name": name,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "forecast_file": str(forecast_file),
            "horizon": len(forecast),
            "date_range": {
                "start": forecast["date"].min().isoformat(),
                "end": forecast["date"].max().isoformat()
            }
        }

        metadata_file = self.models_dir / f"{name}_metadata_{timestamp}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"[INFO] Exported forecast to {forecast_file}")
        print(f"[INFO] Exported metadata to {metadata_file}")

        # Log to MLflow if available
        try:
            import mlflow

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
            mlflow.set_experiment("forecasting")

            with mlflow.start_run():
                # Log metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name, value)

                # Log parameters
                mlflow.log_param("horizon", len(forecast))
                mlflow.log_param("model", name)

                # Log artifacts
                mlflow.log_artifact(forecast_file)
                mlflow.log_artifact(metadata_file)

                print("[INFO] Logged to MLflow")
        except Exception as e:
            print(f"[WARN] MLflow logging failed: {e}")


# CLI interface
if __name__ == "__main__":
    import argparse
    from clickhouse_driver import Client

    parser = argparse.ArgumentParser(description="Enhanced forecasting")
    parser.add_argument("--table", default="transactions_optimized", help="Source table")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon in days")
    parser.add_argument("--holdout", type=int, default=14, help="Holdout period for evaluation")
    args = parser.parse_args()

    # Get data from ClickHouse
    client = Client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
        user=os.getenv("CLICKHOUSE_USER", "analyst"),
        password=os.getenv("CLICKHOUSE_PASSWORD", "admin123"),
        database=os.getenv("CLICKHOUSE_DATABASE", "card_analytics")
    )

    # Query daily volumes
    query = f"""
    SELECT 
        toDate(transaction_date) AS date,
        sum(amount_uzs) AS volume,
        count() AS transactions
    FROM {args.table}
    WHERE transaction_date >= today() - INTERVAL 90 DAY
    GROUP BY date
    ORDER BY date
    """

    rows = client.execute(query)
    df = pd.DataFrame(rows, columns=["date", "volume", "transactions"])

    print(f"[INFO] Loaded {len(df)} days of data")

    # Initialize forecaster
    forecaster = EnhancedForecaster()

    # Prepare data
    train, holdout = forecaster.prepare_data(df, holdout_days=args.holdout)
    print(f"[INFO] Train: {len(train)} days, Holdout: {len(holdout)} days")

    # Diagnose series
    diagnostics = forecaster.diagnose_series(train["volume"], name="volume")
    print(f"[INFO] Series diagnostics:")
    print(f"  - Trend strength: {diagnostics.get('trend_strength', 0):.2f}")
    print(f"  - Seasonal strength: {diagnostics.get('seasonal_strength', 0):.2f}")
    print(f"  - Anomaly rate: {diagnostics.get('anomaly_rate', 0):.1f}%")

    # Generate forecasts
    print(f"[INFO] Generating {args.horizon}-day forecasts...")

    naive_forecast = forecaster.seasonal_naive_forecast(train, args.horizon)
    prophet_forecast = forecaster.prophet_forecast(train, args.horizon)

    # Ensemble
    forecasts = [naive_forecast]
    if prophet_forecast is not None:
        forecasts.append(prophet_forecast)

    ensemble = forecaster.ensemble_forecast(forecasts)

    # Evaluate on holdout
    if len(holdout) > 0:
        metrics = forecaster.evaluate_forecast(ensemble, holdout, train)
        print(f"[INFO] Holdout metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.2f}")
    else:
        metrics = {}
        print("[INFO] No holdout data for evaluation")

    # Rolling validation
    print("[INFO] Running rolling origin validation...")
    cv_results = forecaster.rolling_origin_validation(df, n_splits=3)
    if not cv_results.empty:
        print("\n[INFO] Cross-validation results:")
        print(cv_results.groupby("model")[["mape", "smape", "mase"]].mean())

    # Export
    forecaster.export_forecast(ensemble, metrics, diagnostics, name="ensemble")

    print("\n[SUCCESS] Forecasting complete!")