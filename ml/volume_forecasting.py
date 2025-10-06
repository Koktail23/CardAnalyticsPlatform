#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прогнозирование объёмов: Prophet (если установлен) / наивный MA.
+ MLflow (опц.): лог метрик, forecast.csv, графики.
"""

import os
import argparse
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from clickhouse_driver import Client

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# опционально
try:
    import mlflow
    MLFLOW_AVAILABLE=True
except Exception:
    MLFLOW_AVAILABLE=False

def parse_args():
    ap=argparse.ArgumentParser("Volume Forecasting")
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE","card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST","localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT","9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER","analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD","admin123"))
    ap.add_argument("--table", default=os.getenv("TRAIN_TABLE","transactions_optimized"))
    ap.add_argument("--models_path", default=os.getenv("MODELS_PATH","ml"))
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI","file:./mlruns"))
    ap.add_argument("--experiment", default="volume_forecasting")
    return ap.parse_args()

def ch(args)->Client:
    return Client(host=args.host, port=args.port, user=args.user, password=args.password, database=args.db,
                  connect_timeout=3, send_receive_timeout=10, sync_request_timeout=10)

def load_daily(cli: Client, table_fqn: str)->pd.DataFrame:
    sql = f"""
    SELECT transaction_date AS ds, sum(amount_uzs) AS y
    FROM {table_fqn}
    WHERE transaction_date IS NOT NULL
    GROUP BY ds ORDER BY ds
    """
    df = pd.DataFrame(cli.execute(sql), columns=["ds","y"])
    df["ds"]=pd.to_datetime(df["ds"])
    return df

def naive_forecast(df: pd.DataFrame, horizon: int=30)->pd.DataFrame:
    ma = df['y'].rolling(7, min_periods=1).mean().iloc[-1]
    future = pd.date_range(df['ds'].max()+pd.Timedelta(days=1), periods=horizon)
    out = pd.DataFrame({"ds": future, "yhat": ma, "yhat_lower": ma*0.9, "yhat_upper": ma*1.1})
    return out

def main():
    args=parse_args()
    cli=ch(args)
    MODELS_DIR=Path(args.models_path); MODELS_DIR.mkdir(exist_ok=True, parents=True)

    df=load_daily(cli, f"{args.db}.{args.table}")
    if df.empty:
        print("[ERR] Нет данных для прогноза"); return

    use_mlflow = args.mlflow and MLFLOW_AVAILABLE
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
    run_ctx = (mlflow.start_run() if use_mlflow else None)

    try:
        try:
            from prophet import Prophet
            model=Prophet(weekly_seasonality=True, yearly_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=30)
            fc = model.predict(future)
            # сохраняем будущие 30 дней
            last_date = df["ds"].max()
            out = fc[fc["ds"]>last_date][["ds","yhat","yhat_lower","yhat_upper"]].copy()
            out.to_csv(MODELS_DIR/"forecast.csv", index=False)
            # простая картинка
            plt.figure(figsize=(8,4))
            plt.plot(df["ds"], df["y"], label="history")
            plt.plot(out["ds"], out["yhat"], label="forecast")
            plt.legend(); plt.tight_layout()
            plot_path = MODELS_DIR/"forecast_plot.png"; plt.savefig(plot_path, dpi=120); plt.close()
            if use_mlflow:
                mlflow.log_artifact(str(MODELS_DIR/"forecast.csv"))
                mlflow.log_artifact(str(plot_path))
        except Exception as e:
            print(f"[WARN] Prophet недоступен ({e}); используем наивный прогноз")
            out = naive_forecast(df, 30)
            out.to_csv(MODELS_DIR/"forecast.csv", index=False)
            if use_mlflow: mlflow.log_artifact(str(MODELS_DIR/"forecast.csv"))

        print(f"[OK] forecast.csv -> {MODELS_DIR/'forecast.csv'}")
    finally:
        if run_ctx: mlflow.end_run()

if __name__=="__main__":
    main()
