#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_thresholds.py — калибровка вероятностей (Isotonic) + бизнес-порог по EV.
Совместим с фичами тренера (18 признаков) и со скейлером fraud_scaler.pkl.

Сохраняет:
  - MODELS_PATH/calibration_isotonic.pkl
  - MODELS_PATH/thresholds.json  (ev_optimal + пороги на FPR ~1% и ~2%)

Запуск:
  python ml\\calibrate_thresholds.py --mlflow
"""

import os
import json
import argparse
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from clickhouse_driver import Client
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler
import joblib

# MLflow (опционально)
HAS_MLFLOW = True
try:
    import mlflow
except Exception:
    HAS_MLFLOW = False


def parse_args():
    ap = argparse.ArgumentParser("calibrate_thresholds")
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE", "card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT", "9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER", "analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD", "admin123"))
    ap.add_argument("--view", default=os.getenv("FRAUD_VIEW", "fraud_dataset_v1"))
    ap.add_argument("--valid_days", type=int, default=30)
    ap.add_argument("--models_path", default=os.getenv("MODELS_PATH", "ml"))
    ap.add_argument("--model_file", default=None, help="путь к .pkl (по умолчанию MODELS_PATH/fraud_xgboost.pkl)")
    # бизнес-веса
    ap.add_argument("--gain_tp", type=float, default=1.0, help="выгода от перехваченного фрода")
    ap.add_argument("--cost_fp", type=float, default=0.15, help="стоимость ложного срабатывания")
    # MLflow
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    ap.add_argument("--experiment", default="fraud_calibration")
    return ap.parse_args()


def ch(args) -> Client:
    return Client(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.db,
        connect_timeout=3,
        send_receive_timeout=10,
        sync_request_timeout=10,
        settings={"max_execution_time": 60},
    )


def max_date(cli: Client, fqtn: str) -> pd.Timestamp:
    d = cli.execute(f"SELECT max(toDate(transaction_date)) FROM {fqtn}")[0][0]
    return pd.to_datetime(d)


def fetch_valid_txn(cli: Client, fqtn: str, start, end) -> pd.DataFrame:
    sql = f"""
    SELECT
      toString(hpan) AS hpan,
      toDateTime(transaction_date) AS transaction_date,
      COALESCE(toUInt32OrNull(toString(hour_num)), toHour(toDateTime(transaction_date))) AS hour_num,
      toFloat64OrNull(toString(amount_uzs)) AS amount_uzs,
      coalesce(toUInt32OrNull(extract(toString(mcc), '\\\\d+')), 0) AS mcc,
      COALESCE(toUInt8OrNull(toString(p2p_flag)), 0) AS p2p_flag,
      toString(emitent_region) AS emitent_region,
      toInt32(fraud_proxy) AS fraud_proxy
    FROM {fqtn}
    WHERE toDate(transaction_date) >= toDate('{start.date():%Y-%m-%d}')
      AND toDate(transaction_date) <= toDate('{end.date():%Y-%m-%d}')
    """
    rows = cli.execute(sql)
    cols = ["hpan","transaction_date","hour_num","amount_uzs","mcc","p2p_flag","emitent_region","fraud_proxy"]
    return pd.DataFrame(rows, columns=cols)


def fetch_card_features(cli: Client) -> pd.DataFrame:
    try:
        rows = cli.execute(
            """
            SELECT 
                hpan, txn_count_30d, txn_amount_30d, avg_txn_amount_30d, p2p_ratio_30d,
                unique_mcc_30d, unique_merchants_30d, weekend_ratio, night_txn_ratio
            FROM card_features
            """
        )
        return pd.DataFrame(
            rows,
            columns=[
                "hpan",
                "txn_count_30d",
                "txn_amount_30d",
                "avg_txn_amount_30d",
                "p2p_ratio_30d",
                "unique_mcc_30d",
                "unique_merchants_30d",
                "weekend_ratio",
                "night_txn_ratio",
            ],
        )
    except Exception:
        return pd.DataFrame(columns=["hpan"])


FEATURES: List[str] = [
    "amount_uzs",
    "hour_num",
    "p2p_flag",
    "days_since_prev_txn",
    "amount_change_ratio",
    "amount_deviation",
    "txn_hour_count",
    "is_risky_mcc",
    "is_weekend",
    "is_night",
    "is_capital",
    "txn_count_30d",
    "avg_txn_amount_30d",
    "p2p_ratio_30d",
    "unique_mcc_30d",
    "unique_merchants_30d",
    "weekend_ratio",
    "night_txn_ratio",
]


def build_features(df_txn: pd.DataFrame, df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    df = df_txn.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values(["hpan", "transaction_date"]).reset_index(drop=True)

    df["prev_txn_date"] = df.groupby("hpan")["transaction_date"].shift(1)
    df["days_since_prev_txn"] = (df["transaction_date"] - df["prev_txn_date"]).dt.days.fillna(999)

    df["prev_amount"] = df.groupby("hpan")["amount_uzs"].shift(1)
    df["amount_change_ratio"] = (df["amount_uzs"] / df["prev_amount"].replace(0, 1)).fillna(1)

    df["card_avg_amount"] = df.groupby("hpan")["amount_uzs"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    df["amount_deviation"] = df["amount_узs".replace("уз","uz")] / df["card_avg_amount"].replace(0, 1)

    df["txn_hour_count"] = df.groupby(["hpan", df["transaction_date"].dt.date, "hour_num"]).cumcount() + 1

    df["is_risky_mcc"] = df["mcc"].isin([6010, 6011, 6012, 7995]).astype(int)
    df["is_weekend"] = df["transaction_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour_num"] >= 22) | (df["hour_num"] <= 6)).astype(int)
    df["is_capital"] = (df.get("emitent_region", "") == "Tashkent City").astype(int) if "emitent_region" in df.columns else 0

    if not df_feat.empty:
        df = df.merge(df_feat, on="hpan", how="left")

    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0

    X = df[FEATURES].astype(float).fillna(0.0)
    y = df["fraud_proxy"].astype(int).values
    return X, y


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, scores)
    if len(fpr) == 0:
        return 0.0, 0.5
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx]), float(thr[idx])


def expected_value(y_true: np.ndarray, scores: np.ndarray, thr: float, gain_tp: float, cost_fp: float) -> float:
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return gain_tp * tp - cost_fp * fp


def main():
    args = parse_args()
    MODELS = Path(args.models_path)
    MODELS.mkdir(exist_ok=True, parents=True)

    cli = ch(args)
    fqtn = f"{args.db}.{args.view}"
    mx = max_date(cli, fqtn)
    valid_start = mx - timedelta(days=args.valid_days)
    valid_end = mx

    df_txn = fetch_valid_txn(cli, fqtn, valid_start, valid_end)
    if df_txn.empty:
        print("[ERR] Валидационное окно пустое"); return

    df_feat = fetch_card_features(cli)
    X, y = build_features(df_txn, df_feat)

    model_path = Path(args.model_file) if args.model_file else (MODELS / "fraud_xgboost.pkl")
    if not model_path.exists():
        print(f"[ERR] Модель не найдена: {model_path}"); return
    model = joblib.load(model_path)

    scaler_path = MODELS / "fraud_scaler.pkl"
    if scaler_path.exists():
        scaler: StandardScaler = joblib.load(scaler_path)
        X_pred = scaler.transform(X.values)
    else:
        X_pred = X.values

    raw_scores = model.predict_proba(X_pred)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_scores, y)
    cal_scores = iso.transform(raw_scores)

    brier_raw = float(brier_score_loss(y, raw_scores))
    brier_cal = float(brier_score_loss(y, cal_scores))

    r1, thr1 = recall_at_fpr(y, cal_scores, 0.01)
    r2, thr2 = recall_at_fpr(y, cal_scores, 0.02)

    grid = np.linspace(0.01, 0.99, 200)
    evs = [expected_value(y, cal_scores, t, args.gain_tp, args.cost_fp) for t in grid]
    best_idx = int(np.argmax(evs))
    thr_best = float(grid[best_idx])
    ev_best = float(evs[best_idx])

    out = {
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "thresholds": {
            "fpr_1pct": {"thr": float(thr1), "recall": float(r1)},
            "fpr_2pct": {"thr": float(thr2), "recall": float(r2)},
            "ev_optimal": {"thr": thr_best, "ev": ev_best, "gain_tp": float(args.gain_tp), "cost_fp": float(args.cost_fp)},
        },
        "features_used": FEATURES,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path) if scaler_path.exists() else None,
        "valid_window": {"start": str(valid_start.date()), "end": str(valid_end.date())},
    }

    joblib.dump(iso, MODELS / "calibration_isotonic.pkl")
    (MODELS / "thresholds.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.mlflow and HAS_MLFLOW:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run():
            mlflow.log_metric("brier_raw", brier_raw)
            mlflow.log_metric("brier_calibrated", brier_cal)
            mlflow.log_metric("recall_at_fpr1", float(r1))
            mlflow.log_metric("recall_at_fpr2", float(r2))
            mlflow.log_metric("ev_best", ev_best)
            mlflow.log_param("thr_fpr1", float(thr1))
            mlflow.log_param("thr_fpr2", float(thr2))
            mlflow.log_param("thr_ev", float(thr_best))
            mlflow.log_param("gain_tp", float(args.gain_tp))
            mlflow.log_param("cost_fp", float(args.cost_fp))
            mlflow.log_param("model_path", str(model_path))
            if scaler_path.exists():
                mlflow.log_param("scaler_path", str(scaler_path))
            mlflow.log_artifact(str(MODELS / "thresholds.json"))
            mlflow.log_artifact(str(MODELS / "calibration_isotonic.pkl"))


if __name__ == "__main__":
    main()
