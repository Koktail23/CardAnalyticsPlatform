#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraud Detection Model (XGBoost) — динамическая загрузка, MLflow (опц.), Optuna (опц.)
+ интеграция граф-признаков:
  - graph_card_features (по hpan, последние 90д)
  - graph_merchant_features (по merchant_key = merchant id или lower(merchant_name))
"""

import os
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from clickhouse_driver import Client

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import joblib
import logging

# опционально
try:
    import mlflow, mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fraud_detection")


def parse_args():
    ap = argparse.ArgumentParser("Fraud Detection training")
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE", "card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT", "9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER", "analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD", "admin123"))
    ap.add_argument("--view", default=os.getenv("FRAUD_VIEW", "fraud_dataset_v1"))
    ap.add_argument("--limit", type=int, default=400_000)
    ap.add_argument("--models_path", default=os.getenv("MODELS_PATH", "ml"))
    # MLflow / Optuna
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    ap.add_argument("--experiment", default="fraud_detection")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=0)
    return ap.parse_args()


def ch_client(args) -> Client:
    return Client(
        host=args.host, port=args.port, user=args.user, password=args.password, database=args.db,
        connect_timeout=3, send_receive_timeout=10, sync_request_timeout=10,
        settings={"max_execution_time": 180}
    )


def describe(cli: Client, fqtn: str) -> Dict[str, str]:
    return {r[0]: r[1] for r in cli.execute(f"DESCRIBE {fqtn}")}


def fetch_txn(cli: Client, fqtn: str, limit: int) -> pd.DataFrame:
    cols = describe(cli, fqtn)
    get = cols.__contains__

    hour_expr = "toUInt32OrNull(toString(hour_num))" if get("hour_num") else "toHour(toDateTime(transaction_date))"
    p2p_expr  = "toUInt8OrNull(toString(p2p_flag))"  if get("p2p_flag")  else "toUInt8(0)"
    mcc_expr  = "coalesce(toUInt32OrNull(extract(toString(mcc), '\\\\d+')), 0)"
    region_expr = "toString(emitent_region)" if get("emitent_region") else "toString('')"
    merch_name_expr = "toString(merchant_name)" if get("merchant_name") else "toString('')"
    resp_expr = "toString(respcode)" if get("respcode") else "toString('')"
    rev_expr  = "toString(reversal_flag)" if get("reversal_flag") else "toString('0')"

    sql = f"""
    SELECT
      toString(hpan) AS hpan,
      toDateTime(transaction_date) AS transaction_date,
      {hour_expr} AS hour_num,
      toFloat64OrNull(toString(amount_uzs)) AS amount_uzs,
      {mcc_expr} AS mcc,
      {p2p_expr} AS p2p_flag,
      {merch_name_expr} AS merchant_name,
      {region_expr} AS emitent_region,
      {resp_expr} AS respcode,
      {rev_expr} AS reversal_flag,
      toInt32(fraud_proxy) AS fraud_proxy
    FROM {fqtn}
    WHERE toFloat64OrNull(toString(amount_uzs)) >= 0
    LIMIT {int(limit)}
    """
    rows = cli.execute(sql)
    cols_out = ["hpan","transaction_date","hour_num","amount_uzs","mcc","p2p_flag",
                "merchant_name","emitent_region","respcode","reversal_flag","fraud_proxy"]
    return pd.DataFrame(rows, columns=cols_out)


def load_graph_card_features(cli: Client) -> pd.DataFrame:
    try:
        end_date = cli.execute("SELECT max(window_end) FROM card_analytics.graph_card_features")[0][0]
        if end_date is None:
            return pd.DataFrame(columns=["hpan"])
        rows = cli.execute(f"""
            SELECT hpan,
                   degree_merchants_90d, txn_count_90d, avg_amount_90d, max_amount_90d,
                   neighbor_fraud_merchants_90d, neighbor_fraud_ratio_90d
            FROM card_analytics.graph_card_features
            WHERE window_end = toDate('{end_date:%Y-%m-%d}')
        """)
        return pd.DataFrame(rows, columns=[
            "hpan",
            "card_degree_merchants_90d","card_txn_count_90d","card_avg_amount_90d","card_max_amount_90d",
            "card_neighbor_fraud_merchants_90d","card_neighbor_fraud_ratio_90d"
        ])
    except Exception:
        return pd.DataFrame(columns=["hpan"])


def load_graph_merchant_features(cli: Client) -> pd.DataFrame:
    try:
        end_date = cli.execute("SELECT max(window_end) FROM card_analytics.graph_merchant_features")[0][0]
        if end_date is None:
            return pd.DataFrame(columns=["merchant_key"])
        rows = cli.execute(f"""
            SELECT merchant_key, degree_cards_90d, txn_count_90d, fraud_ratio_90d
            FROM card_analytics.graph_merchant_features
            WHERE window_end = toDate('{end_date:%Y-%m-%d}')
        """)
        return pd.DataFrame(rows, columns=[
            "merchant_key","merch_degree_cards_90d","merch_txn_count_90d","merch_fraud_ratio_90d"
        ])
    except Exception:
        return pd.DataFrame(columns=["merchant_key"])


def build_features(df: pd.DataFrame,
                   df_card_g: pd.DataFrame,
                   df_merch_g: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    # base velocity & binary
    df = df.sort_values(['hpan','transaction_date']).copy()
    df['prev_txn_date'] = df.groupby('hpan')['transaction_date'].shift(1)
    df['days_since_prev_txn'] = (df['transaction_date'] - df['prev_txn_date']).dt.days.fillna(999)

    df['prev_amount'] = df.groupby('hpan')['amount_uzs'].shift(1)
    df['amount_change_ratio'] = (df['amount_uzs'] / df['prev_amount'].replace(0, 1)).fillna(1)

    df['card_avg_amount'] = df.groupby('hpan')['amount_uzs'].transform(lambda x: x.rolling(10,1).mean())
    df['amount_deviation'] = df['amount_uzs'] / df['card_avg_amount'].replace(0, 1)

    df['txn_hour_count'] = df.groupby(['hpan', df['transaction_date'].dt.date, 'hour_num']).cumcount() + 1

    df['is_risky_mcc'] = df['mcc'].isin([6010,6011,6012,7995]).astype(int)
    df['is_weekend']   = df['transaction_date'].dt.dayofweek.isin([5,6]).astype(int)
    df['is_night']     = ((df['hour_num'] >= 22) | (df['hour_num'] <= 6)).astype(int)
    df['is_capital']   = (df.get('emitent_region','') == 'Tashkent City').astype(int) if 'emitent_region' in df.columns else 0

    # graph features: card
    if not df_card_g.empty:
        df = df.merge(df_card_g, on="hpan", how="left")
    else:
        df["card_degree_merchants_90d"]=0
        df["card_txn_count_90d"]=0
        df["card_avg_amount_90d"]=0.0
        df["card_max_amount_90d"]=0.0
        df["card_neighbor_fraud_merchants_90d"]=0
        df["card_neighbor_fraud_ratio_90d"]=0.0

    # merchant_key по текстовому имени (если нет числового id)
    df["merchant_key"] = df["merchant_name"].fillna("").str.lower()

    # graph features: merchant
    if not df_merch_g.empty:
        df = df.merge(df_merch_g, on="merchant_key", how="left")
    else:
        df["merch_degree_cards_90d"]=0
        df["merch_txn_count_90d"]=0
        df["merch_fraud_ratio_90d"]=0.0

    # финальный список фич (18 базовых + 8 графовых = 26)
    features = [
        'amount_uzs','hour_num','p2p_flag',
        'days_since_prev_txn','amount_change_ratio','amount_deviation',
        'txn_hour_count','is_risky_mcc','is_weekend','is_night','is_capital',
        # card_features (если были)
        'txn_count_30d','avg_txn_amount_30d','p2p_ratio_30d',
        'unique_mcc_30d','unique_merchants_30d','weekend_ratio','night_txn_ratio',
        # graph: card
        'card_degree_merchants_90d','card_txn_count_90d','card_avg_amount_90d','card_max_amount_90d',
        'card_neighbor_fraud_merchants_90d','card_neighbor_fraud_ratio_90d',
        # graph: merchant
        'merch_degree_cards_90d','merch_txn_count_90d','merch_fraud_ratio_90d'
    ]
    for c in features:
        if c not in df.columns:
            df[c]=0
    X = df[features].astype(float).fillna(0.0)
    y = df['fraud_proxy'].astype(int).values
    return X, y, features


def train_xgb(X: pd.DataFrame, y: np.ndarray, do_tune: bool, n_trials: int, timeout: int):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr); Xva_s = scaler.transform(Xva)

    scale_pos_weight = max(1.0, (ytr==0).sum() / max(1,(ytr==1).sum()))
    params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, tree_method="hist",
        random_state=42, eval_metric="auc"
    )
    if do_tune and OPTUNA_AVAILABLE:
        def objective(trial):
            p = params.copy()
            p.update(
                max_depth = trial.suggest_int("max_depth", 3, 10),
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                n_estimators = trial.suggest_int("n_estimators", 300, 1200),
                subsample = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0),
                reg_lambda = trial.suggest_float("reg_lambda", 0.5, 5.0),
            )
            m = xgb.XGBClassifier(**p)
            m.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False)
            pva = m.predict_proba(Xva_s)[:,1]
            return roc_auc_score(yva, pva)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout if timeout>0 else None)
        params.update(study.best_params)
        log.info(f"Optuna best params: {study.best_params}")

    model = xgb.XGBClassifier(**params)
    model.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False)
    pva = model.predict_proba(Xva_s)[:,1]
    yhat = (pva>=0.5).astype(int)
    auc = roc_auc_score(yva, pva) if len(np.unique(yva))>1 else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(yva, yhat, average="binary", zero_division=0)

    cm = confusion_matrix(yva, yhat)
    plt.figure(figsize=(4,3)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("CM"); plt.tight_layout()
    outdir = Path(os.getenv("MODELS_PATH","ml")); outdir.mkdir(exist_ok=True, parents=True)
    cm_path = outdir/"cm_fraud.png"; plt.savefig(cm_path, dpi=120); plt.close()

    metrics = dict(auc=float(auc), precision=float(prec), recall=float(rec), f1=float(f1))
    return model, scaler, metrics, cm_path


def main():
    args = parse_args()
    cli = ch_client(args)
    fqtn = f"{args.db}.{args.view}"
    MODELS = Path(args.models_path); MODELS.mkdir(exist_ok=True, parents=True)

    # Загружаем транзакции
    df_txn = fetch_txn(cli, fqtn, args.limit)
    if df_txn.empty:
        print("[ERR] Нет данных для обучения")
        return

    # Граф-фичи
    df_card_g = load_graph_card_features(cli)
    df_merch_g = load_graph_merchant_features(cli)

    # Собираем финальные фичи
    X, y, feat_list = build_features(df_txn, df_card_g, df_merch_g)

    # MLflow
    use_mlflow = args.mlflow and MLFLOW_AVAILABLE
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        run = mlflow.start_run()
    else:
        run = None

    try:
        if use_mlflow:
            mlflow.log_params({"rows": len(X), "features": len(feat_list), "tune": args.tune, "n_trials": args.n_trials})

        model, scaler, metrics, cm_path = train_xgb(X.values, y, args.tune, args.n_trials, args.timeout)

        if use_mlflow:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_param("features", ",".join(feat_list))
            mlflow.sklearn.log_model(model, "xgb_model")

        joblib.dump(model,  MODELS/"fraud_xgboost.pkl")
        joblib.dump(scaler, MODELS/"fraud_scaler.pkl")
        print(json.dumps(metrics, indent=2))

    finally:
        if run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
