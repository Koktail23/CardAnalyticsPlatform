#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Segmentation (RFM + KMeans)
- Фикс INSERT: указываем список колонок, чтобы DEFAULT created_at отработал
- Опционально: MLflow; опционально Optuna для подбора k
"""

import os, json
from pathlib import Path
import logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("segmentation")

import numpy as np
import pandas as pd
from clickhouse_driver import Client
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# опционально
try:
    import mlflow
    MLFLOW_AVAILABLE=True
except Exception:
    MLFLOW_AVAILABLE=False
try:
    import optuna
    OPTUNA_AVAILABLE=True
except Exception:
    OPTUNA_AVAILABLE=False


def parse_args():
    import argparse
    ap=argparse.ArgumentParser("Customer Segmentation")
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE","card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST","localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT","9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER","analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD","admin123"))
    ap.add_argument("--models_path", default=os.getenv("MODELS_PATH","ml"))
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI","file:./mlruns"))
    ap.add_argument("--experiment", default="customer_segmentation")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=8)
    return ap.parse_args()


def ch(args)->Client:
    return Client(host=args.host, port=args.port, user=args.user, password=args.password, database=args.db,
                  connect_timeout=3, send_receive_timeout=10, sync_request_timeout=10)


def load_customer_features(cli: Client)->pd.DataFrame:
    sql = """
    SELECT hpan, txn_count_30d, txn_amount_30d, avg_txn_amount_30d, p2p_ratio_30d,
           unique_mcc_30d, unique_merchants_30d, weekend_ratio, night_txn_ratio,
           days_since_last_txn, max_daily_txn_count
    FROM card_features
    WHERE txn_count_30d > 0
    """
    cols = ['hpan','txn_count_30d','txn_amount_30d','avg_txn_amount_30d','p2p_ratio_30d',
            'unique_mcc_30d','unique_merchants_30d','weekend_ratio','night_txn_ratio',
            'days_since_last_txn','max_daily_txn_count']
    return pd.DataFrame(cli.execute(sql), columns=cols)


def rfm_scores(df: pd.DataFrame)->pd.DataFrame:
    # Recency: меньше — лучше (инвертируем через qcut с labels)
    try: df['R_score'] = pd.qcut(df['days_since_last_txn'], 4, labels=['4','3','2','1'], duplicates='drop')
    except: df['R_score'] = pd.cut(df['days_since_last_txn'], 4, labels=['4','3','2','1'])
    try: df['F_score'] = pd.qcut(df['txn_count_30d'].rank(method='first'), 4, labels=['1','2','3','4'], duplicates='drop')
    except: df['F_score'] = pd.cut(df['txn_count_30d'], 4, labels=['1','2','3','4'])
    try: df['M_score'] = pd.qcut(df['txn_amount_30d'].rank(method='first'), 4, labels=['1','2','3','4'], duplicates='drop')
    except: df['M_score'] = pd.cut(df['txn_amount_30d'], 4, labels=['1','2','3','4'])

    for c in ['R_score','F_score','M_score']: df[c]=df[c].astype(str).fillna('2')
    df['rfm_segment'] = df.apply(lambda r: (
        'Champions' if int(r['R_score'])>=3 and int(r['F_score'])>=3 and int(r['M_score'])>=3 else
        'Loyal'     if int(r['R_score'])>=3 and int(r['F_score'])>=2 and int(r['M_score'])>=2 else
        'Potential' if int(r['R_score'])>=3 and int(r['F_score'])>=1 else
        'At Risk'   if int(r['R_score'])>=2 and int(r['F_score'])>=3 else
        'Hibernating' if int(r['R_score'])>=2 and int(r['F_score'])>=1 else
        'Lost'
    ), axis=1)
    df['rfm_score']= df['R_score']+df['F_score']+df['M_score']
    return df


def choose_k(Xs: np.ndarray, kmin: int, kmax: int)->int:
    if not OPTUNA_AVAILABLE:
        return max(3, min(5, kmax))
    def objective(trial):
        k= trial.suggest_int("k", kmin, kmax)
        km=KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        labels=km.labels_
        if len(np.unique(labels))<2: return -1.0
        return silhouette_score(Xs, labels)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max(1, kmax-kmin+1))
    return study.best_params["k"]


def main():
    args=parse_args()
    MODELS_DIR=Path(args.models_path); MODELS_DIR.mkdir(exist_ok=True, parents=True)
    cli=ch(args)

    df=load_customer_features(cli)
    if df.empty:
        log.error("Нет данных для сегментации"); return

    df=rfm_scores(df)

    features = ['txn_count_30d','txn_amount_30d','avg_txn_amount_30d','p2p_ratio_30d',
                'unique_mcc_30d','weekend_ratio','days_since_last_txn']
    X = df[features].fillna(0.0).values
    scaler = StandardScaler().fit(X); Xs=scaler.transform(X)

    k = choose_k(Xs, args.kmin, args.kmax) if args.tune else 5
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    df["kmeans_cluster"] = km.labels_

    # MLflow
    if args.mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run():
            mlflow.log_params({"k":k, "rows": len(df), "features": len(features), "tune": args.tune})
            sizes = df["kmeans_cluster"].value_counts().sort_index()
            for cid, cnt in sizes.items():
                mlflow.log_metric(f"cluster_{cid}_size", int(cnt))
            # простая картинка распределения по кластерам
            fig, ax = plt.subplots(figsize=(6,3))
            sizes.plot(kind="bar", ax=ax, rot=0)
            ax.set_title(f"KMeans sizes (k={k})"); ax.set_xlabel("cluster"); ax.set_ylabel("count")
            plot_path = MODELS_DIR/"customer_clusters.png"; plt.tight_layout(); plt.savefig(plot_path, dpi=120); plt.close()
            mlflow.log_artifact(str(plot_path))
            mlflow.sklearn.log_model(km, "kmeans_model")
            mlflow.sklearn.log_model(scaler, "segment_scaler")

    # сохранить локально
    joblib.dump(km, MODELS_DIR/"kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR/"segment_scaler.pkl")

    # ClickHouse sink (фикс INSERT с указанием колонок)
    cli.execute("""
        CREATE TABLE IF NOT EXISTS customer_segments
        (hpan String, rfm_segment String, rfm_score String, kmeans_cluster UInt8, created_at DateTime DEFAULT now())
        ENGINE=MergeTree ORDER BY (hpan, created_at)
    """)
    sink = df[["hpan","rfm_segment","rfm_score","kmeans_cluster"]].values.tolist()
    cli.execute("TRUNCATE TABLE customer_segments")
    cli.execute("INSERT INTO customer_segments (hpan, rfm_segment, rfm_score, kmeans_cluster) VALUES", sink)

    print(json.dumps({"k": int(k), "rows": int(len(df))}, indent=2))
    print(f"[OK] Сегменты записаны: {len(df)} строк; артефакты -> {MODELS_DIR}")


if __name__=="__main__":
    main()
