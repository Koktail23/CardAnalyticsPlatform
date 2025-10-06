#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fraud_benchmark.py — бенчмарк бустингов на ClickHouse-датасете:
XGBoost + LightGBM + CatBoost (опционально), Optuna-тюнинг, лог в MLflow.

Запуск:
  python ml/fraud_benchmark.py --mlflow --n_trials 20
  # только XGB:         --models xgb
  # XGB + LGBM:         --models xgb,lgbm
  # XGB + LGBM + CBT:   --models xgb,lgbm,cat

Зависимости: xgboost, lightgbm, catboost (опц.), optuna, mlflow, scikit-learn, clickhouse-driver
"""

import os, argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import timedelta

import numpy as np
import pandas as pd
from clickhouse_driver import Client
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
import joblib

# MLflow/Optuna — опционально
HAS_MLFLOW = True
try:
    import mlflow
except Exception:
    HAS_MLFLOW = False

HAS_OPTUNA = True
try:
    import optuna
except Exception:
    HAS_OPTUNA = False

# Модели — некоторые могут быть не установлены
HAS_LGBM = True
HAS_CAT  = True
try:
    import lightgbm as lgb
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier, Pool
except Exception:
    HAS_CAT = False

import xgboost as xgb


def parse_args():
    ap = argparse.ArgumentParser("fraud_benchmark")
    # ClickHouse
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE", "card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT", "9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER", "analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD", "admin123"))
    ap.add_argument("--view", default=os.getenv("FRAUD_VIEW", "fraud_dataset_v1"))
    # окна
    ap.add_argument("--train_days", type=int, default=120)
    ap.add_argument("--valid_days", type=int, default=30)
    # какие модели
    ap.add_argument("--models", default="xgb,lgbm,cat", help="comma: xgb,lgbm,cat")
    # Optuna/MLflow
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow_uri", default=os.getenv("MLFLOW_TRACKING_URI","file:./mlruns"))
    ap.add_argument("--experiment", default="fraud_benchmark")
    # артефакты
    ap.add_argument("--models_path", default=os.getenv("MODELS_PATH","ml"))
    return ap.parse_args()


def ch_client(args) -> Client:
    return Client(
        host=args.host, port=args.port, user=args.user, password=args.password, database=args.db,
        connect_timeout=3, send_receive_timeout=15, sync_request_timeout=15,
        settings={"max_execution_time": 120},
    )


def max_date(cli: Client, fqtn: str) -> pd.Timestamp:
    d = cli.execute(f"SELECT max(toDate(transaction_date)) FROM {fqtn}")[0][0]
    return pd.to_datetime(d)


def fetch_window(cli: Client, fqtn: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sql = f"""
    SELECT hpan, transaction_code, transaction_date, hour_num, amount_uzs, mcc, p2p_flag,
           merchant_name, emitent_bank, emitent_region, respcode, reversal_flag, fraud_proxy
    FROM {fqtn}
    WHERE toDate(transaction_date) >= toDate('{start.date():%Y-%m-%d}')
      AND toDate(transaction_date) <= toDate('{end.date():%Y-%m-%d}')
    """
    rows = cli.execute(sql)
    cols = ['hpan','transaction_code','transaction_date','hour_num','amount_uzs','mcc','p2p_flag',
            'merchant_name','emitent_bank','emitent_region','respcode','reversal_flag','fraud_proxy']
    df = pd.DataFrame(rows, columns=cols)
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    # Базовая очистка
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values(['hpan','transaction_date']).copy()

    # velocity
    df['prev_date'] = df.groupby('hpan')['transaction_date'].shift(1)
    df['days_since_prev'] = (df['transaction_date'] - df['prev_date']).dt.days.fillna(999)

    df['prev_amt'] = df.groupby('hpan')['amount_uzs'].shift(1)
    df['amt_change_ratio'] = (df['amount_узs'.replace('уз','uz')] / df['prev_amt'].replace(0, 1)).fillna(1)

    df['card_avg_amt'] = df.groupby('hpan')['amount_uzs'].transform(lambda x: x.rolling(10, 1).mean())
    df['amt_deviation'] = (df['amount_uzs'] / df['card_avg_amt'].replace(0, 1)).fillna(1)

    df['txn_hour_count'] = df.groupby(['hpan', df['transaction_date'].dt.date, 'hour_num']).cumcount() + 1
    df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5,6]).astype(int)
    df['is_night']   = ((df['hour_num'] >= 22) | (df['hour_num'] <= 6)).astype(int)
    df['is_risky_mcc'] = df['mcc'].isin([6010,6011,6012,7995]).astype(int)
    df['is_capital'] = (df.get('emitent_region','') == 'Tashkent City').astype(int) if 'emitent_region' in df.columns else 0

    feats = [
        'amount_uzs','hour_num','p2p_flag','mcc',
        'days_since_prev','amt_change_ratio','amt_deviation',
        'txn_hour_count','is_weekend','is_night','is_risky_mcc','is_capital'
    ]
    for c in feats:
        if c not in df.columns: df[c] = 0
    X = df[feats].astype(float).fillna(0.0)
    y = df['fraud_proxy'].astype(int).values
    return X, y, feats


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y_true, scores)
    if len(fpr) == 0:
        return 0.0, 0.5
    idx = np.argmin(np.abs(fpr - target_fpr))
    return float(tpr[idx]), float(thr[idx])


def eval_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    roc = roc_auc_score(y_true, scores) if len(np.unique(y_true))>1 else 0.0
    pr  = average_precision_score(y_true, scores) if len(np.unique(y_true))>1 else 0.0
    # KS
    fpr, tpr, _ = roc_curve(y_true, scores)
    ks = float(np.max(np.abs(tpr - fpr))) if len(fpr) else 0.0
    r1, _ = recall_at_fpr(y_true, scores, 0.01)
    r2, _ = recall_at_fpr(y_true, scores, 0.02)
    return {"roc_auc": roc, "pr_auc": pr, "ks": ks, "recall_at_1pct": r1, "recall_at_2pct": r2}


def train_xgb(Xtr, ytr, Xva, yva, tune=False, n_trials=20):
    base = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0,
        random_state=42, eval_metric="auc", tree_method="hist"
    )
    if tune and HAS_OPTUNA:
        def objective(trial):
            params = base.copy()
            params.update(
                max_depth = trial.suggest_int("max_depth", 3, 10),
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                n_estimators = trial.suggest_int("n_estimators", 200, 1200),
                subsample = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0),
                reg_lambda = trial.suggest_float("reg_lambda", 0.5, 5.0),
            )
            mdl = xgb.XGBClassifier(**params)
            mdl.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False)
            p = mdl.predict_proba(Xva)[:,1]
            return roc_auc_score(yva, p)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        base.update(study.best_params)
    mdl = xgb.XGBClassifier(**base)
    mdl.fit(Xtr, ytr, eval_set=[(Xva,yva)], verbose=False)
    return mdl, base


def train_lgbm(Xtr, ytr, Xva, yva, tune=False, n_trials=20):
    if not HAS_LGBM:
        return None, {}
    base = dict(
        n_estimators=700, learning_rate=0.05, max_depth=-1,
        num_leaves=63, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.0, reg_lambda=1.0, random_state=42
    )
    if tune and HAS_OPTUNA:
        def objective(trial):
            params = base.copy()
            params.update(
                num_leaves = trial.suggest_int("num_leaves", 31, 255),
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                n_estimators = trial.suggest_int("n_estimators", 300, 1200),
                subsample = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0),
                reg_lambda = trial.suggest_float("reg_lambda", 0.5, 5.0),
            )
            mdl = lgb.LGBMClassifier(**params)
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            p = mdl.predict_proba(Xva)[:,1]
            return roc_auc_score(yva, p)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        base.update(study.best_params)
    mdl = lgb.LGBMClassifier(**base)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return mdl, base


def train_catboost(Xtr, ytr, Xva, yva, tune=False, n_trials=20):
    if not HAS_CAT:
        return None, {}
    base = dict(
        depth=6, learning_rate=0.05, iterations=800,
        subsample=0.9, l2_leaf_reg=3.0, random_state=42, verbose=False,
        eval_metric="AUC"
    )
    if tune and HAS_OPTUNA:
        def objective(trial):
            params = base.copy()
            params.update(
                depth = trial.suggest_int("depth", 4, 10),
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                iterations = trial.suggest_int("iterations", 300, 1500),
                subsample = trial.suggest_float("subsample", 0.6, 1.0),
                l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 6.0),
            )
            mdl = CatBoostClassifier(**params)
            mdl.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
            p = mdl.predict_proba(Xva)[:,1]
            return roc_auc_score(yva, p)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        base.update(study.best_params)
    mdl = CatBoostClassifier(**base)
    mdl.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
    return mdl, base


def main():
    args = parse_args()
    cli = ch_client(args)
    fqtn = f"{args.db}.{args.view}"
    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]

    # Определяем окна
    mx = max_date(cli, fqtn)
    train_start = mx - timedelta(days=args.train_days + args.valid_days)
    train_end   = mx - timedelta(days=args.valid_days)
    valid_start = mx - timedelta(days=args.valid_days)
    valid_end   = mx

    print(f"[INFO] train: {train_start.date()} .. {train_end.date()} | valid: {valid_start.date()} .. {valid_end.date()}")

    df_tr = fetch_window(cli, fqtn, train_start, train_end)
    df_va = fetch_window(cli, fqtn, valid_start, valid_end)

    Xtr, ytr, feats = build_features(df_tr)
    Xva, yva, _     = build_features(df_va)

    # MLflow
    if args.mlflow and HAS_MLFLOW:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)

    MODELS_DIR = Path(args.models_path); MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # XGB
    if "xgb" in models_to_run:
        with (mlflow.start_run(run_name="xgb") if args.mlflow and HAS_MLFLOW else nullcontext()):
            mdl, params = train_xgb(Xtr.values, ytr, Xva.values, yva, tune=args.tune, n_trials=args.n_trials)
            scores = mdl.predict_proba(Xva.values)[:,1]
            met = eval_metrics(yva, scores)
            results.append(("xgb", met))
            joblib.dump(mdl, MODELS_DIR / "fraud_xgb_benchmark.pkl")
            if args.mlflow and HAS_MLFLOW:
                mlflow.log_params({f"xgb_{k}":v for k,v in params.items()})
                mlflow.log_metrics({f"xgb_{k}":v for k,v in met.items()})

    # LGBM
    if "lgbm" in models_to_run and HAS_LGBM:
        with (mlflow.start_run(run_name="lgbm") if args.mlflow and HAS_MLFLOW else nullcontext()):
            mdl, params = train_lgbm(Xtr.values, ytr, Xva.values, yva, tune=args.tune, n_trials=args.n_trials)
            if mdl is not None:
                scores = mdl.predict_proba(Xva.values)[:,1]
                met = eval_metrics(yva, scores)
                results.append(("lgbm", met))
                joblib.dump(mdl, MODELS_DIR / "fraud_lgbm_benchmark.pkl")
                if args.mlflow and HAS_MLFLOW:
                    mlflow.log_params({f"lgbm_{k}":v for k,v in params.items()})
                    mlflow.log_metrics({f"lgbm_{k}":v for k,v in met.items()})
            else:
                print("[WARN] LightGBM не установлен — пропускаю")

    # CatBoost
    if "cat" in models_to_run and HAS_CAT:
        with (mlflow.start_run(run_name="catboost") if args.mlflow and HAS_MLFLOW else nullcontext()):
            mdl, params = train_catboost(Xtr.values, ytr, Xva.values, yva, tune=args.tune, n_trials=args.n_trials)
            if mdl is not None:
                scores = mdl.predict_proba(Xva.values)[:,1]
                met = eval_metrics(yva, scores)
                results.append(("catboost", met))
                joblib.dump(mdl, MODELS_DIR / "fraud_catboost_benchmark.pkl")
                if args.mlflow and HAS_MLFLOW:
                    mlflow.log_params({f"cat_{k}":v for k,v in params.items()})
                    mlflow.log_metrics({f"cat_{k}":v for k,v in met.items()})
            else:
                print("[WARN] CatBoost не установлен — пропускаю")

    print(json.dumps(dict(results), indent=2, ensure_ascii=False))


# utility to allow optional mlflow runs
from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield


if __name__ == "__main__":
    main()
