#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hier_forecasting.py — мультисерийный иерархический прогноз объёма транзакций
Уровни: merchant_key -> MCC -> TOTAL. Reconciliation: bottom-up.

Алгоритм:
1) Выгружаем из ClickHouse ежедневные суммы по (MCC, merchant_key).
2) Для каждого MCC выбираем top merchants так, чтобы покрыть >= coverage (по умолчанию 0.9),
   и ограничиваем максимумом (по умолчанию 50). Остальное идёт в "OTHER".
3) Прогнозируем КАЖДОГО мерчанта (Prophet, если доступен; иначе наивный MA-7).
4) Bottom-up: MCC-forecast = сумма merchant-forecast (включая OTHER). TOTAL = сумма MCC.
5) Сохраняем CSV:
   - ml/forecast_hier_merchant.csv (level='merchant')
   - ml/forecast_hier_mcc.csv      (level='mcc')
   - ml/forecast_hier_total.csv    (level='total')
   - ml/forecast_hier_summary.json — параметры пробега/покрытия/кол-во серий

Запуск:
  python ml\\hier_forecasting.py --horizon 30 --coverage 0.9 --max_merchants 50 --max_mcc 20
Параметры подключения берутся из .env (CLICKHOUSE_*), источник — fraud_dataset_v1 (если есть), иначе TRAIN_TABLE.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from clickhouse_driver import Client

# Prophet опционален; если нет — используем наивный MA-7
_HAS_PROPHET = True
try:
    from prophet import Prophet  # type: ignore
except Exception:
    _HAS_PROPHET = False


def parse_args():
    ap = argparse.ArgumentParser("hier_forecasting")
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE","card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST","localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT","9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER","analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD","admin123"))

    ap.add_argument("--view", default=os.getenv("FRAUD_VIEW","fraud_dataset_v1"),
                    help="Источник: fraud_dataset_v1 (если существует), иначе TRAIN_TABLE")
    ap.add_argument("--table", default=os.getenv("TRAIN_TABLE","transactions_optimized"),
                    help="Фолбэк, если нет вью")

    ap.add_argument("--horizon", type=int, default=30, help="горизонт прогноза (дней)")
    ap.add_argument("--coverage", type=float, default=0.9, help="доля покрытия топ-мерчантов в MCC")
    ap.add_argument("--max_merchants", type=int, default=50, help="макс. число мерчантов в одном MCC")
    ap.add_argument("--max_mcc", type=int, default=20, help="макс. число MCC по объёму")
    ap.add_argument("--models_path", default=os.getenv("MODELS_PATH","ml"))
    return ap.parse_args()


def ch_client(args) -> Client:
    return Client(
        host=args.host, port=args.port, user=args.user, password=args.password, database=args.db,
        connect_timeout=3, send_receive_timeout=15, sync_request_timeout=15,
        settings={"max_execution_time": 180}
    )


def source_fqtn(cli: Client, args) -> str:
    # Проверим наличие fraud_dataset_v1
    try:
        rows = cli.execute("SHOW TABLES")
        names = set([r[0] for r in rows])
        if args.view in names:
            return f"{args.db}.{args.view}"
    except Exception:
        pass
    return f"{args.db}.{args.table}"


def fetch_daily(cli: Client, fqtn: str) -> pd.DataFrame:
    # merchant_key: если есть merchant id — берём его; иначе lower(merchant_name)
    cols = {r[0]: r[1] for r in cli.execute(f"DESCRIBE {fqtn}")}
    has_merch_id = "merchant" in cols
    merchant_expr = "toString(merchant)" if has_merch_id else "lower(toString(merchant_name))"

    sql = f"""
    SELECT
      toDate(transaction_date) AS ds,
      coalesce(toUInt32OrNull(extract(toString(mcc), '\\\\d+')), 0) AS mcc,
      {merchant_expr} AS merchant_key,
      sum(toFloat64OrNull(toString(amount_uzs))) AS y
    FROM {fqtn}
    WHERE transaction_date IS NOT NULL
    GROUP BY ds, mcc, merchant_key
    ORDER BY ds
    """
    rows = cli.execute(sql)
    df = pd.DataFrame(rows, columns=["ds","mcc","merchant_key","y"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def top_mcc_and_merchants(df: pd.DataFrame, coverage: float, max_mcc: int, max_merch: int) -> Tuple[List[int], Dict[int, List[str]]]:
    # Топ MCC по сумме за всё время
    mcc_tot = df.groupby("mcc")["y"].sum().sort_values(ascending=False)
    top_mcc = mcc_tot.head(max_mcc).index.tolist()

    # Для каждого MCC — список топ-мерчантов по покрытию
    select_merchants: Dict[int, List[str]] = {}
    for m in top_mcc:
        dfm = df[df["mcc"]==m]
        merch_tot = dfm.groupby("merchant_key")["y"].sum().sort_values(ascending=False)
        cumsum = merch_tot.cumsum() / (merch_tot.sum() or 1)
        keep = merch_tot.index[(cumsum <= coverage)].tolist()
        if len(keep) < 1 and len(merch_tot) > 0:
            keep = [merch_tot.index[0]]
        keep = keep[:max_merch]
        select_merchants[m] = keep
    return top_mcc, select_merchants


def prophet_or_naive(df_xy: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """df_xy: столбцы ds, y"""
    df_xy = df_xy.sort_values("ds")
    if len(df_xy) == 0:
        future = pd.date_range(pd.Timestamp.today().normalize(), periods=horizon)
        return pd.DataFrame({"ds": future, "yhat": 0.0})

    if _HAS_PROPHET and len(df_xy) >= 14:
        try:
            m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
            m.fit(df_xy.rename(columns={"ds":"ds","y":"y"}))
            future = m.make_future_dataframe(periods=horizon, freq="D")
            fc = m.predict(future)[["ds","yhat"]]
            return fc.tail(horizon)
        except Exception:
            pass
    # Наивный MA-7
    y = df_xy["y"].astype(float)
    ma = y.rolling(7, min_periods=1).mean().iloc[-1]
    future = pd.date_range(df_xy["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"ds": future, "yhat": ma})


def main():
    args = parse_args()
    MODELS = Path(args.models_path); MODELS.mkdir(exist_ok=True, parents=True)
    cli = ch_client(args)
    fqtn = source_fqtn(cli, args)

    print(f"[INFO] Source: {fqtn}")
    df = fetch_daily(cli, fqtn)
    if df.empty:
        print("[ERR] Пустые данные"); return

    # Выбор верхних MCC и мерчантов
    top_mcc, keep_merchants = top_mcc_and_merchants(df, args.coverage, args.max_mcc, args.max_merchants)
    print(f"[INFO] Top MCC count: {len(top_mcc)}")

    merch_fc_frames = []
    for m in top_mcc:
        keep = set(keep_merchants[m])
        df_mcc = df[df["mcc"]==m]
        # total по mcc (для OTHER)
        df_mcc_tot = df_mcc.groupby("ds", as_index=False)["y"].sum().rename(columns={"y":"y_mcc"})
        # сумма выбранных мерчантов
        df_sel = df_mcc[df_mcc["merchant_key"].isin(keep)]
        # Прогноз каждого выбранного мерчанта
        for merch, grp in df_sel.groupby("merchant_key"):
            fc = prophet_or_naive(grp[["ds","y"]], args.horizon)
            fc["mcc"] = m
            fc["merchant_key"] = merch
            merch_fc_frames.append(fc[["ds","mcc","merchant_key","yhat"]])

        # OTHER = mcc_total - sum(selected merchants)
        df_sel_sum = df_sel.groupby("ds", as_index=False)["y"].sum().rename(columns={"y":"y_sel"})
        df_other_hist = df_mcc_tot.merge(df_sel_sum, on="ds", how="left").fillna({"y_sel":0.0})
        df_other_hist["y_other"] = (df_other_hist["y_mcc"] - df_other_hist["y_sel"]).clip(lower=0.0)
        fc_other = prophet_or_naive(df_other_hist[["ds","y_other"]].rename(columns={"y_other":"y"}), args.horizon)
        fc_other["mcc"] = m
        fc_other["merchant_key"] = "__OTHER__"
        merch_fc_frames.append(fc_other[["ds","mcc","merchant_key","yhat"]])

    if merch_fc_frames:
        merchant_fc = pd.concat(merch_fc_frames, ignore_index=True)
    else:
        merchant_fc = pd.DataFrame(columns=["ds","mcc","merchant_key","yhat"])
    merchant_fc["level"] = "merchant"

    # Bottom-up: MCC forecast = сумма мерчантов
    mcc_fc = (merchant_fc
              .groupby(["ds","mcc"], as_index=False)["yhat"].sum())
    mcc_fc["level"] = "mcc"

    # Bottom-up: TOTAL = сумма MCC
    total_fc = (mcc_fc.groupby("ds", as_index=False)["yhat"].sum())
    total_fc["level"] = "total"

    # Сохраняем
    merch_path = MODELS/"forecast_hier_merchant.csv"
    mcc_path   = MODELS/"forecast_hier_mcc.csv"
    total_path = MODELS/"forecast_hier_total.csv"

    merchant_fc[["level","mcc","merchant_key","ds","yhat"]].to_csv(merch_path, index=False)
    mcc_fc[["level","mcc","ds","yhat"]].to_csv(mcc_path, index=False)
    total_fc[["level","ds","yhat"]].to_csv(total_path, index=False)

    # summary
    summary = {
        "source": fqtn,
        "horizon": args.horizon,
        "coverage": args.coverage,
        "max_merchants": args.max_merchants,
        "max_mcc": args.max_mcc,
        "mcc_count": int(len(top_mcc)),
        "merchant_series": int(merchant_fc["merchant_key"].nunique()) if not merchant_fc.empty else 0,
        "files": {
            "merchant": str(merch_path),
            "mcc": str(mcc_path),
            "total": str(total_path)
        }
    }
    (MODELS/"forecast_hier_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[OK] Hierarchical forecast ready")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
