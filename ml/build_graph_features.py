#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_graph_features.py — генерация граф-признаков за окно (по умолчанию 90 дней)
из представления card_analytics.fraud_dataset_v1.

Создаёт/обновляет:
  - card_analytics.graph_merchant_features
      (merchant_key, window_start, window_end, degree_cards_90d, txn_count_90d, fraud_ratio_90d, updated_at)
  - card_analytics.graph_card_features
      (hpan, window_start, window_end, degree_merchants_90d, txn_count_90d, avg_amount_90d,
       max_amount_90d, neighbor_fraud_merchants_90d, neighbor_fraud_ratio_90d, updated_at)

Запуск:
  python ml\\build_graph_features.py --window_days 90
"""

import os
import argparse
from datetime import timedelta
from clickhouse_driver import Client


def parse_args():
    ap = argparse.ArgumentParser("build_graph_features")
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE", "card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT", "9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER", "analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD", "admin123"))
    ap.add_argument("--view", default=os.getenv("FRAUD_VIEW", "fraud_dataset_v1"))
    ap.add_argument("--window_days", type=int, default=90)
    return ap.parse_args()


def ch(args) -> Client:
    return Client(
        host=args.host, port=args.port, user=args.user, password=args.password, database=args.db,
        connect_timeout=3, send_receive_timeout=15, sync_request_timeout=15,
        settings={"max_execution_time": 120},
    )


def max_date(cli: Client, fqtn: str):
    d = cli.execute(f"SELECT max(toDate(transaction_date)) FROM {fqtn}")[0][0]
    return d  # Date


def has_column(cli: Client, fqtn: str, name: str) -> bool:
    cols = [r[0] for r in cli.execute(f"DESCRIBE {fqtn}")]
    return name in cols


def main():
    args = parse_args()
    cli = ch(args)
    fqtn = f"{args.db}.{args.view}"

    # merchant_key: merchant (если есть), иначе lower(merchant_name)
    has_merch_id = has_column(cli, fqtn, "merchant")
    merchant_key_expr = "toString(merchant)" if has_merch_id else "lower(toString(merchant_name))"

    # окно
    end_date = max_date(cli, fqtn)
    if end_date is None:
        print("[ERR] Нет данных во вью"); return
    start_date = end_date - timedelta(days=args.window_days - 1)
    print(f"[INFO] window: {start_date} .. {end_date}")

    # таблицы-назначения
    cli.execute(f"""
        CREATE TABLE IF NOT EXISTS {args.db}.graph_merchant_features
        (
          merchant_key String,
          window_start Date,
          window_end   Date,
          degree_cards_90d UInt32,
          txn_count_90d    UInt32,
          fraud_ratio_90d  Float64,
          updated_at DateTime DEFAULT now()
        )
        ENGINE=MergeTree
        ORDER BY (merchant_key, window_end)
    """)
    cli.execute(f"""
        CREATE TABLE IF NOT EXISTS {args.db}.graph_card_features
        (
          hpan String,
          window_start Date,
          window_end   Date,
          degree_merchants_90d UInt32,
          txn_count_90d       UInt32,
          avg_amount_90d      Float64,
          max_amount_90d      Float64,
          neighbor_fraud_merchants_90d UInt32,
          neighbor_fraud_ratio_90d     Float64,
          updated_at DateTime DEFAULT now()
        )
        ENGINE=MergeTree
        ORDER BY (hpan, window_end)
    """)

    # подчистим заранее только для текущего окна
    cli.execute(f"ALTER TABLE {args.db}.graph_merchant_features DELETE WHERE window_end = toDate('{end_date:%Y-%m-%d}')")
    cli.execute(f"ALTER TABLE {args.db}.graph_card_features    DELETE WHERE window_end = toDate('{end_date:%Y-%m-%d}')")

    # ===== MERCHANT FEATURES =====
    merchant_sql = f"""
        INSERT INTO {args.db}.graph_merchant_features
            (merchant_key, window_start, window_end, degree_cards_90d, txn_count_90d, fraud_ratio_90d)
        SELECT
          {merchant_key_expr} AS merchant_key,
          toDate('{start_date:%Y-%m-%d}') AS window_start,
          toDate('{end_date:%Y-%m-%d}')   AS window_end,
          uniqExact(hpan) AS degree_cards_90d,
          count()         AS txn_count_90d,
          avg(toFloat64(fraud_proxy)) AS fraud_ratio_90d
        FROM {fqtn}
        WHERE toDate(transaction_date) BETWEEN toDate('{start_date:%Y-%m-%d}') AND toDate('{end_date:%Y-%m-%d}')
        GROUP BY merchant_key
    """
    cli.execute(merchant_sql)
    print("[OK] graph_merchant_features inserted")

    # ===== CARD FEATURES =====
    card_sql = f"""
        INSERT INTO {args.db}.graph_card_features
            (hpan, window_start, window_end, degree_merchants_90d, txn_count_90d,
             avg_amount_90d, max_amount_90d, neighbor_fraud_merchants_90d, neighbor_fraud_ratio_90d)
        WITH merchants AS (
          SELECT merchant_key, fraud_ratio_90d
          FROM {args.db}.graph_merchant_features
          WHERE window_end = toDate('{end_date:%Y-%m-%d}')
        )
        SELECT
          toString(hpan) AS hpan,
          toDate('{start_date:%Y-%m-%d}') AS window_start,
          toDate('{end_date:%Y-%m-%d}')   AS window_end,
          uniqExact({merchant_key_expr}) AS degree_merchants_90d,
          count() AS txn_count_90d,
          avg(toFloat64OrNull(toString(amount_uzs))) AS avg_amount_90d,
          max(toFloat64OrNull(toString(amount_uzs))) AS max_amount_90d,
          sum( (ifNull(m.fraud_ratio_90d, 0) > 0) ) AS neighbor_fraud_merchants_90d,
          if(uniqExact({merchant_key_expr})=0, 0,
             sum( (ifNull(m.fraud_ratio_90d, 0) > 0) ) / uniqExact({merchant_key_expr}) ) AS neighbor_fraud_ratio_90d
        FROM {fqtn}
        LEFT JOIN merchants m ON {merchant_key_expr} = m.merchant_key
        WHERE toDate(transaction_date) BETWEEN toDate('{start_date:%Y-%m-%d}') AND toDate('{end_date:%Y-%m-%d}')
        GROUP BY hpan
    """
    cli.execute(card_sql)
    print("[OK] graph_card_features inserted")

    print("[DONE] graph features window ready")


if __name__ == "__main__":
    main()
