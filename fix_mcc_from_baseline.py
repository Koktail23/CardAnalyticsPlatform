#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Восстановление MCC в ClickHouse из baseline CSV.

Использование:
  python fix_mcc_from_baseline.py --csv "D:\\Projects\\CardAnalyticsPlatform\\data_100k.csv" --swap

Без --swap создаст fixed-таблицу и покажет сводку, но не тронет исходную.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
from datetime import datetime, date

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from clickhouse_driver import Client

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore MCC in transactions_grade_a from baseline CSV")
    p.add_argument("--csv", default=r"D:\Projects\CardAnalyticsPlatform\data_100k.csv",
                   help="Путь к baseline CSV (с заголовком). По умолчанию: D:\\Projects\\CardAnalyticsPlatform\\data_100k.csv")
    p.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE", "card_analytics"),
                   help="БД ClickHouse (по умолчанию из .env CLICKHOUSE_DATABASE)")
    p.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT", "9000")))
    p.add_argument("--user", default=os.getenv("CLICKHOUSE_USER", "analyst"))
    p.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD", "admin123"))
    p.add_argument("--table", default="transactions_grade_a",
                   help="Исходная таблица, в которую подставляем MCC (default: transactions_grade_a)")
    p.add_argument("--stage", default="stage_baseline",
                   help="Временная стадийная таблица (default: stage_baseline)")
    p.add_argument("--swap", action="store_true",
                   help="Поменять таблицы местами после сборки fixed (перенесёт fixed на место оригинальной)")
    p.add_argument("--batch", type=int, default=5000, help="Размер батча вставки в stage (default: 5000)")
    return p.parse_args()

# ---------- ClickHouse ----------
def ch_client(args: argparse.Namespace) -> Client:
    return Client(
        host=args.host, port=args.port, user=args.user, password=args.password, database=args.db,
        connect_timeout=3, send_receive_timeout=10, sync_request_timeout=10,
        settings={"max_execution_time": 30, "use_uncompressed_cache": 0}
    )

# ---------- SQL helpers ----------
def sql_create_stage(db: str, stage: str) -> str:
    return f"""
    DROP TABLE IF EXISTS {db}.{stage};

    CREATE TABLE {db}.{stage}
    (
      hpan              String,
      transaction_code  String,
      rday              UInt32,
      transaction_date  Date,
      amount_uzs        Float64,
      reqamt            Float64,
      conamt            Float64,
      mcc               String,       -- строкой (нормализуем позже)
      merchant_name     String,
      merchant_type     String,
      merchant          UInt32,
      p2p_flag          UInt8,
      p2p_type          String
    )
    ENGINE = MergeTree
    ORDER BY (transaction_date, hpan);
    """

def sql_create_map_by_merchant(db: str) -> str:
    # ВАЖНО: агрегат называется mcc_code, исходное поле - mcc_raw в подзапросе
    return f"""
    DROP TABLE IF EXISTS {db}.mcc_map_merchant;

    CREATE TABLE {db}.mcc_map_merchant
    ENGINE = MergeTree
    ORDER BY merchant AS
    SELECT
      merchant,
      anyHeavy(toUInt16OrNull(extract(toString(mcc_raw), '\\\\d+'))) AS mcc_code
    FROM (
      SELECT toUInt32(merchant) AS merchant, mcc AS mcc_raw
      FROM {db}.stage_baseline
    ) sb
    WHERE toUInt16OrNull(extract(toString(mcc_raw), '\\\\d+')) > 0
    GROUP BY merchant;
    """

def sql_create_map_by_merchant_name(db: str) -> str:
    # То же самое по merchant_name
    return f"""
    DROP TABLE IF EXISTS {db}.mcc_map_merchant_name;

    CREATE TABLE {db}.mcc_map_merchant_name
    ENGINE = MergeTree
    ORDER BY merchant_name AS
    SELECT
      merchant_name,
      anyHeavy(toUInt16OrNull(extract(toString(mcc_raw), '\\\\d+'))) AS mcc_code
    FROM (
      SELECT toString(merchant_name) AS merchant_name, mcc AS mcc_raw
      FROM {db}.stage_baseline
    ) sb
    WHERE toUInt16OrNull(extract(toString(mcc_raw), '\\\\d+')) > 0
    GROUP BY merchant_name;
    """

def sql_build_fixed_by_merchant(db: str, src: str) -> str:
    return f"""
    DROP TABLE IF EXISTS {db}.{src}_fixed;

    CREATE TABLE {db}.{src}_fixed
    ENGINE = MergeTree
    ORDER BY (transaction_date, hpan) AS
    SELECT 
      t.hpan,
      t.transaction_code,
      t.rday,
      t.transaction_date,
      t.amount_uzs,
      t.reqamt,
      t.conamt,
      toUInt16(coalesce(m.mcc_code, t.mcc)) AS mcc,
      t.merchant_name,
      t.merchant_type,
      t.merchant,
      t.p2p_flag,
      t.p2p_type
    FROM {db}.{src} AS t
    LEFT JOIN {db}.mcc_map_merchant AS m USING (merchant);
    """

def sql_build_fixed_by_merchant_name(db: str, src: str) -> str:
    return f"""
    DROP TABLE IF EXISTS {db}.{src}_fixed;

    CREATE TABLE {db}.{src}_fixed
    ENGINE = MergeTree
    ORDER BY (transaction_date, hpan) AS
    SELECT 
      t.hpan,
      t.transaction_code,
      t.rday,
      t.transaction_date,
      t.amount_uzs,
      t.reqamt,
      t.conamt,
      toUInt16(coalesce(m.mcc_code, t.mcc)) AS mcc,
      t.merchant_name,
      t.merchant_type,
      t.merchant,
      t.p2p_flag,
      t.p2p_type
    FROM {db}.{src} AS t
    LEFT JOIN {db}.mcc_map_merchant_name AS m USING (merchant_name);
    """

def sql_summary(table_fqn: str) -> str:
    return f"""
    SELECT 
      count() AS rows_total,
      countIf(mcc = 0) AS rows_mcc_zero,
      countIf(mcc > 0) AS rows_mcc_gt0,
      uniq(mcc) AS uniq_mcc,
      min(mcc) AS mcc_min,
      max(mcc) AS mcc_max
    FROM {table_fqn};
    """

# ---------- CSV ingest ----------
def pick(d: dict, *names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default

def coerce_int(v: Optional[str], default: int = 0) -> int:
    if v is None: return default
    s = str(v).strip()
    if s == "": return default
    try:
        return int(float(s))
    except Exception:
        return default

def coerce_float(v: Optional[str], default: float = 0.0) -> float:
    if v is None: return default
    s = str(v).strip().replace(",", ".")
    if s == "": return default
    try:
        return float(s)
    except Exception:
        return default

def coerce_date(v: Optional[str]) -> date:
    if v is None:
        return date(1970, 1, 1)
    s = str(v).strip()
    if s == "":
        return date(1970, 1, 1)
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except Exception:
            pass
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        return date(1970, 1, 1)

def read_csv_rows(csv_path: Path, batch: int) -> Iterable[List[Tuple]]:
    encodings = ["utf-8-sig", "utf-8", "cp1251", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as fh:
                sample = fh.read(4096)
                fh.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    delimiter = dialect.delimiter
                except Exception:
                    delimiter = ","
                reader = csv.DictReader(fh, delimiter=delimiter)
                buf: List[Tuple] = []
                for row in reader:
                    rec = (
                        str(pick(row, "hpan", "card", "pan", default="")),
                        str(pick(row, "transaction_code", "transactionId", "txn_code", default="")),
                        coerce_int(pick(row, "rday", "relative_day", "day_index")),
                        coerce_date(pick(row, "transaction_date", "date", "txn_date")),
                        coerce_float(pick(row, "amount_uzs", "amount", "sum", "total")),
                        coerce_float(pick(row, "reqamt", "requested_amount")),
                        coerce_float(pick(row, "conamt", "confirmed_amount")),
                        str(pick(row, "mcc", "mcc_code", "merchant_category_code", default="")),
                        str(pick(row, "merchant_name", "merchantName", default="")),
                        str(pick(row, "merchant_type", "merchantType", default="")),
                        coerce_int(pick(row, "merchant", "merchant_id")),
                        1 if str(pick(row, "p2p_flag", "is_p2p", "p2p", default="0")).strip().lower() in ("1","true","yes") else 0,
                        str(pick(row, "p2p_type", "p2pKind", default="")),
                    )
                    buf.append(rec)
                    if len(buf) >= batch:
                        yield buf
                        buf = []
                if buf:
                    yield buf
            return
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err

# ---------- main ----------
def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}")
        sys.exit(2)

    ch = ch_client(args)
    db = args.db
    src = args.table
    stage = args.stage

    print(f"[INFO] DB: {db}, SRC: {src}, CSV: {csv_path}")

    # 0) Покажем состояние исходной таблицы
    try:
        print("[INFO] Source table summary BEFORE:")
        print(ch.execute(sql_summary(f"{db}.{src}")))
    except Exception as e:
        print("[WARN] Can't summarize source table:", e)

    # 1) Создаём stage-таблицу
    print("[STEP] Create stage table …")
    for stmt in sql_create_stage(db, stage).split(";"):
        s = stmt.strip()
        if s:
            ch.execute(s)

    # 2) Вставляем CSV в stage батчами
    print("[STEP] Insert CSV into stage … (batch =", args.batch, ")")
    total = 0
    for batch in read_csv_rows(csv_path, args.batch):
        ch.execute(f"INSERT INTO {db}.{stage} VALUES", batch)
        total += len(batch)
        if total % (args.batch * 5) == 0:
            print(f"  inserted {total} rows …")
    print(f"[OK] Inserted {total} rows into {db}.{stage}")

    # 3) Создаём маппинги
    print("[STEP] Build mapping by merchant …")
    for stmt in sql_create_map_by_merchant(db).split(";"):
        s = stmt.strip()
        if s:
            ch.execute(s)

    rows_map_merchant = ch.execute(f"SELECT countIf(mcc_code > 0) FROM {db}.mcc_map_merchant")[0][0]
    print(f"[INFO] mcc_map_merchant: {rows_map_merchant} rows with mcc>0")

    if rows_map_merchant == 0:
        print("[STEP] merchant mapping empty → try mapping by merchant_name …")
        for stmt in sql_create_map_by_merchant_name(db).split(";"):
            s = stmt.strip()
            if s:
                ch.execute(s)
        rows_map_name = ch.execute(f"SELECT countIf(mcc_code > 0) FROM {db}.mcc_map_merchant_name")[0][0]
        print(f"[INFO] mcc_map_merchant_name: {rows_map_name} rows with mcc>0")

        if rows_map_name == 0:
            print("[ERR] Both mappings are empty. Check baseline CSV (column mcc should contain numeric codes).")
            sys.exit(3)

        # 4) Сборка fixed по merchant_name
        print("[STEP] Build fixed table (join by merchant_name) …")
        for stmt in sql_build_fixed_by_merchant_name(db, src).split(";"):
            s = stmt.strip()
            if s:
                ch.execute(s)
    else:
        # 4) Сборка fixed по merchant
        print("[STEP] Build fixed table (join by merchant) …")
        for stmt in sql_build_fixed_by_merchant(db, src).split(";"):
            s = stmt.strip()
            if s:
                ch.execute(s)

    # 5) Сводка по fixed
    print("[INFO] Fixed table summary:")
    print(ch.execute(sql_summary(f"{db}.{src}_fixed")))

    # 6) Переключение таблиц по желанию
    if args.swap:
        print("[STEP] Swap tables …")
        ch.execute(f"RENAME TABLE {db}.{src} TO {db}.{src}_zero, {db}.{src}_fixed TO {db}.{src}")
        print("[OK] Swapped.")
        print("[INFO] Source table summary AFTER:")
        print(ch.execute(sql_summary(f"{db}.{src}")))
    else:
        print("[NOTE] Swap is disabled. To apply replacement, run with --swap")

    print("[DONE] MCC restore pipeline finished.")

if __name__ == "__main__":
    main()
