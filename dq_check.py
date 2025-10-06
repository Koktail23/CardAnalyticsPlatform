#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Check for ClickHouse tables using Pandera validation.
Запуск: python dq_check.py --table transactions_grade_a --limit 200000
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from clickhouse_driver import Client
# --- NEW: force UTF-8 console on Windows/PIPEs ---
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# --- /NEW ---

# ВАЖНО: используем pandas-бэкенд импорта pandera
try:
    import pandera.pandas as pa

    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False
    print("[WARN] Pandera not installed, schema validation will be skipped")

from dq_schema import subset_schema_for, business_checks


def parse_args():
    ap = argparse.ArgumentParser(description="Data Quality Check for ClickHouse table")
    ap.add_argument("--table", default="transactions_grade_a", help="Table name in DB")
    ap.add_argument("--db", default=os.getenv("CLICKHOUSE_DATABASE", "card_analytics"))
    ap.add_argument("--host", default=os.getenv("CLICKHOUSE_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("CLICKHOUSE_PORT", "9000")))
    ap.add_argument("--user", default=os.getenv("CLICKHOUSE_USER", "analyst"))
    ap.add_argument("--password", default=os.getenv("CLICKHOUSE_PASSWORD", "admin123"))
    ap.add_argument("--limit", type=int, default=200000, help="Row limit for sampling")
    ap.add_argument("--reports", default=os.getenv("REPORTS_PATH", "./reports"), help="Reports directory")
    return ap.parse_args()


def ch_client(args) -> Client:
    return Client(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.db,
        connect_timeout=3,
        send_receive_timeout=10,
        sync_request_timeout=10,
        settings={
            "max_execution_time": 30,
            "use_uncompressed_cache": 0,
            "strings_encoding": "utf-8"
        },
    )


def describe_table(client: Client, fqn: str) -> dict[str, str]:
    """Get table schema description"""
    rows = client.execute(f"DESCRIBE {fqn}")
    return {r[0]: r[1] for r in rows}


def is_numeric_type(ch_type: str) -> bool:
    """Check if ClickHouse type is numeric"""
    if ch_type.startswith("Nullable("):
        ch_type = ch_type[9:-1]
    return any(ch_type.startswith(p) for p in ("UInt", "Int", "Float", "Decimal"))


def select_clause_for(cols: dict[str, str]) -> str:
    """
    Build safe SELECT clause with type conversions.
    For numeric columns in numeric schema - use directly.
    For string columns or mixed schemas - use safe conversions.
    """
    s = []
    has = cols.__contains__

    # Check if it's a numeric schema
    numeric_schema = any(is_numeric_type(cols.get(c, "")) for c in ["amount_uzs", "mcc", "hour_num"])

    # String columns
    if has("hpan"):
        s.append("toString(hpan) AS hpan")
    if has("transaction_code"):
        s.append("toString(transaction_code) AS transaction_code")

    # Date/time columns
    if has("rday"):
        if is_numeric_type(cols["rday"]):
            s.append("rday AS rday")
        else:
            s.append("toUInt32OrNull(toString(rday)) AS rday")

    if has("transaction_date"):
        s.append("transaction_date AS transaction_date")

    # Amount column - most important to handle correctly
    if has("amount_uzs"):
        if is_numeric_type(cols["amount_uzs"]):
            s.append("amount_uzs AS amount_uzs")
        else:
            s.append("toFloat64OrNull(toString(amount_uzs)) AS amount_uzs")

    # MCC column
    if has("mcc"):
        if is_numeric_type(cols["mcc"]):
            s.append("mcc AS mcc")
        else:
            s.append("toUInt32OrNull(toString(mcc)) AS mcc")

    # Other string columns
    if has("merchant_name"):
        s.append("toString(merchant_name) AS merchant_name")
    if has("merchant_type"):
        s.append("toString(merchant_type) AS merchant_type")

    if has("merchant"):
        if is_numeric_type(cols["merchant"]):
            s.append("merchant AS merchant")
        else:
            s.append("toUInt32OrNull(toString(merchant)) AS merchant")

    # Flags
    if has("p2p_flag"):
        if is_numeric_type(cols["p2p_flag"]):
            s.append("p2p_flag AS p2p_flag")
        else:
            s.append("toUInt8OrNull(toString(p2p_flag)) AS p2p_flag")

    if has("respcode"):
        s.append("toString(respcode) AS respcode")

    if has("hour_num"):
        if is_numeric_type(cols["hour_num"]):
            s.append("hour_num AS hour_num")
        else:
            s.append("toUInt32OrNull(toString(hour_num)) AS hour_num")

    if has("emitent_bank"):
        s.append("toString(emitent_bank) AS emitent_bank")

    return ", ".join(s) if s else "*"


def nulls_uniques_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate null rates and unique values for each column"""
    total = len(df)
    rows = []
    for c in df.columns:
        n_null = int(df[c].isna().sum())
        n_uniq = int(df[c].nunique(dropna=True))
        rows.append((c, total, n_null, n_null / total if total else 0.0, n_uniq))

    return (
        pd.DataFrame(rows, columns=["column", "rows", "nulls", "null_rate", "uniques"])
        .sort_values("null_rate", ascending=False)
        .reset_index(drop=True)
    )


def main():
    args = parse_args()
    reports_dir = Path(args.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Determine full table name
    if "." in args.table:
        fqn = args.table
    else:
        fqn = f"{args.db}.{args.table}"

    print(f"[INFO] Checking data quality for: {fqn}")

    client = ch_client(args)

    # Get table schema
    try:
        cols = describe_table(client, fqn)
    except Exception as e:
        print(f"[ERROR] Failed to describe table {fqn}: {e}")
        return 1

    # Build SELECT query
    select_list = select_clause_for(cols)
    if select_list == "*":
        print(f"[WARN] Using SELECT * - no known columns found in {fqn}")

    sql = f"SELECT {select_list} FROM {fqn} LIMIT {args.limit}"
    print(f"[INFO] Fetching sample with query:")
    print(f"       {sql[:100]}..." if len(sql) > 100 else f"       {sql}")

    try:
        data = client.execute(sql)
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return 1

    # Parse column names from SELECT clause
    out_cols = []
    for piece in select_list.split(", "):
        if " AS " in piece:
            out_cols.append(piece.split(" AS ")[-1].strip())
        elif piece == "*":
            # Use all columns from DESCRIBE
            out_cols = list(cols.keys())
            break
        else:
            out_cols.append(piece.strip())

    # Create DataFrame
    df = pd.DataFrame(data, columns=out_cols[:len(data[0])] if data else [])
    print(f"[INFO] Loaded {len(df):,} rows with {len(df.columns)} columns")

    # Pandera validation (if available)
    errors = None
    if HAS_PANDERA:
        try:
            schema = subset_schema_for(df.columns.tolist())
            schema.validate(df, lazy=True)
            print("[OK] Pandera schema validation PASSED")
        except pa.errors.SchemaErrors as e:
            errors = e
            print(f"[FAIL] Pandera validation found {len(e.failure_cases)} issues")
        except Exception as e:
            print(f"[ERROR] Pandera validation error: {e}")
    else:
        print("[SKIP] Pandera validation skipped (not installed)")

    # Business checks
    print("[INFO] Running business checks...")
    b_issues = business_checks(df)
    if b_issues:
        print(f"[WARN] Found {len(b_issues)} business rule violations")
    else:
        print("[OK] No business rule violations found")

    # Calculate summary statistics
    summary = nulls_uniques_summary(df)

    # Save report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = reports_dir / f"dq_report_{args.table}_{ts}.md"
    csv_path = reports_dir / f"dq_summary_{args.table}_{ts}.csv"

    # Save CSV summary
    summary.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[INFO] Summary CSV saved: {csv_path}")

    # Generate Markdown report
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Data Quality Report — `{fqn}`\n\n")
        fh.write(f"- **Rows sampled:** {len(df):,} (LIMIT {args.limit:,})\n")
        fh.write(f"- **Generated:** {ts}\n")
        fh.write(f"- **Columns:** {', '.join(df.columns)}\n\n")

        # Pandera results
        if HAS_PANDERA:
            if errors is not None:
                fh.write("## ❌ Pandera Validation — FAILED\n\n")
                try:
                    fc = errors.failure_cases  # DataFrame with failures
                    fh.write(f"**Violations found:** {len(fc):,}\n\n")

                    # Group by check type
                    by_check = fc.groupby('check').size().sort_values(ascending=False)
                    fh.write("### Violations by type:\n\n")
                    for check, count in by_check.items():
                        fh.write(f"- {check}: {count:,} violations\n")
                    fh.write("\n")

                    # Show first 50 failures
                    fh.write("### First 50 violations:\n\n")
                    fh.write(fc.head(50).to_markdown(index=False))
                    fh.write("\n\n")
                except Exception as e:
                    fh.write(f"Could not format failure cases: {e}\n\n")
            else:
                fh.write("## ✅ Pandera Validation — PASSED\n\n")
                fh.write("All schema checks passed successfully.\n\n")
        else:
            fh.write("## ⚠️ Pandera Validation — SKIPPED\n\n")
            fh.write("Pandera library not installed.\n\n")

        # Business checks
        fh.write("## 🔎 Business Checks\n\n")
        if b_issues:
            for issue in b_issues:
                fh.write(f"- ⚠️ {issue}\n")
        else:
            fh.write("✅ No business rule violations detected.\n")
        fh.write("\n")

        # Summary statistics
        fh.write("## 📊 Column Statistics\n\n")
        fh.write("### Null rates and unique values:\n\n")
        fh.write(summary.to_markdown(index=False))
        fh.write("\n\n")

        # Data types
        fh.write("### Column data types:\n\n")
        for col in df.columns:
            if col in cols:
                fh.write(f"- **{col}**: {cols[col]}\n")
        fh.write("\n")

    print(f"[INFO] Markdown report saved: {md_path}")

    # Print summary (avoid emoji for Windows console compatibility)
    print("\n" + "=" * 50)
    print("DATA QUALITY CHECK SUMMARY")
    print("=" * 50)
    print(f"Table: {fqn}")
    print(f"Rows checked: {len(df):,}")
    if HAS_PANDERA:
        if errors is None:
            print(f"Schema validation: PASSED [OK]")
        else:
            print(f"Schema validation: FAILED ({len(errors.failure_cases)} issues)")
    print(f"Business violations: {len(b_issues)}")
    print(f"Columns with nulls: {(summary['null_rate'] > 0).sum()}/{len(summary)}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())