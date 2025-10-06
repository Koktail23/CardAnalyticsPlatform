#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly Report generator — отчёт за период [start..end] или последние N дней от max(transaction_date).
Без небезопасных приведения типов: используем amount_uzs как числовое поле.
"""

import os
import argparse
from datetime import date, timedelta, datetime
from pathlib import Path

import pandas as pd
from clickhouse_driver import Client

CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", "9000"))
CH_USER = os.getenv("CLICKHOUSE_USER", "analyst")
CH_PASS = os.getenv("CLICKHOUSE_PASSWORD", "admin123")
CH_DB = os.getenv("CLICKHOUSE_DATABASE", "card_analytics")
TABLE = os.getenv("REPORT_TABLE", os.getenv("TRAIN_TABLE", "transactions_grade_a"))


def cli():
    return Client(host=CH_HOST, port=CH_PORT, user=CH_USER, password=CH_PASS,
                  database=CH_DB, settings={"strings_encoding": "utf-8"})


def ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)


def has_column(client: Client, table: str, column: str) -> bool:
    """Check if column exists in table"""
    try:
        rows = client.execute(f"DESCRIBE {table}")
        cols = [r[0] for r in rows]
        return column in cols
    except:
        return False


def main():
    p = argparse.ArgumentParser(description="Generate weekly report from ClickHouse data")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--days", type=int, default=7, help="Number of days for report period")
    p.add_argument("--table", type=str, default=None, help="Table name to use for report")
    args = p.parse_args()

    c = cli()

    # Priority: 1) CLI argument, 2) Environment variable, 3) Default
    if args.table:
        table = args.table
    else:
        # Use transactions_grade_a as preferred table
        tables = [r[0] for r in c.execute("SHOW TABLES")]
        if "transactions_grade_a" in tables:
            table = "transactions_grade_a"
        elif "transactions_optimized" in tables:
            table = "transactions_optimized"
        else:
            table = TABLE

    print(f"[INFO] Using table: {table}")

    # Determine date range
    if not has_column(c, table, "transaction_date"):
        ensure_reports_dir()
        out_md = f"reports/weekly_report_{date.today():%Y%m%d}.md"
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(f"# Weekly Report — таблица {table} не содержит transaction_date\n\n")
            f.write("Для генерации отчёта требуется колонка transaction_date.\n")
        print(f"[ERROR] Table {table} doesn't have transaction_date column")
        return

    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        # Get max date from table
        end = c.execute(f"SELECT max(transaction_date) FROM {table}")[0][0]
        if end is None:  # Empty table
            ensure_reports_dir()
            out_md = f"reports/weekly_report_{date.today():%Y%m%d}.md"
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(f"# Weekly Report — таблица {table} пуста\n\n")
                f.write("Нет данных для построения отчёта.\n")
            print(f"[OK] Empty report -> {out_md}")
            return

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start = end - timedelta(days=args.days - 1)

    params = {"s": start, "e": end}
    prev_start = start - timedelta(days=(end - start).days + 1)
    prev_end = start - timedelta(days=1)
    prev_params = {"s": prev_start, "e": prev_end}

    ensure_reports_dir()

    print(f"[INFO] Report period: {start} to {end}")
    print(f"[INFO] Previous period: {prev_start} to {prev_end}")

    # KPI for current week (without toString on amount_uzs)
    has_p2p = has_column(c, table, "p2p_flag")

    if has_p2p:
        kpi_query = f"""
        SELECT 
            sum(amount_uzs) AS volume,
            count() AS tx,
            round(100.0 * sumIf(1, p2p_flag = 1) / count(), 2) AS p2p_share
        FROM {table}
        WHERE transaction_date BETWEEN %(s)s AND %(e)s
        """
    else:
        kpi_query = f"""
        SELECT 
            sum(amount_uzs) AS volume,
            count() AS tx,
            0 AS p2p_share
        FROM {table}
        WHERE transaction_date BETWEEN %(s)s AND %(e)s
        """

    try:
        vol, tx, p2p = c.execute(kpi_query, params)[0]
        vol = vol or 0
        tx = tx or 0
        p2p = p2p or 0
    except Exception as e:
        print(f"[WARN] Error getting current KPI: {e}")
        vol, tx, p2p = 0, 0, 0

    # KPI for previous week
    try:
        vol_p, tx_p, p2p_p = c.execute(kpi_query, prev_params)[0]
        vol_p = vol_p or 0
        tx_p = tx_p or 0
        p2p_p = p2p_p or 0
    except Exception as e:
        print(f"[WARN] Error getting previous KPI: {e}")
        vol_p, tx_p, p2p_p = 0, 0, 0

    # Top MCC by volume
    try:
        if has_column(c, table, "mcc"):
            top_mcc_query = f"""
            SELECT 
                mcc,
                sum(amount_uzs) AS volume
            FROM {table}
            WHERE transaction_date BETWEEN %(s)s AND %(e)s
              AND mcc > 0
            GROUP BY mcc
            ORDER BY volume DESC
            LIMIT 20
            """
            top_mcc = pd.DataFrame(
                c.execute(top_mcc_query, params),
                columns=["mcc", "volume"]
            )
        else:
            top_mcc = pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Error getting top MCC: {e}")
        top_mcc = pd.DataFrame()

    # Top merchants by volume
    try:
        if has_column(c, table, "merchant_name"):
            top_merch_query = f"""
            SELECT 
                merchant_name,
                sum(amount_uzs) AS volume
            FROM {table}
            WHERE transaction_date BETWEEN %(s)s AND %(e)s
              AND merchant_name != ''
            GROUP BY merchant_name
            ORDER BY volume DESC
            LIMIT 20
            """
            top_merch = pd.DataFrame(
                c.execute(top_merch_query, params),
                columns=["merchant_name", "volume"]
            )
        else:
            top_merch = pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Error getting top merchants: {e}")
        top_merch = pd.DataFrame()

    # Hourly seasonality
    try:
        if has_column(c, table, "transaction_date"):
            # Check if we have hour_num column or need to extract from date
            if has_column(c, table, "hour_num"):
                hourly_query = f"""
                SELECT 
                    hour_num AS hour,
                    sum(amount_uzs) AS volume
                FROM {table}
                WHERE transaction_date BETWEEN %(s)s AND %(e)s
                  AND hour_num BETWEEN 0 AND 23
                GROUP BY hour
                ORDER BY hour
                """
            else:
                hourly_query = f"""
                SELECT 
                    toHour(toDateTime(transaction_date)) AS hour,
                    sum(amount_uzs) AS volume
                FROM {table}
                WHERE transaction_date BETWEEN %(s)s AND %(e)s
                GROUP BY hour
                ORDER BY hour
                """
            hourly = pd.DataFrame(
                c.execute(hourly_query, params),
                columns=["hour", "volume"]
            )
        else:
            hourly = pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Error getting hourly data: {e}")
        hourly = pd.DataFrame()

    # Data quality metrics for the week
    try:
        # Check which columns exist
        has_mcc = has_column(c, table, "mcc")
        has_merchant = has_column(c, table, "merchant_name")

        dq_parts = [
            "countIf(amount_uzs = 0 OR amount_uzs IS NULL) AS empty_amt",
            f"countIf(mcc = 0 OR mcc IS NULL) AS empty_mcc" if has_mcc else "0 AS empty_mcc",
            f"countIf(merchant_name = '' OR merchant_name IS NULL) AS empty_merch" if has_merchant else "0 AS empty_merch",
            "count() AS total_w"
        ]

        dq_query = f"""
        SELECT {', '.join(dq_parts)}
        FROM {table}
        WHERE transaction_date BETWEEN %(s)s AND %(e)s
        """
        empty_amt, empty_mcc, empty_merch, total_w = c.execute(dq_query, params)[0]

        fill_amt = 100.0 * (1 - empty_amt / max(total_w, 1))
        fill_mcc = 100.0 * (1 - empty_mcc / max(total_w, 1)) if has_mcc else 100.0
        fill_mer = 100.0 * (1 - empty_merch / max(total_w, 1)) if has_merchant else 100.0
    except Exception as e:
        print(f"[WARN] Error getting DQ metrics: {e}")
        fill_amt, fill_mcc, fill_mer = 100.0, 100.0, 100.0

    # Mini PSI calculation for amount_uzs
    psi_status = "Не рассчитан"
    try:
        # Check if we have enough data
        if vol_p > 0 and vol > 0:
            # Get distribution for baseline (previous week)
            psi_baseline_query = f"""
            SELECT 
                quantile(0.1)(amount_uzs) AS q10,
                quantile(0.2)(amount_uzs) AS q20,
                quantile(0.3)(amount_uzs) AS q30,
                quantile(0.4)(amount_uzs) AS q40,
                quantile(0.5)(amount_uzs) AS q50,
                quantile(0.6)(amount_uzs) AS q60,
                quantile(0.7)(amount_uzs) AS q70,
                quantile(0.8)(amount_uzs) AS q80,
                quantile(0.9)(amount_uzs) AS q90
            FROM {table}
            WHERE transaction_date BETWEEN %(s)s AND %(e)s
              AND amount_uzs > 0
            """

            baseline_quantiles = c.execute(psi_baseline_query, prev_params)[0]
            current_quantiles = c.execute(psi_baseline_query, params)[0]

            # Simple PSI estimation based on quantile shifts
            shifts = []
            for i, (b, c_val) in enumerate(zip(baseline_quantiles, current_quantiles)):
                if b and b > 0:
                    shift = abs(c_val - b) / b
                    shifts.append(shift)

            if shifts:
                avg_shift = sum(shifts) / len(shifts)
                psi_status = "Стабильно" if avg_shift < 0.1 else "Умеренный дрейф" if avg_shift < 0.25 else "Критический дрейф"
            else:
                psi_status = "Недостаточно данных"
        else:
            psi_status = "Недостаточно данных"
    except Exception as e:
        print(f"[WARN] Error calculating PSI: {e}")
        psi_status = "Ошибка расчета"

    # Generate Markdown report
    out_md = f"reports/weekly_report_{end:%Y%m%d}.md"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Weekly Report — {start} … {end}\n\n")
        f.write(f"**Источник данных:** `{table}`\n\n")

        # KPI section
        f.write("## 1) Ключевые показатели (KPI)\n\n")
        f.write(f"| Метрика | Текущая неделя | Прошлая неделя | Изменение |\n")
        f.write(f"|---------|----------------|----------------|----------|\n")

        # Calculate volume change percentage safely
        vol_change_pct = ((vol / vol_p - 1) * 100) if vol_p > 0 else 0
        vol_change_str = f"{vol_change_pct:+.1f}%" if vol_p > 0 else "н/д"

        f.write(f"| **Объём (UZS)** | {int(vol):,} | {int(vol_p):,} | {int(vol - vol_p):+,} ({vol_change_str}) |\n")
        f.write(f"| **Транзакций** | {int(tx):,} | {int(tx_p):,} | {int(tx - tx_p):+,} |\n")

        # Only show P2P if we have the column
        if has_p2p:
            f.write(f"| **P2P доля** | {p2p:.2f}% | {p2p_p:.2f}% | {p2p - p2p_p:+.2f}% |\n")

        f.write("\n")

        # Top MCC
        f.write("## 2) Топ MCC по объёму\n\n")
        if not top_mcc.empty:
            f.write("| MCC | Объём (UZS) | Доля |\n")
            f.write("|-----|-------------|------|\n")
            total_vol = top_mcc['volume'].sum()
            for _, row in top_mcc.head(10).iterrows():
                share = row['volume'] / total_vol * 100 if total_vol > 0 else 0
                f.write(f"| {int(row['mcc'])} | {int(row['volume']):,} | {share:.1f}% |\n")
        else:
            f.write("_Данные отсутствуют_\n")
        f.write("\n")

        # Top Merchants
        f.write("## 3) Топ Merchants по объёму\n\n")
        if not top_merch.empty:
            f.write("| Merchant | Объём (UZS) |\n")
            f.write("|----------|-------------|\n")
            for _, row in top_merch.head(10).iterrows():
                f.write(f"| {row['merchant_name'][:50]} | {int(row['volume']):,} |\n")
        else:
            f.write("_Данные отсутствуют_\n")
        f.write("\n")

        # Hourly seasonality
        f.write("## 4) Сезонность по часам\n\n")
        if not hourly.empty:
            f.write("| Час | Объём (UZS) |\n")
            f.write("|-----|-------------|\n")
            for _, row in hourly.iterrows():
                f.write(f"| {int(row['hour']):02d}:00 | {int(row['volume']):,} |\n")
        else:
            f.write("_Данные отсутствуют_\n")
        f.write("\n")

        # Data Quality
        f.write("## 5) Качество данных за неделю\n\n")
        f.write(f"- **amount_uzs заполнены:** {fill_amt:.1f}%\n")

        if has_column(c, table, "mcc"):
            f.write(f"- **MCC заполнены:** {fill_mcc:.1f}%\n")

        if has_column(c, table, "merchant_name"):
            f.write(f"- **merchant_name заполнены:** {fill_mer:.1f}%\n")

        f.write("\n")

        # Mini PSI
        f.write("## 6) Мониторинг дрейфа (мини-PSI)\n\n")
        f.write(f"**Статус распределения amount_uzs:** {psi_status}\n\n")
        f.write("_PSI оценивается по сдвигу квантилей между текущей и прошлой неделей_\n\n")

        # Footer
        f.write("---\n")
        f.write(f"_Отчёт сгенерирован: {datetime.now():%Y-%m-%d %H:%M:%S}_\n")

    print(f"[OK] Weekly report saved to: {out_md}")

    # Print first part of report to console
    with open(out_md, "r", encoding="utf-8") as f:
        content = f.read()
        print("\n" + "=" * 50)
        print("REPORT PREVIEW:")
        print("=" * 50)
        print(content[:800])
        if len(content) > 800:
            print("\n... (truncated)")


if __name__ == "__main__":
    main()