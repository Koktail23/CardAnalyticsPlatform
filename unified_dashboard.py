#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Analytics Dashboard — Enhanced Version with improved RAG & Forecasting
- TF-IDF based RAG with citations
- Forecast metrics & confidence intervals
- Rolling origin validation
- Top-down reconciliation option
"""

import os
import sys
import io
import json
import subprocess
import warnings
import joblib
import hashlib
from pathlib import Path
from threading import RLock
from typing import Optional, Dict, List, Tuple
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from clickhouse_driver import Client
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# =========================
# ENV / PAGE CONFIG
# =========================
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
st.set_page_config(page_title="Unified Analytics Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
.metric-card { padding: 10px 14px; border-radius: 12px; background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.25); margin: 6px 0; }
.small-note { color: #666; font-size: 12px }
.kbd { padding: 2px 6px; border:1px solid #ccc; border-bottom-width:2px; border-radius:4px; background:#f7f7f7; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
.citation { color: #6366f1; font-size: 11px; vertical-align: super; }
.source-box { padding: 8px; background: #f8f9fa; border-left: 3px solid #6366f1; margin: 4px 0; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# =========================
# PATHS / ARTIFACTS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = Path(os.getenv("MODELS_PATH", "ml"))
MODELS_DIR_ABS = (PROJECT_ROOT / MODELS_DIR).resolve()
REPORTS_DIR = Path(os.getenv("REPORTS_PATH", "./reports")).resolve()
CACHE_DIR = REPORTS_DIR / ".rag_cache"

for p in [MODELS_DIR_ABS, REPORTS_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Add ml directory to path for imports
ml_path = PROJECT_ROOT / "ml"
if ml_path.exists():
    sys.path.insert(0, str(ml_path))
    print(f"[INFO] Added {ml_path} to sys.path")
else:
    print(f"[WARN] ML directory not found at {ml_path}")


def find_artifact(name: str) -> str:
    for p in [
        PROJECT_ROOT / name,
        MODELS_DIR_ABS / name,
        PROJECT_ROOT / "models" / name,
        MODELS_DIR_ABS / "models" / name,
    ]:
        if p.exists():
            return str(p)
    return str(PROJECT_ROOT / name)


# =========================
# Enhanced modules imports
# =========================
try:
    from rag_enhanced import EnhancedRAG, create_enhanced_prompt

    HAS_ENHANCED_RAG = True
    print("[INFO] Enhanced RAG module loaded successfully")
except ImportError as e:
    HAS_ENHANCED_RAG = False
    print(f"[WARN] Enhanced RAG not available: {e}")

try:
    from forecasting_enhanced import EnhancedForecaster, ForecastMetrics, TimeSeriesDiagnostics

    HAS_ENHANCED_FORECAST = True
    print("[INFO] Enhanced Forecasting module loaded successfully")
except ImportError as e:
    HAS_ENHANCED_FORECAST = False
    print(f"[WARN] Enhanced Forecasting not available: {e}")


# =========================
# RAG Cache Helper
# =========================
def get_rag_cache_key(weeks: int) -> str:
    """Generate cache key based on latest report files."""
    files_to_check = []

    # Check weekly reports
    weekly = sorted(REPORTS_DIR.glob("weekly_report_*.md"),
                    key=lambda p: p.stat().st_mtime, reverse=True)[:weeks]
    files_to_check.extend(weekly)

    # Check other reports
    for pattern in ["dq_report_*.md", "data_dictionary_*.md"]:
        latest = sorted(REPORTS_DIR.glob(pattern),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:1]
        files_to_check.extend(latest)

    # Create hash from file dates
    if files_to_check:
        dates_str = "_".join([f"{f.stem}_{f.stat().st_mtime}" for f in files_to_check])
        return hashlib.md5(dates_str.encode()).hexdigest()
    return "empty"


@st.cache_resource
def get_cached_rag(weeks: int = 4):
    """Get cached RAG instance or create new one."""
    cache_key = get_rag_cache_key(weeks)
    cache_file = CACHE_DIR / f"rag_{cache_key}_{weeks}w.pkl"

    # Check if valid cache exists
    if cache_file.exists():
        try:
            with st.spinner("Loading cached RAG model..."):
                rag = joblib.load(cache_file)
                print(f"[INFO] Loaded RAG from cache: {cache_file}")
                return rag
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")

    # Create new RAG instance
    with st.spinner("Building RAG index..."):
        rag = EnhancedRAG()
        rag.documents = rag.load_documents(last_n_weeks=weeks)
        rag.chunk_documents()

        # Save to cache
        try:
            # Clean old cache files
            for old_file in CACHE_DIR.glob("rag_*.pkl"):
                if old_file != cache_file:
                    old_file.unlink()

            joblib.dump(rag, cache_file)
            print(f"[INFO] Saved RAG to cache: {cache_file}")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

    return rag


# =========================
# Suggested Prompts for RAG
# =========================
SUGGESTED_PROMPTS = [
    "Какая динамика P2P платежей за последний месяц?",
    "Есть ли риски по качеству данных?",
    "Какие MCC дали наибольший вклад в рост?",
    "Что изменилось за последние 4 недели?",
    "Покажи аномалии в транзакциях за неделю",
    "Какие метрики модели fraud detection?",
    "Есть ли сезонность в объемах транзакций?",
    "Топ-5 мерчантов по объему за неделю",
    "Сравни текущую неделю с прошлой",
    "Какой прогноз на следующие 7 дней?"
]

# =========================
# MLflow helpers (safe)
# =========================
HAS_MLFLOW = False
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


@st.cache_data(ttl=60, show_spinner=False)
def mlflow_last_run(exp_name: str):
    if not HAS_MLFLOW:
        return None
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        exp = None
        for e in client.search_experiments():
            if e.name == exp_name:
                exp = e
                break
        if exp is None:
            return None
        runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
        if not runs:
            return None
        run = runs[0]
        arts = []
        try:
            for a in client.list_artifacts(run.info.run_id):
                arts.append(a.path)
        except Exception:
            pass
        return dict(run_id=run.info.run_id, start_time=run.info.start_time, metrics=run.data.metrics,
                    params=run.data.params, artifacts=arts)
    except Exception:
        return None


# =========================
# ClickHouse client (locked)
# =========================
@st.cache_resource
def get_client():
    host = os.getenv("CLICKHOUSE_HOST") or os.getenv("CH_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT") or os.getenv("CH_PORT", "9000"))
    user = os.getenv("CLICKHOUSE_USER") or os.getenv("CH_USER", "analyst")
    password = os.getenv("CLICKHOUSE_PASSWORD") or os.getenv("CH_PASSWORD", "admin123")
    database = os.getenv("CLICKHOUSE_DATABASE") or os.getenv("CH_DB", "card_analytics")
    return Client(
        host=host, port=port, user=user, password=password, database=database,
        connect_timeout=3, send_receive_timeout=6, sync_request_timeout=6,
        settings={"max_execution_time": 8, "use_uncompressed_cache": 0, "strings_encoding": "utf-8"},
    )


client = get_client()
_EXEC_LOCK = RLock()
_orig_execute = client.execute


def _locked_execute(*args, **kwargs):
    with _EXEC_LOCK:
        return _orig_execute(*args, **kwargs)


client.execute = _locked_execute  # type: ignore


@st.cache_data(ttl=300, show_spinner=False)
def ch_rows(sql: str) -> List[tuple]:
    with _EXEC_LOCK:
        return client.execute(sql)


@st.cache_data(ttl=300, show_spinner=False)
def ch_df(sql: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    rows = ch_rows(sql)
    return pd.DataFrame(rows, columns=columns)


# =========================
# Helpers (schema/typed sql)
# =========================
def describe_table(table: str) -> Dict[str, str]:
    rows = ch_rows(f"DESCRIBE {table}")
    return {r[0]: r[1] for r in rows}


def list_transaction_tables() -> List[str]:
    return [r[0] for r in ch_rows("SHOW TABLES") if r[0].startswith("transactions_")]


def base_type(ch_t: Optional[str]) -> str:
    if not ch_t: return ""
    return ch_t[9:-1] if ch_t.startswith("Nullable(") and ch_t.endswith(")") else ch_t


def is_numeric_schema(table: str) -> bool:
    d = describe_table(table)

    def is_num(x: str) -> bool: return any(x.startswith(p) for p in ("UInt", "Int", "Float", "Decimal"))

    return any(is_num(base_type(d.get(c, "String"))) for c in ("amount_uzs", "hour_num", "mcc"))


def human_name(t: str) -> str:
    return t.replace("transactions_", "").replace("_", " ").title()


def has_col(table: str, name: str) -> bool:
    try:
        return name in describe_table(table)
    except Exception:
        return False


def detect_hour_expr(table: str) -> Optional[str]:
    if has_col(table, "hour_num"):
        return "toUInt32OrNull(toString(hour_num))"
    if has_col(table, "transaction_date"):
        return "toHour(toDateTime(transaction_date))"
    return None

# --- NEW: date helpers for anchored windows ---
@st.cache_data(ttl=300, show_spinner=False)  # Добавьте кэширование!
def get_max_date(table: str) -> Optional[pd.Timestamp]:
    """Возвращает последнюю дату в таблице (по transaction_date или rday)."""
    try:
        if has_col(table, "transaction_date"):
            d = ch_rows(f"SELECT max(toDate(transaction_date)) FROM {table}")[0][0]
            return pd.to_datetime(d) if d else None
        if has_col(table, "rday"):
            d = ch_rows(f"SELECT max(toDate('1900-01-01') + toUInt32OrNull(rday)) FROM {table}")[0][0]
            return pd.to_datetime(d) if d else None
    except Exception as e:
        print(f"[ERROR] get_max_date failed: {e}")
    return None

def date_filter_sql(table: str, days: int, end: Optional[pd.Timestamp] = None) -> Tuple[str, pd.Timestamp, pd.Timestamp]:
    """Строит WHERE‑фильтр «последние N дней», якорясь на max(date) в источнике."""
    end = (end or get_max_date(table) or pd.Timestamp(date.today())).normalize()
    start = (end - pd.Timedelta(days=days - 1)).normalize()
    if has_col(table, "transaction_date"):
        filt = f"toDate(transaction_date) BETWEEN toDate('{start.date()}') AND toDate('{end.date()}')"
    elif has_col(table, "rday"):
        filt = f"(toDate('1900-01-01') + toUInt32OrNull(rday)) BETWEEN toDate('{start.date()}') AND toDate('{end.date()}')"
    else:
        filt = "1"
    return filt, start, end
# --- /NEW ---

def mcc_expr(table: str) -> Optional[str]:
    if has_col(table, "mcc"):
        if is_numeric_schema(table):
            return "mcc"
        else:
            return "coalesce(toUInt32OrNull(extract(toString(mcc), '\\\\d+')), 0)"
    return None


def scalar(sql: str, default=None):
    try:
        return ch_rows(sql)[0][0]
    except Exception:
        return default


def volume_expr(numeric: bool) -> str:
    return "sum(amount_uzs)" if numeric else "sum(coalesce(toFloat64OrNull(toString(amount_uzs)),0))"


def get_numeric_columns(table: str) -> List[str]:
    d = describe_table(table)
    numeric_cols = []

    def is_num(x: str) -> bool:
        return any(x.startswith(p) for p in ("UInt", "Int", "Float", "Decimal"))

    for col, dtype in d.items():
        if is_num(base_type(dtype)):
            numeric_cols.append(col)
    return numeric_cols


def get_categorical_columns(table: str) -> List[str]:
    d = describe_table(table)
    categorical_cols = []
    numeric = get_numeric_columns(table)
    for col in d.keys():
        if col not in numeric:
            categorical_cols.append(col)
    return categorical_cols


# =========================
# PSI Calculation
# =========================
def calculate_psi(baseline: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Calculate Population Stability Index."""
    try:
        # Create quantile bins from baseline
        _, bin_edges = pd.qcut(baseline.dropna(), q=n_bins, retbins=True, duplicates='drop')

        # Apply same bins to both distributions
        baseline_binned = pd.cut(baseline, bins=bin_edges, include_lowest=True)
        current_binned = pd.cut(current, bins=bin_edges, include_lowest=True)

        # Calculate proportions
        baseline_prop = baseline_binned.value_counts(normalize=True)
        current_prop = current_binned.value_counts(normalize=True)

        # Align indexes
        baseline_prop = baseline_prop.reindex(baseline_prop.index.union(current_prop.index), fill_value=0.001)
        current_prop = current_prop.reindex(baseline_prop.index, fill_value=0.001)

        # Calculate PSI
        psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))
        return float(psi)
    except Exception:
        return np.nan


# =========================
# Sidebar
# =========================
st.sidebar.header("📊 Источник данных")
tables = list_transaction_tables()
if not tables:
    st.sidebar.error("⌠Таблицы вида transactions_* не найдены")
    st.stop()

pairs = []
for t in tables:
    try:
        cnt = scalar(f"SELECT count() FROM {t}", 0)
        if cnt > 0: pairs.append((t, cnt))
    except Exception:
        pass
pairs.sort(key=lambda x: (-x[1], x[0]))

show = [f"{human_name(t)} ({cnt:,})" for t, cnt in pairs]
sel = st.sidebar.selectbox("Таблица для анализа:", show, index=0)
idx = show.index(sel)
selected_table = pairs[idx][0]
st.sidebar.success(f"✅ Используется: **{selected_table}**")

numeric = is_numeric_schema(selected_table)
st.sidebar.caption("Схема: " + ("числовая ✅" if numeric else "строковая ⚠️"))

if st.sidebar.button("🔄 Обновить", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# =========================
# Tabs
# =========================
tabs = [
    "📈 Dashboard", "🔎 Аналитика", "🧪 EDA", "📈 Drift & Monitoring", "🚨 Fraud Detection",
    "👥 Segmentation", "📊 Forecasting", "🏗 Hierarchical Forecast", "📝 Weekly Report / RAG",
    "💳 Транзакции", "📋 Статус+DQ", "📚 Dictionary", "😊 Chat / NL2SQL",
]
(tab1, tab2, tabEDA, tabDrift, tab3, tab4, tab5, tabHier, tabReport, tab6, tab7, tabDict, tab8) = st.tabs(tabs)

# ======================================================
# TAB 1: Dashboard
# ======================================================
with tab1:
    st.header("Dashboard — ключевые метрики")
    st.caption(f"Источник: **{selected_table}**")

    try:
        total = scalar(f"SELECT count() FROM {selected_table}", 0)
        vol = scalar(f"SELECT {volume_expr(numeric)} FROM {selected_table}", 0.0)
        cards = scalar(f"SELECT uniq(hpan) FROM {selected_table}", None) if has_col(selected_table, "hpan") else None
        p2p = scalar(f"SELECT countIf(toString(p2p_flag)='1') FROM {selected_table}", 0) if has_col(selected_table,
                                                                                                    "p2p_flag") else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📢 Транзакций", f"{total:,}")
        c2.metric("💰 Объём", f"{(vol or 0):,.0f} UZS")
        c3.metric("💳 Уникальных карт", "-" if cards is None else f"{cards:,}")
        c4.metric("💸 P2P доля", f"{(p2p / total * 100 if total else 0):.1f}%")

        st.divider()
        colA, colB = st.columns(2)

        # Объём по дням
        with colA:
            if numeric and has_col(selected_table, "transaction_date"):
                q = f"""
                    SELECT transaction_date AS date, {volume_expr(True)} AS volume
                    FROM {selected_table} WHERE transaction_date IS NOT NULL
                    GROUP BY date ORDER BY date
                """
            elif has_col(selected_table, "rday"):
                q = f"""
                    SELECT toDate('1900-01-01') + toUInt32OrNull(rday) AS date,
                           {volume_expr(False)} AS volume
                    FROM {selected_table} WHERE toUInt32OrNull(rday) IS NOT NULL
                    GROUP BY date ORDER BY date
                """
            else:
                q = None
            if q:
                daily = ch_df(q, ["date", "volume"])
                if not daily.empty:
                    fig = px.area(daily, x='date', y='volume', title='📈 Объём по дням',
                                  labels={'volume': 'UZS', 'date': 'Дата'})
                    fig.update_traces(fillcolor='rgba(102,126,234,0.45)')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет дат для построения дневной динамики.")

        # Топ MCC
        with colB:
            mcc_e = mcc_expr(selected_table)
            if mcc_e:
                q_top = f"""
                    WITH all_mcc AS (
                        SELECT {mcc_e} AS mcc, {volume_expr(numeric)} AS volume
                        FROM {selected_table} GROUP BY mcc
                    ), top AS (
                        SELECT mcc, volume, row_number() OVER (ORDER BY volume DESC) rn FROM all_mcc
                    ), tops AS ( SELECT mcc, volume FROM top WHERE rn <= 9 ),
                    sum_top AS (SELECT sum(volume) s FROM tops),
                    sum_all AS (SELECT sum(volume) s FROM all_mcc),
                    other AS ( SELECT toUInt32(0) AS mcc, (sum_all.s - sum_top.s) AS volume FROM sum_all, sum_top )
                    SELECT mcc, volume FROM tops
                    UNION ALL
                    SELECT mcc, volume FROM other
                    ORDER BY volume DESC
                """
                dfm = ch_df(q_top, ["mcc", "volume"])
                if not dfm.empty:
                    alias = {5411: "Grocery", 6011: "ATM", 6012: "Financial", 4814: "Telecom",
                             5541: "Fuel", 5812: "Restaurants", 5912: "Pharmacies", 5999: "Retail", 0: "OTHER"}
                    dfm["label"] = dfm["mcc"].astype(int).map(lambda x: f"{alias.get(x, f'MCC {x}')}")
                    dfm["share"] = dfm["volume"] / (dfm["volume"].sum() or 1)
                    fig = px.bar(dfm.sort_values("volume"),
                                 x="volume", y="label", orientation="h",
                                 text=dfm["share"].map(lambda v: f"{v * 100:.1f}%"),
                                 title="🛒 Топ MCC по объёму (+ OTHER)")
                    fig.update_traces(textposition="outside")
                    fig.update_layout(xaxis_title="UZS", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Нет колонки MCC — раздел скрыт.")

    except Exception as e:
        st.error(f"Ошибка Dashboard: {e}")

# ===================================
# TAB 2: Аналитика
# ===================================
with tab2:
    st.header("🔍 Углублённая аналитика")
    st.caption(f"Источник: **{selected_table}**")
    atype = st.radio("Тип анализа", ["Анализ по банкам", "Анализ по мерчантам", "Временной анализ", "P2P анализ"],
                     horizontal=True)

    try:
        if atype == "Анализ по банкам":
            if not has_col(selected_table, "emitent_bank"):
                st.info(
                    "В этой таблице нет колонки `emitent_bank`. Попробуйте другой тип анализа (например, «Анализ по мерчантам»).")
            else:
                q = f"""
                SELECT emitent_bank, count() cnt, {volume_expr(numeric)} volume,
                       {("avg(amount_uzs)" if numeric else "avg(coalesce(toFloat64OrNull(toString(amount_uzs)),0))")} avg_amount,
                       {("uniq(hpan)" if has_col(selected_table, 'hpan') else "0")} cards
                FROM {selected_table} WHERE toString(emitent_bank)!=''
                GROUP BY emitent_bank ORDER BY cnt DESC LIMIT 20
                """
                dfb = ch_df(q, ["bank", "count", "volume", "avg_amount", "cards"])
                if not dfb.empty:
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(px.bar(dfb, x='bank', y='count', title='По количеству'), use_container_width=True)
                    c2.plotly_chart(px.bar(dfb, x='bank', y='volume', title='По объёму'), use_container_width=True)
                else:
                    st.info("Нет данных по банкам-эмитентам.")

        elif atype == "Анализ по мерчантам":
            if has_col(selected_table, "merchant_name"):
                q = f"""
                SELECT toString(merchant_name) AS merchant, count() cnt, {volume_expr(numeric)} volume
                FROM {selected_table} WHERE toString(merchant_name)!=''
                GROUP BY merchant ORDER BY volume DESC LIMIT 25
                """
                dfm = ch_df(q, ["merchant", "count", "volume"])
                if not dfm.empty:
                    st.plotly_chart(px.bar(dfm, x='merchant', y='volume', title='Топ мерчантов'),
                                    use_container_width=True)
                else:
                    st.info("Нет данных по мерчантам.")
            else:
                st.info("В этой таблице нет колонки `merchant_name`.")

        elif atype == "Временной анализ":
            if numeric and has_col(selected_table, "transaction_date"):
                q = f"""SELECT transaction_date AS date, {volume_expr(True)} volume
                        FROM {selected_table} WHERE transaction_date IS NOT NULL
                        GROUP BY date ORDER BY date"""
            elif has_col(selected_table, "rday"):
                q = f"""SELECT toDate('1900-01-01')+toUInt32OrNull(rday) AS date, {volume_expr(False)} volume
                        FROM {selected_table} WHERE toUInt32OrNull(rday) IS NOT NULL
                        GROUP BY date ORDER BY date"""
            else:
                q = None
            if q:
                dft = ch_df(q, ["date", "volume"])
                if not dft.empty:
                    st.plotly_chart(px.line(dft, x='date', y='volume', title='Объёмы по датам'),
                                    use_container_width=True)
            else:
                st.info("Нет колонок даты/дня для временного анализа.")

        else:  # P2P
            h = detect_hour_expr(selected_table)
            if h and has_col(selected_table, "p2p_flag"):
                q = f"""
                SELECT {h} AS hour, count() total,
                       countIf(toString(p2p_flag)='1') AS p2p
                FROM {selected_table} WHERE {h} BETWEEN 0 AND 23
                GROUP BY hour ORDER BY hour
                """
                dfp = ch_df(q, ["hour", "total", "p2p"])
                if not dfp.empty:
                    dfp["ratio"] = (dfp["p2p"] / dfp["total"] * 100).fillna(0)
                    st.plotly_chart(px.line(dfp, x='hour', y='ratio', title='P2P доля по часам'),
                                    use_container_width=True)
                else:
                    st.info("Нет данных для P2P анализа.")
            else:
                st.info("Нет колонки часа или p2p_flag — P2P анализ недоступен.")

    except Exception as e:
        st.error(f"Ошибка аналитики: {e}")

# ===================================
# TAB 3: EDA
# ===================================
with tabEDA:
    st.header("🧪 EDA - Exploratory Data Analysis")
    st.caption(f"Источник: **{selected_table}**")

    try:
        # Get numeric and categorical columns
        numeric_cols = get_numeric_columns(selected_table)
        categorical_cols = get_categorical_columns(selected_table)

        # Sample data for analysis
        sample_size = st.number_input("Размер выборки для анализа", 1000, 100000, 10000, step=1000)
        df_sample = ch_df(f"SELECT * FROM {selected_table} LIMIT {sample_size}")

        if not df_sample.empty:
            cols = [c[0] for c in ch_rows(f"DESCRIBE {selected_table}")]
            df_sample.columns = cols[:len(df_sample.columns)]

            col1, col2 = st.columns(2)

            # Numeric statistics
            with col1:
                st.subheader("📊 Числовые колонки")
                if numeric_cols:
                    stats_data = []
                    for col in numeric_cols:
                        if col in df_sample.columns:
                            series = pd.to_numeric(df_sample[col], errors='coerce')
                            stats_data.append({
                                'Column': col,
                                'Count': series.count(),
                                'Mean': series.mean(),
                                'Std': series.std(),
                                'Min': series.min(),
                                'P25': series.quantile(0.25),
                                'P50': series.quantile(0.50),
                                'P75': series.quantile(0.75),
                                'Max': series.max()
                            })
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df.round(2), use_container_width=True)
                else:
                    st.info("Числовые колонки не обнаружены")

            # Categorical statistics
            with col2:
                st.subheader("🔡 Категориальные колонки")
                if categorical_cols:
                    cat_stats = []
                    for col in categorical_cols[:10]:  # Top 10
                        if col in df_sample.columns:
                            unique_count = df_sample[col].nunique()
                            null_count = df_sample[col].isna().sum()
                            cat_stats.append({
                                'Column': col,
                                'Unique': unique_count,
                                'Nulls': null_count,
                                'Null %': f"{null_count / len(df_sample) * 100:.1f}%"
                            })
                    if cat_stats:
                        cat_df = pd.DataFrame(cat_stats)
                        st.dataframe(cat_df, use_container_width=True)
                else:
                    st.info("Категориальные колонки не обнаружены")

            st.divider()

            # Distribution plots
            col3, col4 = st.columns(2)

            with col3:
                if has_col(selected_table, "amount_uzs"):
                    st.subheader("💰 Распределение amount_uzs")
                    log_scale = st.checkbox("Логарифмическая шкала", value=False)

                    if numeric:
                        q = f"SELECT amount_uzs FROM {selected_table} WHERE amount_uzs > 0 LIMIT {sample_size}"
                    else:
                        q = f"SELECT coalesce(toFloat64OrNull(toString(amount_uzs)), 0) AS amount_uzs FROM {selected_table} WHERE coalesce(toFloat64OrNull(toString(amount_uzs)), 0) > 0 LIMIT {sample_size}"

                    amounts = ch_df(q, ["amount_uzs"])
                    if not amounts.empty:
                        if log_scale:
                            amounts["amount_uzs"] = np.log10(amounts["amount_uzs"] + 1)
                            title = "Распределение log10(amount_uzs)"
                        else:
                            title = "Распределение amount_uzs"

                        fig = px.histogram(amounts, x="amount_uzs", nbins=50, title=title)
                        st.plotly_chart(fig, use_container_width=True)

            with col4:
                if has_col(selected_table, "mcc"):
                    st.subheader("🛒 Топ частот MCC")
                    mcc_e = mcc_expr(selected_table)
                    if mcc_e:
                        q = f"""
                        SELECT {mcc_e} AS mcc, count() AS cnt
                        FROM {selected_table}
                        WHERE {mcc_e} > 0
                        GROUP BY mcc
                        ORDER BY cnt DESC
                        LIMIT 15
                        """
                        mcc_freq = ch_df(q, ["mcc", "count"])
                        if not mcc_freq.empty:
                            fig = px.bar(mcc_freq, x="mcc", y="count", title="Топ-15 MCC по частоте")
                            st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap
            st.divider()
            if len(numeric_cols) >= 2:
                st.subheader("🔥 Корреляционная матрица")
                corr_cols = st.multiselect(
                    "Выберите колонки для корреляции",
                    [c for c in numeric_cols if c in df_sample.columns],
                    default=[c for c in ["amount_uzs", "mcc", "hour_num", "p2p_flag"] if
                             c in numeric_cols and c in df_sample.columns][:5]
                )

                if len(corr_cols) >= 2:
                    corr_data = df_sample[corr_cols].apply(pd.to_numeric, errors='coerce')
                    corr_matrix = corr_data.corr()

                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        title="Матрица корреляций"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка EDA: {e}")

# ===================================
# TAB 4: Drift & Monitoring
# ===================================
with tabDrift:
    st.header("📈 Drift & Monitoring")
    st.caption(f"Источник: **{selected_table}**")

    try:
        # Date detection
        date_col = None
        if has_col(selected_table, "transaction_date"):
            date_col = "transaction_date"
        elif has_col(selected_table, "rday"):
            date_col = "toDate('1900-01-01') + toUInt32OrNull(rday)"

        if date_col:
            # Get date range
            if date_col == "transaction_date":
                min_date = scalar(f"SELECT min({date_col}) FROM {selected_table}")
                max_date = scalar(f"SELECT max({date_col}) FROM {selected_table}")
            else:
                min_date = scalar(f"SELECT min({date_col}) FROM {selected_table}")
                max_date = scalar(f"SELECT max({date_col}) FROM {selected_table}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Baseline период")
                baseline_start = st.date_input("Начало", value=min_date if min_date else date.today())
                baseline_end = st.date_input("Конец", value=min_date + timedelta(days=14) if min_date else date.today())

            with col2:
                st.subheader("Current период")
                current_start = st.date_input("Начало",
                                              value=max_date - timedelta(days=14) if max_date else date.today())
                current_end = st.date_input("Конец", value=max_date if max_date else date.today())

            if st.button("📊 Рассчитать PSI"):
                with st.spinner("Вычисляем PSI..."):
                    # Prepare date columns
                    if date_col == "transaction_date":
                        date_expr = date_col
                    else:
                        date_expr = date_col

                    # Get baseline data
                    baseline_query = f"""
                    SELECT 
                        {('amount_uzs' if numeric else 'coalesce(toFloat64OrNull(toString(amount_uzs)), 0)')} AS amount_uzs
                        {(', ' + mcc_expr(selected_table) + ' AS mcc' if has_col(selected_table, 'mcc') else '')}
                        {(', ' + detect_hour_expr(selected_table) + ' AS hour_num' if detect_hour_expr(selected_table) else '')}
                    FROM {selected_table}
                    WHERE {date_expr} BETWEEN '{baseline_start}' AND '{baseline_end}'
                    LIMIT 100000
                    """

                    current_query = f"""
                    SELECT 
                        {('amount_uzs' if numeric else 'coalesce(toFloat64OrNull(toString(amount_uzs)), 0)')} AS amount_uzs
                        {(', ' + mcc_expr(selected_table) + ' AS mcc' if has_col(selected_table, 'mcc') else '')}
                        {(', ' + detect_hour_expr(selected_table) + ' AS hour_num' if detect_hour_expr(selected_table) else '')}
                    FROM {selected_table}
                    WHERE {date_expr} BETWEEN '{current_start}' AND '{current_end}'
                    LIMIT 100000
                    """

                    baseline_df = ch_df(baseline_query)
                    current_df = ch_df(current_query)

                    # Set column names
                    cols = ["amount_uzs"]
                    if has_col(selected_table, "mcc"):
                        cols.append("mcc")
                    if detect_hour_expr(selected_table):
                        cols.append("hour_num")

                    baseline_df.columns = cols[:len(baseline_df.columns)]
                    current_df.columns = cols[:len(current_df.columns)]

                    # Calculate PSI
                    psi_results = []

                    for col in baseline_df.columns:
                        if col in current_df.columns:
                            psi = calculate_psi(baseline_df[col], current_df[col])
                            psi_results.append({
                                "Feature": col,
                                "PSI": psi,
                                "Status": "🟢 Stable" if psi < 0.1 else "🟡 Warning" if psi < 0.25 else "🔴 Critical"
                            })

                    # Display results
                    st.subheader("📊 PSI Results")
                    psi_df = pd.DataFrame(psi_results)
                    st.dataframe(psi_df, use_container_width=True)

                    # PSI interpretation
                    with st.expander("📖 Интерпретация PSI"):
                        st.markdown("""
                        **Population Stability Index (PSI)** измеряет стабильность распределения:
                        - **PSI < 0.1** 🟢 - Стабильное распределение, изменения незначительны
                        - **PSI 0.1-0.25** 🟡 - Умеренный дрейф, требует внимания
                        - **PSI > 0.25** 🔴 - Критический дрейф, требует исследования
                        """)

                    # Volume comparison chart
                    st.divider()
                    st.subheader("📈 Сравнение объёмов по периодам")

                    volume_query = f"""
                    SELECT 
                        {date_expr} AS date,
                        {volume_expr(numeric)} AS volume,
                        CASE 
                            WHEN {date_expr} BETWEEN '{baseline_start}' AND '{baseline_end}' THEN 'Baseline'
                            WHEN {date_expr} BETWEEN '{current_start}' AND '{current_end}' THEN 'Current'
                        END AS period
                    FROM {selected_table}
                    WHERE {date_expr} BETWEEN '{baseline_start}' AND '{baseline_end}'
                       OR {date_expr} BETWEEN '{current_start}' AND '{current_end}'
                    GROUP BY date, period
                    ORDER BY date
                    """

                    volume_df = ch_df(volume_query, ["date", "volume", "period"])

                    if not volume_df.empty:
                        fig = px.line(volume_df, x="date", y="volume", color="period",
                                      title="Объёмы транзакций: Baseline vs Current")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Для мониторинга дрейфа требуется колонка с датой (transaction_date или rday)")

    except Exception as e:
        st.error(f"Ошибка Drift Monitoring: {e}")

# ===================================
# TAB 5: Fraud Detection
# ===================================
with tab3:
    st.header("🚨 Fraud Detection Analytics")
    st.caption("Анализ модели обнаружения мошенничества")

    try:
        last = mlflow_last_run("fraud_detection") if HAS_MLFLOW else None
        if last:
            m = last["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC", f"{m.get('auc', '—'):.3f}" if 'auc' in m else "—")
            c2.metric("Precision", f"{m.get('precision', '—'):.3f}" if 'precision' in m else "—")
            c3.metric("Recall", f"{m.get('recall', '—'):.3f}" if 'recall' in m else "—")
            c4.metric("F1", f"{m.get('f1', '—'):.3f}" if 'f1' in m else "—")
            st.caption(
                f"Последний запуск MLflow: run_id=`{last['run_id']}`; артефакты: {', '.join(last['artifacts']) or '—'}")
        else:
            st.info("Модель ещё не логировалась в MLflow. Обучите её из пайплайна/CLI.")

        # thresholds.json
        thr = MODELS_DIR_ABS / "thresholds.json"
        cm = MODELS_DIR_ABS / "cm_fraud.png"
        cL, cR = st.columns([1, 1])
        with cL:
            if thr.exists():
                j = json.loads(thr.read_text(encoding="utf-8"))
                st.subheader("Пороги/калибровка")
                st.json(j.get("thresholds", {}))
            else:
                st.caption("Файл thresholds.json не найден — выполните калибровку модели.")
        with cR:
            if cm.exists():
                st.subheader("Confusion Matrix")
                st.image(str(cm), use_container_width=True)
            else:
                st.caption("cm_fraud.png не найден — обучите модель.")

        with st.expander("📖 Что означают эти метрики?"):
            st.markdown("""
            ### Метрики производительности модели:

            **AUC (Area Under Curve)** — площадь под ROC-кривой. Значение от 0 до 1:
            - 0.5 = случайный классификатор
            - 0.7-0.8 = приемлемая модель
            - 0.8-0.9 = хорошая модель
            - > 0.9 = отличная модель

            **Precision (Точность)** — доля правильных предсказаний среди всех предсказанных фродов:
            - Precision = TP / (TP + FP)
            - Высокая precision = мало ложных срабатываний

            **Recall (Полнота)** — доля найденных фродов от всех реальных фродов:
            - Recall = TP / (TP + FN)
            - Высокий recall = мало пропущенных фродов

            **F1 Score** — гармоническое среднее precision и recall:
            - F1 = 2 × (Precision × Recall) / (Precision + Recall)
            - Баланс между точностью и полнотой

            ### Confusion Matrix (Матрица ошибок):

            ```
            Predicted
                 0    1
            0   TN   FP  ← Actual
            1   FN   TP
            ```

            - **TN (True Negative)** — верно определённые легитимные транзакции
            - **FP (False Positive)** — ложные срабатывания (легитимные помечены как фрод)
            - **FN (False Negative)** — пропущенные фроды (фрод помечен как легитимный)
            - **TP (True Positive)** — верно определённые фроды

            ### Пороги и калибровка:

            В файле `thresholds.json` хранятся оптимальные пороги:
            - **FPR ≈ 1%** — порог для низкого уровня ложных срабатываний (1%)
            - **FPR ≈ 2%** — порог для умеренного уровня ложных срабатываний (2%)
            - **EV Optimal** — экономически оптимальный порог с учётом Gain/Cost
            """)

    except Exception as e:
        st.error(f"Fraud: {e}")

# ===================================
# TAB 6: Segmentation
# ===================================
with tab4:
    st.header("👥 Customer Segmentation")
    try:
        seg = ch_df("""
            SELECT cs.rfm_segment, cs.rfm_score,
                   cf.txn_amount_30d, cf.txn_count_30d, cf.p2p_ratio_30d, cf.days_since_last_txn
            FROM customer_segments cs
            LEFT JOIN card_features cf ON cs.hpan = cf.hpan
        """, ["rfm_segment", "rfm_score", "txn_amount_30d", "txn_count_30d", "p2p_ratio_30d", "days_since_last_txn"])
        if seg.empty:
            st.info("Нет customer_segments/card_features. Запустите пайплайн сегментации.")
        else:
            summary = seg.groupby("rfm_segment").agg(
                clients=("rfm_score", "count"),
                avg_amount=("txn_amount_30d", "mean"),
                avg_txn=("txn_count_30d", "mean"),
                avg_recency=("days_since_last_txn", "mean"),
                avg_p2p=("p2p_ratio_30d", "mean")
            ).reset_index()
            summary["share"] = (summary["clients"] / summary["clients"].sum() * 100)
            c1, c2 = st.columns([2, 3])
            with c1:
                st.subheader("Распределение RFM")
                st.plotly_chart(px.pie(summary, values="clients", names="rfm_segment", hole=0.45),
                                use_container_width=True)
            with c2:
                st.subheader("Метрики по сегментам")
                show = summary[
                    ["rfm_segment", "clients", "share", "avg_amount", "avg_txn", "avg_recency", "avg_p2p"]].round(
                    2).rename(columns={
                    "rfm_segment": "Сегмент", "clients": "Клиентов", "share": "Доля,%", "avg_amount": "Avg объём",
                    "avg_txn": "Avg частота", "avg_recency": "Avg Recency (дн.)", "avg_p2p": "Avg P2P"
                })
                st.dataframe(show, use_container_width=True)
                st.download_button("⬇️ CSV (метрики)", data=show.to_csv(index=False).encode("utf-8"),
                                   file_name="segments_metrics.csv")

            st.subheader("R×F heatmap (из RFM_score)")
            tmp = seg.copy()
            tmp["R"] = tmp["rfm_score"].str[0].astype(str)
            tmp["F"] = tmp["rfm_score"].str[1].astype(str)
            hmp = tmp.pivot_table(index="R", columns="F", values="rfm_score", aggfunc="count").fillna(0)
            st.plotly_chart(px.imshow(hmp, text_auto=True, color_continuous_scale="Blues"), use_container_width=True)

            st.subheader("Рекомендации")
            st.markdown("""
- 🏆 Champions — удержание, персональные VIP-предложения
- 💎 Loyal — кросс/апсейл, программы лояльности
- 🌱 Potential — welcome-кампании
- ⚠️ At Risk — реактивация
- 😴 Hibernating — win-back
- ⌠Lost — агрессивный win-back или исключение
""")
    except Exception as e:
        st.error(f"Segmentation: {e}")

# ===================================
# TAB 7: Enhanced Forecasting
# ===================================
with tab5:
    st.header("📊 Enhanced Forecasting")
    st.caption(f"Прогнозирование с метриками качества и диагностикой")

    # Forecast mode selector
    forecast_mode = st.radio(
        "Режим прогнозирования",
        ["📈 Простой прогноз", "📬 Диагностика рядов", "📊 Валидация моделей", "🎯 Сценарный анализ"],
        horizontal=True
    )

    if forecast_mode == "📈 Простой прогноз":
        st.subheader("Настройки прогноза")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_to_forecast = st.selectbox(
                "Метрика",
                ["volume", "transactions", "unique_cards"],
                format_func=lambda x: {
                    "volume": "Объём (UZS)",
                    "transactions": "Количество транзакций",
                    "unique_cards": "Уникальные карты"
                }[x]
            )
        with col2:
            forecast_horizon = st.number_input("Горизонт (дней)", 7, 90, 14)
        with col3:
            holdout_days = st.number_input("Holdout (дней)", 0, 30, 14)
        with col4:
            confidence_level = st.selectbox("Доверительный интервал", [80, 90, 95], index=2)

        # Add reconciliation option
        reconcile_option = st.checkbox(
            "🔗 Top-down reconciliation by MCC",
            help="Распределить TOTAL прогноз на MCC по средним долям"
        )

        if st.button("🚀 Построить прогноз", type="primary"):
            if HAS_ENHANCED_FORECAST:
                try:
                    with st.spinner("Загрузка данных..."):
                        # Get max date from table first
                        max_date = get_max_date(selected_table)
                        if not max_date:
                            st.error("Не удалось определить максимальную дату в таблице")
                            st.stop()

                        # Use max_date as anchor, format as date
                        start_date = (max_date - timedelta(days=120)).date()
                        end_date = max_date.date()

                        # Get historical data
                        if metric_to_forecast == "volume":
                            query = f"""
                                SELECT 
                                    toDate(transaction_date) AS date,
                                    sum(amount_uzs) AS value
                                FROM {selected_table}
                                WHERE toDate(transaction_date) BETWEEN toDate('{start_date}') AND toDate('{end_date}')
                                GROUP BY date
                                ORDER BY date
                                """
                        elif metric_to_forecast == "transactions":
                            query = f"""
                                SELECT 
                                    toDate(transaction_date) AS date,
                                    count() AS value
                                FROM {selected_table}
                                WHERE toDate(transaction_date) BETWEEN toDate('{start_date}') AND toDate('{end_date}')
                                GROUP BY date
                                ORDER BY date
                                """
                        else:  # unique_cards
                            if has_col(selected_table, "hpan"):
                                query = f"""
                                    SELECT 
                                        toDate(transaction_date) AS date,
                                        uniq(hpan) AS value
                                    FROM {selected_table}
                                    WHERE toDate(transaction_date) BETWEEN toDate('{start_date}') AND toDate('{end_date}')
                                    GROUP BY date
                                    ORDER BY date
                                    """
                            else:
                                st.error("Нет колонки hpan для подсчета уникальных карт")
                                query = None

                        if query:
                            df = ch_df(query, ["date", "value"])
                            if not df.empty:
                                # Ensure date column is datetime
                                df["date"] = pd.to_datetime(df["date"])

                                # Remove NaN values
                                df = df.dropna()

                                if df.empty:
                                    st.warning(f"""
                                    ⚠️ Нет валидных данных за период {start_date} — {end_date}
                                    (все значения NULL)

                                    Максимальная дата в таблице: {max_date.date()}
                                    Попробуйте выбрать другую таблицу.
                                    """)
                                    st.stop()

                                # Check for zeros or negative values
                                if metric_to_forecast == "volume":
                                    df = df[df["value"] > 0]
                                    if df.empty:
                                        st.warning("Все значения объёма равны 0 или отрицательные")
                                        st.stop()

                                # Rename value column to metric name
                                df.rename(columns={"value": metric_to_forecast}, inplace=True)
                        else:
                            df = pd.DataFrame()

                    if not df.empty:
                        st.success(f"✅ Загружено {len(df)} дней данных")

                        # Initialize forecaster
                        forecaster = EnhancedForecaster()

                        # Prepare data
                        train, holdout = forecaster.prepare_data(
                            df,
                            value_col=metric_to_forecast,
                            holdout_days=holdout_days
                        )

                        with st.spinner("Генерация прогнозов..."):
                            # Generate multiple forecasts
                            forecasts = []
                            forecast_names = []

                            # Seasonal Naive
                            naive_fc = forecaster.seasonal_naive_forecast(
                                train, forecast_horizon, value_col=metric_to_forecast
                            )
                            forecasts.append(naive_fc)
                            forecast_names.append("Seasonal Naive")

                            # Prophet if available
                            try:
                                prophet_fc = forecaster.prophet_forecast(
                                    train, forecast_horizon, value_col=metric_to_forecast
                                )
                                if prophet_fc is not None:
                                    forecasts.append(prophet_fc)
                                    forecast_names.append("Prophet")
                            except Exception as e:
                                st.warning(f"Prophet недоступен: {e}")

                            # Ensemble
                            ensemble = forecaster.ensemble_forecast(forecasts)

                        # Evaluate on holdout if available
                        metrics_dict = {}
                        if len(holdout) > 0:
                            st.info(f"📊 Оценка на holdout периоде ({holdout_days} дней)")

                            cols = st.columns(len(forecasts) + 1)

                            for i, (fc, name) in enumerate(zip(forecasts, forecast_names)):
                                metrics = forecaster.evaluate_forecast(
                                    fc, holdout, train, value_col=metric_to_forecast
                                )
                                metrics_dict[name] = metrics

                                with cols[i]:
                                    st.markdown(f"**{name}**")
                                    if "error" in metrics:
                                        st.error(metrics["error"])
                                    else:
                                        if "mape" in metrics and not np.isnan(metrics.get("mape", np.nan)):
                                            st.metric("MAPE", f"{metrics['mape']:.1f}%")
                                        if "mase" in metrics and not np.isnan(metrics.get("mase", np.nan)):
                                            st.metric("MASE", f"{metrics['mase']:.2f}")
                                        if "coverage_95" in metrics and not np.isnan(
                                                metrics.get("coverage_95", np.nan)):
                                            st.metric("Coverage 95%", f"{metrics['coverage_95']:.0f}%")

                            # Ensemble metrics
                            ensemble_metrics = forecaster.evaluate_forecast(
                                ensemble, holdout, train, value_col=metric_to_forecast
                            )
                            metrics_dict["Ensemble"] = ensemble_metrics

                            with cols[-1]:
                                st.markdown("**🎯 Ensemble**")
                                if "error" in ensemble_metrics:
                                    st.error(ensemble_metrics["error"])
                                else:
                                    if "mape" in ensemble_metrics and not np.isnan(
                                            ensemble_metrics.get("mape", np.nan)):
                                        st.metric("MAPE", f"{ensemble_metrics['mape']:.1f}%")
                                    if "mase" in ensemble_metrics and not np.isnan(
                                            ensemble_metrics.get("mase", np.nan)):
                                        st.metric("MASE", f"{ensemble_metrics['mase']:.2f}")
                                    if "coverage_95" in ensemble_metrics and not np.isnan(
                                            ensemble_metrics.get("coverage_95", np.nan)):
                                        st.metric("Coverage 95%", f"{ensemble_metrics['coverage_95']:.0f}%")

                        # Visualization
                        st.subheader("📈 Визуализация прогноза")

                        fig = go.Figure()

                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=train["date"],
                            y=train[metric_to_forecast],
                            mode="lines",
                            name="История",
                            line=dict(color="black", width=2)
                        ))

                        # Holdout if available
                        if len(holdout) > 0:
                            fig.add_trace(go.Scatter(
                                x=holdout["date"],
                                y=holdout[metric_to_forecast],
                                mode="markers",
                                name="Holdout (факт)",
                                marker=dict(color="red", size=8)
                            ))

                        # Ensemble forecast
                        fig.add_trace(go.Scatter(
                            x=ensemble["date"],
                            y=ensemble["yhat"],
                            mode="lines+markers",
                            name="Прогноз",
                            line=dict(color="blue", width=2)
                        ))

                        fig.update_layout(
                            title=f"Прогноз: {metric_to_forecast}",
                            xaxis_title="Дата",
                            yaxis_title={
                                "volume": "Объём (UZS)",
                                "transactions": "Количество",
                                "unique_cards": "Карты"
                            }[metric_to_forecast],
                            height=500,
                            hovermode="x unified"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Export
                        csv = ensemble.to_csv(index=False)
                        st.download_button(
                            "📥 Скачать прогноз (CSV)",
                            data=csv,
                            file_name=f"forecast_{metric_to_forecast}_{datetime.now():%Y%m%d}.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Ошибка прогнозирования: {e}")
                    import traceback

                    with st.expander("🐛 Детали ошибки"):
                        st.code(traceback.format_exc())
            else:
                st.error("Enhanced Forecasting модуль не установлен. Выполните: pip install statsmodels prophet")

    elif forecast_mode == "📬 Диагностика рядов":
        st.subheader("Диагностика временных рядов")

        if HAS_ENHANCED_FORECAST:
            try:
                # Get max date from table
                max_date = get_max_date(selected_table)
                if not max_date:
                    st.error("Не удалось определить максимальную дату в таблице")
                    st.stop()

                # Format dates properly for ClickHouse
                start_date = (max_date - timedelta(days=90)).date()
                end_date = max_date.date()

                # Load time series
                query = f"""
                    SELECT 
                        toDate(transaction_date) AS date,
                        sum(amount_uzs) AS volume,
                        count() AS transactions
                    FROM {selected_table}
                    WHERE toDate(transaction_date) BETWEEN toDate('{start_date}') AND toDate('{end_date}')
                    GROUP BY date
                    ORDER BY date
                    """
                df = ch_df(query, ["date", "volume", "transactions"])

                if not df.empty:
                    # Ensure date is datetime
                    df["date"] = pd.to_datetime(df["date"])

                    # Select series to diagnose
                    series_name = st.selectbox("Выберите ряд", ["volume", "transactions"])

                    # Initialize diagnostics
                    diagnostics = TimeSeriesDiagnostics()

                    # STL decomposition
                    with st.spinner("Выполняю STL декомпозицию..."):
                        ts = pd.Series(
                            df[series_name].values,
                            index=pd.DatetimeIndex(df["date"])
                        )

                        stl_result = diagnostics.stl_decompose(ts)

                    if stl_result:
                        # Display decomposition
                        st.subheader("📊 STL Декомпозиция")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Сила тренда",
                                f"{stl_result.get('strength_trend', 0):.2f}",
                                help="Близко к 1 = сильный тренд"
                            )
                        with col2:
                            st.metric(
                                "Сила сезонности",
                                f"{stl_result.get('strength_seasonal', 0):.2f}",
                                help="Близко к 1 = сильная сезонность"
                            )

                        # Plot components
                        fig = go.Figure()

                        components = ["trend", "seasonal", "residual"]
                        for i, comp in enumerate(components):
                            if comp in stl_result:
                                fig.add_trace(go.Scatter(
                                    x=df["date"],
                                    y=stl_result[comp],
                                    mode="lines",
                                    name=comp.capitalize()
                                ))

                        fig.update_layout(
                            title="Компоненты временного ряда",
                            xaxis_title="Дата",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Anomaly detection
                    st.subheader("🔍 Обнаружение аномалий")

                    try:
                        anomalies = diagnostics.detect_anomalies(ts, threshold=2.5)
                        anomaly_dates = df.loc[anomalies, "date"].tolist()

                        if anomaly_dates:
                            st.warning(f"Обнаружено {len(anomaly_dates)} аномалий")

                            # Plot with anomalies
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=df["date"],
                                y=df[series_name],
                                mode="lines",
                                name="Данные"
                            ))

                            fig.add_trace(go.Scatter(
                                x=df.loc[anomalies, "date"],
                                y=df.loc[anomalies, series_name],
                                mode="markers",
                                name="Аномалии",
                                marker=dict(color="red", size=10)
                            ))

                            fig.update_layout(
                                title=f"Аномалии в {series_name}",
                                xaxis_title="Дата",
                                height=350
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # List anomaly dates
                            with st.expander("Даты аномалий"):
                                for d in anomaly_dates:
                                    st.write(f"• {d}")
                        else:
                            st.success("Аномалии не обнаружены")

                    except Exception as e:
                        st.warning(f"Не удалось обнаружить аномалии: {e}")

                    # Autocorrelation test
                    st.subheader("📈 Тест автокорреляции")

                    # Calculate residuals (simple detrending)
                    residuals = ts - ts.rolling(window=7, center=True).mean()
                    residuals = residuals.dropna()

                    ljung_result = diagnostics.ljung_box_test(residuals, lags=14)

                    if ljung_result:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Ljung-Box статистика",
                                f"{ljung_result.get('statistic', 0):.2f}"
                            )
                        with col2:
                            st.metric(
                                "P-value",
                                f"{ljung_result.get('p_value', 1):.4f}",
                                help="< 0.05 означает наличие автокорреляции"
                            )

                        if ljung_result.get("autocorrelated"):
                            st.warning("⚠️ Обнаружена автокорреляция в остатках")
                        else:
                            st.success("✅ Автокорреляция не обнаружена")


            except Exception as e:
                st.error(f"Ошибка диагностики: {e}")
                import traceback
                with st.expander("🐛 Детали ошибки"):
                    st.code(traceback.format_exc())
        else:
            st.error("Enhanced Forecasting модуль не установлен")

    elif forecast_mode == "📊 Валидация моделей":
        st.subheader("Rolling Origin Cross-Validation")

        if HAS_ENHANCED_FORECAST:
            col1, col2, col3 = st.columns(3)
            with col1:
                n_splits = st.number_input("Количество фолдов", 2, 10, 3)
            with col2:
                test_size = st.number_input("Размер теста (дней)", 7, 30, 14)
            with col3:
                metric_name = st.selectbox("Метрика", ["volume", "transactions"])

            if st.button("🔄 Запустить валидацию", type="primary"):
                try:
                    with st.spinner("Загрузка данных..."):
                        # Get max date
                        max_date = get_max_date(selected_table)
                        if not max_date:
                            st.error("Не удалось определить максимальную дату в таблице")
                            st.stop()

                        # Format dates properly
                        start_date = (max_date - timedelta(days=180)).date()
                        end_date = max_date.date()

                        query = f"""
                                    SELECT 
                                        toDate(transaction_date) AS date,
                                        {"sum(amount_uzs)" if metric_name == "volume" else "count()"} AS value
                                    FROM {selected_table}
                                    WHERE toDate(transaction_date) BETWEEN toDate('{start_date}') AND toDate('{end_date}')
                                    GROUP BY date
                                    ORDER BY date
                                    """
                        df = ch_df(query, ["date", "value"])

                        if df.empty:
                            st.warning(f"Нет данных за период {start_date} — {end_date}")
                            st.stop()

                        df["date"] = pd.to_datetime(df["date"])
                        df.rename(columns={"value": metric_name}, inplace=True)

                    if not df.empty:
                        # Initialize forecaster
                        forecaster = EnhancedForecaster()

                        with st.spinner(f"Валидация на {n_splits} фолдах..."):
                            cv_results = forecaster.rolling_origin_validation(
                                df,
                                n_splits=n_splits,
                                test_size=test_size,
                                value_col=metric_name
                            )

                        if not cv_results.empty:
                            # Summary statistics
                            st.subheader("📊 Результаты кросс-валидации")

                            summary = cv_results.groupby("model")[["mape", "smape", "mase", "rmse"]].agg(
                                ["mean", "std"])

                            # Display metrics
                            for model in summary.index:
                                with st.expander(f"Модель: {model}"):
                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        mean_mape = summary.loc[model, ("mape", "mean")]
                                        std_mape = summary.loc[model, ("mape", "std")]
                                        st.metric(
                                            "MAPE",
                                            f"{mean_mape:.1f}%",
                                            f"±{std_mape:.1f}%"
                                        )

                                    with col2:
                                        mean_smape = summary.loc[model, ("smape", "mean")]
                                        std_smape = summary.loc[model, ("smape", "std")]
                                        st.metric(
                                            "sMAPE",
                                            f"{mean_smape:.1f}%",
                                            f"±{std_smape:.1f}%"
                                        )

                                    with col3:
                                        mean_mase = summary.loc[model, ("mase", "mean")]
                                        std_mase = summary.loc[model, ("mase", "std")]
                                        st.metric(
                                            "MASE",
                                            f"{mean_mase:.2f}",
                                            f"±{std_mase:.2f}"
                                        )

                                    with col4:
                                        mean_rmse = summary.loc[model, ("rmse", "mean")]
                                        st.metric(
                                            "RMSE",
                                            f"{mean_rmse:,.0f}"
                                        )

                            # Box plot of metrics
                            fig = go.Figure()

                            for model in cv_results["model"].unique():
                                model_data = cv_results[cv_results["model"] == model]
                                fig.add_trace(go.Box(
                                    y=model_data["mape"],
                                    name=model,
                                    boxmean=True
                                ))

                            fig.update_layout(
                                title="Распределение MAPE по фолдам",
                                yaxis_title="MAPE (%)",
                                height=400
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Download results
                            csv = cv_results.to_csv(index=False)
                            st.download_button(
                                "📥 Скачать результаты валидации",
                                data=csv,
                                file_name=f"cv_results_{metric_name}_{datetime.now():%Y%m%d}.csv",
                                mime="text/csv"
                            )


                except Exception as e:
                    st.error(f"Ошибка валидации: {e}")
                    import traceback
                    with st.expander("🐛 Детали ошибки"):
                        st.code(traceback.format_exc())
        else:
            st.error("Enhanced Forecasting модуль не установлен")

    else:  # Сценарный анализ
        st.subheader("🎯 Сценарный анализ")
        st.info("Моделирование влияния изменений на прогноз")

        # Scenario parameters
        col1, col2 = st.columns(2)

        with col1:
            scenario_type = st.selectbox(
                "Тип сценария",
                ["Изменение тренда", "Сезонный шок", "Разовое событие"]
            )

        with col2:
            impact_percent = st.slider(
                "Величина воздействия (%)",
                -50, 50, 0, step=5
            )

        if scenario_type == "Изменение тренда":
            st.markdown("""
            **Изменение тренда** - постепенное изменение базового уровня
            - Положительное значение = рост
            - Отрицательное значение = спад
            """)
        elif scenario_type == "Сезонный шок":
            st.markdown("""
            **Сезонный шок** - изменение сезонной компоненты
            - Влияет на недельную периодичность
            - Может моделировать праздники/акции
            """)
        else:
            event_date = st.date_input("Дата события")
            event_duration = st.slider("Длительность (дней)", 1, 14, 3)

        if st.button("📊 Построить сценарий"):
            st.info("""
                🚧 Функция в разработке

                В следующей версии будет доступно:
                - Моделирование изменения тренда
                - Сезонные шоки
                - Анализ разовых событий
                - What-if анализ

                Пока вы можете использовать режим "Простой прогноз" для базового прогнозирования.
                """)

# ===================================
# TAB 8: Hierarchical Forecast
# ===================================
with tabHier:
    st.header("🏗 Hierarchical Forecast (из файлов)")
    merch_file = Path(find_artifact("forecast_hier_merchant.csv"))
    mcc_file = Path(find_artifact("forecast_hier_mcc.csv"))
    total_file = Path(find_artifact("forecast_hier_total.csv"))
    if total_file.exists():
        df_total = pd.read_csv(total_file, parse_dates=["ds"])
        st.plotly_chart(px.line(df_total, x="ds", y="yhat", title="TOTAL (Bottom-Up)"), use_container_width=True)
    if mcc_file.exists():
        df_mcc = pd.read_csv(mcc_file, parse_dates=["ds"])
        mccs = sorted(df_mcc["mcc"].unique().tolist())
        sl = st.selectbox("MCC", mccs, index=0)
        st.plotly_chart(px.line(df_mcc[df_mcc["mcc"] == sl], x="ds", y="yhat", title=f"MCC {sl} (BU)"),
                        use_container_width=True)

# ===================================
# TAB 9: Enhanced Weekly Report / RAG
# ===================================
with tabReport:
    st.header("📝 Enhanced Weekly Report / RAG")

    # Tabs for different modes
    rag_mode = st.radio(
        "Режим работы",
        ["📊 Генерация отчёта", "🤖 RAG-ассистент", "📈 Анализ трендов"],
        horizontal=True
    )

    if rag_mode == "📊 Генерация отчёта":
        with st.expander("ℹ️ Как работает Weekly Report?", expanded=False):
            st.markdown("""
            ### 📊 Weekly Report
            - Генерирует отчёт за выбранный период (по умолчанию 7 дней)
            - Включает: KPI vs предыдущей недели, топ MCC/мерчантов, сезонность, качество данных
            - Рассчитывает мини-PSI для оценки дрейфа данных
            """)

        # Get max date from selected table
        try:
            if has_col(selected_table, "transaction_date"):
                maxd = scalar(f"SELECT max(transaction_date) FROM {selected_table}")
            else:
                maxd = date.today()
        except:
            maxd = date.today()

        end_default = pd.to_datetime(maxd).date() if maxd else date.today()

        col1, col2 = st.columns([1, 3])
        with col1:
            end_date = st.date_input("📅 Конечная дата", value=end_default)
            days = st.number_input("📆 Период (дней)", min_value=1, max_value=30, value=7, step=1)
        with col2:
            st.info(f"""
            📊 Будет сгенерирован отчёт за период: **[{end_date - timedelta(days=days - 1)} — {end_date}]**

            Источник данных: `{selected_table}`
            """)

        if st.button("🚀 Сформировать отчёт", type="primary", use_container_width=True):
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [sys.executable, str(PROJECT_ROOT / "ml" / "generate_weekly_report.py"),
                   "--days", str(int(days)),
                   "--end", str(end_date),
                   "--table", selected_table]  # Pass selected table
            try:
                with st.status("Генерируем недельный отчёт...", expanded=False) as status:
                    out = subprocess.check_output(cmd, cwd=str(PROJECT_ROOT), text=True,
                                                  stderr=subprocess.STDOUT, env=env)
                    status.update(label="✅ Отчёт готов!", state="complete")
                    with st.expander("📋 Лог выполнения"):
                        st.code(out)
            except subprocess.CalledProcessError as e:
                st.error("⌠Ошибка генерации отчёта")
                st.code(e.output or str(e))

        # Show latest report
        md_files = sorted(REPORTS_DIR.glob("weekly_report_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if md_files:
            latest = md_files[0]
            st.success(f"📄 Последний отчёт: **{latest.name}**")
            text = latest.read_text(encoding="utf-8", errors="ignore")

            with st.expander("📊 Просмотр отчёта", expanded=True):
                st.markdown(text)

            st.download_button(
                "⬇️ Скачать отчёт (Markdown)",
                data=text.encode("utf-8"),
                file_name=latest.name,
                mime="text/markdown"
            )
        else:
            st.info("🔭 Отчётов пока нет. Нажмите кнопку выше для генерации.")

    elif rag_mode == "🤖 RAG-ассистент":
        st.subheader("🤖 Enhanced RAG Assistant with Citations")

        with st.expander("ℹ️ Как работает Enhanced RAG?", expanded=False):
            st.markdown("""
            ### 🔍 Улучшенный RAG с семантическим поиском

            **Источники контекста:**
            - 📊 Последние 4 недельных отчёта
            - 📚 Data Dictionary
            - 📝 DQ Reports
            - 🎯 Модельные артефакты (thresholds, feature importance)
            - 📈 MLflow метрики
            - 📉 Тренды за 4 недели

            **Возможности:**
            - ✨ TF-IDF поиск релевантных фрагментов
            - 📌 Цитирование источников с указанием секций
            - 📋 Структурированные ответы (insights/risks/actions)
            - 📢 SQL-запросы при необходимости
            - 💾 Кэширование для быстрого поиска
            """)

        # Settings
        col1, col2, col3 = st.columns(3)
        with col1:
            weeks_to_load = st.selectbox("📅 Недель для анализа", [2, 4, 8, 12], index=1)
        with col2:
            chunks_to_retrieve = st.selectbox("📄 Фрагментов контекста", [3, 5, 10, 15], index=1)
        with col3:
            response_style = st.selectbox(
                "💬 Стиль ответа",
                ["Краткий", "Подробный", "Технический"],
                index=0
            )

        # Suggested prompts
        st.markdown("**💡 Попробуйте эти вопросы:**")
        cols = st.columns(5)
        for i, prompt in enumerate(SUGGESTED_PROMPTS[:10]):
            col_idx = i % 5
            with cols[col_idx]:
                if st.button(prompt[:20] + "...", key=f"prompt_{i}", help=prompt, use_container_width=True):
                    st.session_state["rag_query"] = prompt

        # Initialize Enhanced RAG with caching
        if HAS_ENHANCED_RAG:
            rag = get_cached_rag(weeks_to_load)

            # Query input
            query = st.text_area(
                "💬 Ваш вопрос",
                value=st.session_state.get("rag_query", ""),
                placeholder="Например: Какая динамика P2P платежей за последний месяц? Есть ли риски по качеству данных?",
                height=100
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                search_button = st.button("🔍 Поиск", type="primary", use_container_width=True)
            with col2:
                include_trends = st.checkbox("📈 Включить анализ трендов", value=True)
                safe_sql_exec = st.checkbox("🔒 Разрешить выполнение SQL (только SELECT)", value=True)

            if search_button and query:
                with st.spinner("Анализирую контекст..."):
                    # Get trend summary if requested
                    trend_summary = ""
                    if include_trends:
                        try:
                            trend_summary = rag.get_trend_summary(selected_table)
                        except Exception as e:
                            st.warning(f"Не удалось получить тренды: {e}")

                    # Retrieve relevant chunks
                    chunks = rag.retrieve_relevant_chunks(query, top_k=chunks_to_retrieve)

                    if not chunks:
                        st.warning("Не найдено релевантных фрагментов. Попробуйте изменить запрос.")
                    else:
                        # Display retrieved sources with citations
                        with st.expander(f"📚 Найдено {len(chunks)} релевантных источников", expanded=False):
                            for i, (chunk_text, source) in enumerate(chunks):
                                st.markdown(f'<div class="source-box">', unsafe_allow_html=True)
                                st.caption(f"**Источник {i + 1}:** {source['doc_id']} ({source['type']}) "
                                           f'<span class="citation">[{i + 1}]</span>')
                                st.text(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)
                                st.markdown('</div>', unsafe_allow_html=True)

                        # Create enhanced prompt
                        prompt = create_enhanced_prompt(query, chunks, trend_summary)

                        # Call Claude API if available
                        ANTHROPIC_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
                        if ANTHROPIC_KEY:
                            try:
                                from anthropic import Anthropic

                                client_llm = Anthropic(api_key=ANTHROPIC_KEY)

                                # Adjust system prompt based on style
                                system_prompts = {
                                    "Краткий": "Отвечай кратко, по существу. Максимум 3-4 предложения на пункт.",
                                    "Подробный": "Дай развёрнутый ответ с примерами и пояснениями.",
                                    "Технический": "Используй технические термины, включи метрики и SQL где уместно."
                                }

                                with st.spinner("Генерирую ответ..."):
                                    response = client_llm.messages.create(
                                        model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
                                        max_tokens=1500,
                                        temperature=0,
                                        system=system_prompts.get(response_style, system_prompts["Краткий"]),
                                        messages=[{"role": "user", "content": prompt}]
                                    )

                                    raw_answer = response.content[0].text if response and response.content else ""

                                if raw_answer:
                                    # Format structured response
                                    structured = rag.format_response(
                                        raw_answer,
                                        [c[1] for c in chunks],
                                        include_sql=True
                                    )

                                    # Display structured response
                                    st.markdown("### 💡 Ответ")

                                    # Main summary with inline citations
                                    answer_with_citations = raw_answer
                                    for i in range(len(chunks)):
                                        # Add citation marks
                                        answer_with_citations = answer_with_citations.replace(
                                            f"источник {i + 1}",
                                            f'источник {i + 1} <span class="citation">[{i + 1}]</span>'
                                        )
                                    st.markdown(answer_with_citations, unsafe_allow_html=True)

                                    # Structured insights if extracted
                                    st.divider()
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        if structured["insights"]:
                                            st.markdown("#### ✨ Ключевые выводы")
                                            for insight in structured["insights"]:
                                                st.markdown(f"• {insight}")

                                    with col2:
                                        if structured["risks"]:
                                            st.markdown("#### ⚠️ Риски")
                                            for risk in structured["risks"]:
                                                st.markdown(f"• {risk}")

                                    with col3:
                                        if structured["actions"]:
                                            st.markdown("#### 📋 Рекомендации")
                                            for action in structured["actions"]:
                                                st.markdown(f"• {action}")

                                    # SQL query if present
                                    if structured.get("sql_query"):
                                        st.divider()
                                        st.markdown("#### 📢 SQL запрос")
                                        st.code(structured["sql_query"], language="sql")

                                        if safe_sql_exec and st.button("▶️ Выполнить запрос", key="exec_sql"):
                                            try:
                                                # Safe execution - only SELECT with LIMIT
                                                sql = structured["sql_query"].strip()
                                                if sql.upper().startswith("SELECT"):
                                                    # Add LIMIT if not present
                                                    if "LIMIT" not in sql.upper():
                                                        sql += " LIMIT 200"
                                                    result = ch_df(sql)
                                                    st.dataframe(result, use_container_width=True)

                                                    # Export option
                                                    csv = result.to_csv(index=False)
                                                    st.download_button(
                                                        "📥 Скачать результат",
                                                        data=csv,
                                                        file_name=f"query_result_{datetime.now():%Y%m%d_%H%M%S}.csv",
                                                        mime="text/csv"
                                                    )
                                                else:
                                                    st.error("Можно выполнять только SELECT запросы")
                                            except Exception as e:
                                                st.error(f"Ошибка выполнения: {e}")

                                    # Citations footer
                                    with st.expander("📌 Источники и цитаты"):
                                        for i, (_, source) in enumerate(chunks):
                                            st.caption(f'**[{i + 1}]** {source["doc_id"]} - '
                                                       f'{source.get("path", "N/A")[:50]}...')

                            except Exception as e:
                                st.error(f"Ошибка генерации ответа: {e}")
                        else:
                            st.warning("Не настроен CLAUDE_API_KEY/ANTHROPIC_API_KEY")

                            # Show prompt for manual review
                            with st.expander("📝 Сгенерированный промпт (для ручной проверки)"):
                                st.text_area("Prompt", prompt, height=400)
        else:
            st.error("Enhanced RAG модуль не установлен. Выполните: pip install scikit-learn")

    else:  # Анализ трендов
        st.subheader("📈 Анализ трендов")

        # First, check available date range in the table
        if has_col(selected_table, "transaction_date"):
            try:
                date_range_query = f"""
                SELECT 
                    min(transaction_date) as min_date,
                    max(transaction_date) as max_date,
                    count() as total_records
                FROM {selected_table}
                WHERE transaction_date IS NOT NULL
                """
                date_range = ch_rows(date_range_query)[0]
                min_date, max_date, total_records = date_range

                if min_date and max_date:
                    st.info(f"""
                    📅 **Доступный диапазон данных:**
                    - Начало: {min_date}
                    - Конец: {max_date}
                    - Всего записей: {total_records:,}
                    """)
                else:
                    st.warning("В таблице нет данных с заполненным transaction_date")
            except Exception as e:
                st.error(f"Ошибка получения диапазона дат: {e}")
        else:
            st.error("В таблице отсутствует колонка transaction_date")
            st.stop()

        # Trend analysis settings
        col1, col2, col3 = st.columns(3)

        with col1:
            analysis_mode = st.radio(
                "Режим анализа",
                ["Последние N недель", "Выбрать период"],
                index=0
            )

        with col2:
            if analysis_mode == "Последние N недель":
                trend_weeks = st.selectbox("Период анализа (недель)", [4, 8, 12, 16, 24], index=0)
                # Calculate dates based on max_date in table, not today()
                if max_date:
                    end_date = max_date
                    start_date = max_date - timedelta(weeks=trend_weeks)
                else:
                    end_date = date.today()
                    start_date = date.today() - timedelta(weeks=trend_weeks)
            else:
                # Manual date selection
                start_date = st.date_input(
                    "Начало периода",
                    value=max_date - timedelta(weeks=4) if max_date else date.today() - timedelta(weeks=4),
                    min_value=min_date if min_date else None,
                    max_value=max_date if max_date else None
                )
                end_date = st.date_input(
                    "Конец периода",
                    value=max_date if max_date else date.today(),
                    min_value=min_date if min_date else None,
                    max_value=max_date if max_date else None
                )

        with col3:
            comparison_type = st.selectbox(
                "Тип сравнения",
                ["Week-over-Week", "vs Медиана", "vs Среднее"],
                index=0
            )

        # Show what period will be analyzed
        st.caption(f"📊 Будет проанализирован период: **{start_date} — {end_date}**")

        if st.button("📊 Построить тренды", type="primary"):
            try:
                # Build query based on available columns
                query_parts = [
                    "toMonday(transaction_date) AS week",
                    f"{volume_expr(numeric)} AS volume",
                    "count() AS transactions",
                    f"{'avg(amount_uzs)' if numeric else 'avg(coalesce(toFloat64OrNull(toString(amount_uzs)), 0))'} AS avg_check"
                ]

                if has_col(selected_table, "hpan"):
                    query_parts.append("uniq(hpan) AS unique_cards")
                else:
                    query_parts.append("0 AS unique_cards")

                if has_col(selected_table, "p2p_flag"):
                    query_parts.append("round(sumIf(1, p2p_flag = 1) * 100.0 / count(), 2) AS p2p_share")
                else:
                    query_parts.append("0 AS p2p_share")

                query = f"""
                SELECT {', '.join(query_parts)}
                FROM {selected_table}
                WHERE transaction_date BETWEEN '{start_date}' AND '{end_date}'
                  AND transaction_date IS NOT NULL
                GROUP BY week
                ORDER BY week
                """

                # Debug info
                with st.expander("🔍 SQL запрос для отладки"):
                    st.code(query, language="sql")

                # Execute query
                trend_df = ch_df(query)

                if trend_df.empty:
                    st.warning(f"""
                    ⚠️ Нет данных за период {start_date} — {end_date}

                    Возможные причины:
                    1. В таблице нет данных за этот период
                    2. Данные есть, но transaction_date не заполнен

                    Попробуйте:
                    - Выбрать другой период
                    - Проверить данные в таблице
                    """)
                else:
                    # Set column names based on actual columns returned
                    expected_cols = ["week", "volume", "transactions", "avg_check", "unique_cards", "p2p_share"]
                    trend_df.columns = expected_cols[:len(trend_df.columns)]

                    st.success(f"✅ Найдено {len(trend_df)} недель данных")

                    # Ensure numeric types
                    for col in ["volume", "transactions", "avg_check", "unique_cards", "p2p_share"]:
                        if col in trend_df.columns:
                            trend_df[col] = pd.to_numeric(trend_df[col], errors='coerce').fillna(0)

                    # Calculate changes
                    for col in ["volume", "transactions", "avg_check", "unique_cards"]:
                        if col in trend_df.columns:
                            if comparison_type == "Week-over-Week":
                                trend_df[f"{col}_change"] = trend_df[col].pct_change() * 100
                            elif comparison_type == "vs Медиана":
                                median = trend_df[col].median()
                                if median != 0:
                                    trend_df[f"{col}_change"] = (trend_df[col] - median) / median * 100
                                else:
                                    trend_df[f"{col}_change"] = 0
                            else:  # vs Среднее
                                mean = trend_df[col].mean()
                                if mean != 0:
                                    trend_df[f"{col}_change"] = (trend_df[col] - mean) / mean * 100
                                else:
                                    trend_df[f"{col}_change"] = 0

                    # Display metrics
                    st.markdown("### 📊 Недельные метрики")

                    # Format for display
                    display_df = trend_df.copy()
                    display_df["week"] = pd.to_datetime(display_df["week"]).dt.strftime("%Y-%m-%d")
                    display_df["volume"] = display_df["volume"].map(lambda x: f"{x:,.0f}")
                    display_df["transactions"] = display_df["transactions"].map(lambda x: f"{int(x):,}")
                    display_df["avg_check"] = display_df["avg_check"].map(lambda x: f"{x:,.0f}")

                    if "unique_cards" in display_df.columns:
                        display_df["unique_cards"] = display_df["unique_cards"].map(lambda x: f"{int(x):,}")

                    # Add change indicators
                    for col in ["volume", "transactions", "avg_check", "unique_cards"]:
                        if f"{col}_change" in trend_df.columns:
                            display_df[f"{col}_Δ"] = trend_df[f"{col}_change"].map(
                                lambda x: f"{'🔴' if x < -5 else '🟡' if abs(x) < 5 else '🟢'} {x:+.1f}%" if pd.notna(
                                    x) else "-"
                            )

                    # Select columns to display
                    display_cols = ["week", "volume", "volume_Δ", "transactions", "transactions_Δ",
                                    "avg_check", "avg_check_Δ"]
                    if "unique_cards" in display_df.columns and trend_df["unique_cards"].sum() > 0:
                        display_cols.extend(["unique_cards", "unique_cards_Δ"])

                    # Filter existing columns
                    display_cols = [c for c in display_cols if c in display_df.columns]

                    st.dataframe(
                        display_df[display_cols],
                        use_container_width=True,
                        hide_index=True
                    )

                    # Visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=trend_df["week"],
                            y=trend_df["volume"],
                            mode="lines+markers",
                            name="Объём",
                            line=dict(color="blue", width=2)
                        ))
                        fig.update_layout(
                            title="Динамика объёма транзакций",
                            xaxis_title="Неделя",
                            yaxis_title="Объём (UZS)",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        if "volume_change" in trend_df.columns:
                            # Create color list based on values
                            colors = ['red' if x < 0 else 'green' for x in trend_df["volume_change"]]

                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=trend_df["week"],
                                y=trend_df["volume_change"],
                                marker_color=colors,
                                name=f"Изменение ({comparison_type})"
                            ))
                            fig.update_layout(
                                title=f"Изменение объёма ({comparison_type})",
                                xaxis_title="Неделя",
                                yaxis_title="Изменение (%)",
                                height=350
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Risk indicators
                    st.markdown("### ⚠️ Индикаторы риска")

                    # Calculate volatility
                    vol_cv = trend_df["volume"].std() / trend_df["volume"].mean() * 100 if trend_df[
                                                                                               "volume"].mean() != 0 else 0
                    tx_cv = trend_df["transactions"].std() / trend_df["transactions"].mean() * 100 if trend_df[
                                                                                                          "transactions"].mean() != 0 else 0

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        risk_level = "🟢 Низкий" if vol_cv < 15 else "🟡 Средний" if vol_cv < 30 else "🔴 Высокий"
                        st.metric(
                            "Волатильность объёма",
                            f"{vol_cv:.1f}%",
                            risk_level
                        )

                    with col2:
                        risk_level = "🟢 Низкий" if tx_cv < 15 else "🟡 Средний" if tx_cv < 30 else "🔴 Высокий"
                        st.metric(
                            "Волатильность транзакций",
                            f"{tx_cv:.1f}%",
                            risk_level
                        )

                    with col3:
                        # Detect anomalous weeks
                        if trend_df["volume"].std() != 0:
                            z_scores = np.abs(
                                (trend_df["volume"] - trend_df["volume"].mean()) / trend_df["volume"].std())
                            anomalies = (z_scores > 2).sum()
                        else:
                            anomalies = 0

                        st.metric(
                            "Аномальных недель",
                            f"{anomalies} из {len(trend_df)}",
                            "⚠️" if anomalies > 0 else "✅"
                        )

                    # Add P2P share visualization if available
                    if has_col(selected_table, "p2p_flag") and "p2p_share" in trend_df.columns:
                        st.divider()
                        st.markdown("### 💸 P2P динамика")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=trend_df["week"],
                            y=trend_df["p2p_share"],
                            mode="lines+markers",
                            name="P2P доля (%)",
                            line=dict(color="purple", width=2)
                        ))
                        fig.update_layout(
                            title="Доля P2P платежей по неделям",
                            xaxis_title="Неделя",
                            yaxis_title="P2P доля (%)",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Export
                    csv = trend_df.to_csv(index=False)
                    st.download_button(
                        "📥 Скачать данные трендов",
                        data=csv,
                        file_name=f"trends_{selected_table}_{datetime.now():%Y%m%d}.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Ошибка анализа трендов: {e}")
                import traceback

                with st.expander("🐛 Детали ошибки"):
                    st.code(traceback.format_exc())

# Continue with remaining tabs (10-13)...
# [Code for tabs 10-13 remains the same as in the original]

# ===================================
# TAB 10: Транзакции
# ===================================
with tab6:
    st.header("💳 Анализ транзакций")
    st.caption(f"Источник: **{selected_table}**")
    try:
        df = pd.DataFrame(ch_rows(f"SELECT * FROM {selected_table} LIMIT 1000"))
        if not df.empty:
            cols = [c[0] for c in ch_rows(f"DESCRIBE {selected_table}")]
            df.columns = cols[:len(df.columns)]
            st.dataframe(df, use_container_width=True, height=480)
    except Exception as e:
        st.error(f"Transactions: {e}")

# ===================================
# TAB 11: Статус+DQ
# ===================================
with tab7:
    st.header("📋 Статус и качество данных")
    st.caption(f"Источник: **{selected_table}**")

    try:
        col1, col2 = st.columns(2)

        # Summary
        with col1:
            st.subheader("📊 Сводка по таблице")

            # Build query parts based on available columns
            query_parts = [
                "count() AS total",
                "uniq(hpan) AS cards" if has_col(selected_table, 'hpan') else "0 AS cards",
                "uniq(pinfl) AS clients" if has_col(selected_table, 'pinfl') else "0 AS clients",
                "uniq(merchant_name) AS merchants" if has_col(selected_table, 'merchant_name') else "0 AS merchants",
                "uniq(emitent_bank) AS banks" if has_col(selected_table, 'emitent_bank') else "0 AS banks",
                volume_expr(numeric) + " AS volume"
            ]

            q = f"SELECT {', '.join(query_parts)} FROM {selected_table}"
            res = ch_rows(q)[0]

            labels = ["Всего транзакций", "Уникальных карт", "Уникальных клиентов",
                      "Уникальных мерчантов", "Банков-эмитентов", "Общий объём (UZS)"]
            for lab, val in zip(labels, res):
                if isinstance(val, (int, float)):
                    st.metric(lab, f"{int(val):,}" if val != 0 else "—")
                else:
                    st.metric(lab, "—")

        # Data Quality
        with col2:
            st.subheader("✅ Качество и валидность")

            # Build quality check query
            h = detect_hour_expr(selected_table) or "toUInt32(0)"
            m = mcc_expr(selected_table) or "toUInt32(0)"

            if numeric:
                amount_check = "countIf(amount_uzs = 0 OR amount_uzs IS NULL)"
            else:
                amount_check = "countIf(coalesce(toFloat64OrNull(toString(amount_uzs)), 0) = 0)"

            q = f"""
            SELECT
                {amount_check} AS empty_amounts,
                countIf({m} = 0) AS bad_mcc,
                countIf({h} < 0 OR {h} > 23) AS bad_hour,
                count() AS total
            FROM {selected_table}
            """
            ea, emcc, ehour, tot = ch_rows(q)[0]

            st.progress(1 - ea / (tot or 1), text=f"Суммы заполнены: {(1 - ea / (tot or 1)) * 100:.1f}%")

            if has_col(selected_table, "mcc"):
                st.progress(1 - emcc / (tot or 1), text=f"MCC валидны: {(1 - emcc / (tot or 1)) * 100:.1f}%")

            if detect_hour_expr(selected_table):
                st.progress(1 - ehour / (tot or 1), text=f"Часы валидны: {(1 - ehour / (tot or 1)) * 100:.1f}%")

        st.divider()

        # Null rate analysis
        st.subheader("🔍 Топ-10 колонок по пустотам")
        cols = [c[0] for c in ch_rows(f"DESCRIBE {selected_table}")]
        checks = []
        for c in cols:
            qn = f"SELECT countIf({c} IS NULL OR toString({c}) = '') AS n, count() AS t FROM {selected_table}"
            try:
                n, t = ch_rows(qn)[0]
                checks.append((c, int(n), n / (t or 1)))
            except:
                pass

        if checks:
            df_null = pd.DataFrame(checks, columns=["column", "nulls", "null_rate"]).sort_values("null_rate",
                                                                                                 ascending=False).head(
                10)
            st.dataframe(df_null, use_container_width=True)

        # Quantiles
        if has_col(selected_table, "amount_uzs"):
            st.subheader("🔍 Квантили по сумме (amount_uzs)")

            if numeric:
                qa = f"""
                SELECT 
                    quantile(0.10)(amount_uzs) AS q10,
                    quantile(0.25)(amount_uzs) AS q25,
                    quantile(0.50)(amount_uzs) AS q50,
                    quantile(0.75)(amount_uzs) AS q75,
                    quantile(0.90)(amount_uzs) AS q90
                FROM {selected_table}
                WHERE amount_uzs > 0
                """
            else:
                qa = f"""
                SELECT 
                    quantile(0.10)(coalesce(toFloat64OrNull(toString(amount_uzs)), 0)) AS q10,
                    quantile(0.25)(coalesce(toFloat64OrNull(toString(amount_uzs)), 0)) AS q25,
                    quantile(0.50)(coalesce(toFloat64OrNull(toString(amount_uzs)), 0)) AS q50,
                    quantile(0.75)(coalesce(toFloat64OrNull(toString(amount_uzs)), 0)) AS q75,
                    quantile(0.90)(coalesce(toFloat64OrNull(toString(amount_uzs)), 0)) AS q90
                FROM {selected_table}
                WHERE coalesce(toFloat64OrNull(toString(amount_uzs)), 0) > 0
                """

            try:
                q10, q25, q50, q75, q90 = ch_rows(qa)[0]
                quantiles_df = pd.DataFrame([
                    ["P10", f"{q10:,.0f}"],
                    ["P25", f"{q25:,.0f}"],
                    ["P50 (Медиана)", f"{q50:,.0f}"],
                    ["P75", f"{q75:,.0f}"],
                    ["P90", f"{q90:,.0f}"]
                ], columns=["Квантиль", "Значение (UZS)"])
                st.table(quantiles_df)
            except Exception as e:
                st.error(f"Ошибка расчёта квантилей: {e}")

        # DQ check button
        st.divider()
        st.subheader("📬 Полный DQ-отчёт (Pandera)")

        col5, col6 = st.columns([2, 1])
        with col5:
            limit = st.number_input(
                "Размер выборки для проверки (LIMIT)",
                min_value=10000,
                max_value=2000000,
                value=200000,
                step=50000
            )
        with col6:
            st.info("Pandera выполнит детальную валидацию схемы данных")

        if st.button("▶️ Запустить DQ-проверку", type="primary", use_container_width=True):
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"  # Force UTF-8 encoding
            cmd = [sys.executable, str(PROJECT_ROOT / "dq_check.py"),
                   "--table", selected_table, "--limit", str(int(limit))]
            try:
                with st.status("Генерация DQ-отчёта...", expanded=False) as status:
                    # Use encoding="utf-8" for proper decoding
                    out = subprocess.check_output(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        text=True,
                        stderr=subprocess.STDOUT,
                        env=env,
                        encoding="utf-8",  # Specify UTF-8 encoding
                        errors="replace"  # Replace any bad characters
                    )
                    status.update(label="✅ DQ-отчёт готов!", state="complete")
                    with st.expander("📋 Лог выполнения"):
                        st.code(out)
            except subprocess.CalledProcessError as e:
                st.error("⌠ Ошибка запуска dq_check.py")
                # Decode output with UTF-8 if available
                error_output = e.output or str(e)
                if isinstance(error_output, bytes):
                    error_output = error_output.decode('utf-8', errors='replace')
                st.code(error_output)

        # Show latest DQ reports
        mds = sorted(REPORTS_DIR.glob(f"dq_report_{selected_table}_*.md"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        csvs = sorted(REPORTS_DIR.glob(f"dq_summary_{selected_table}_*.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)

        if mds or csvs:
            st.divider()
            st.subheader("📊 Последние DQ-отчёты")

            if mds:
                latest_md = mds[0]
                col7, col8 = st.columns([3, 1])
                with col7:
                    st.success(f"📄 Последний отчёт: **{latest_md.name}**")
                with col8:
                    st.download_button(
                        "⬇️ Скачать MD",
                        data=latest_md.read_bytes(),
                        file_name=latest_md.name,
                        mime="text/markdown"
                    )

                txt = latest_md.read_text(encoding="utf-8", errors="ignore")
                with st.expander("📖 Просмотр отчёта"):
                    st.markdown(txt[:2000] + ("... _(полный текст в файле)_" if len(txt) > 2000 else ""))

            if csvs:
                latest_csv = csvs[0]
                st.caption(f"📊 Сводка: {latest_csv.name}")
                df_sum = pd.read_csv(latest_csv)
                st.dataframe(df_sum.head(100), use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка Status+DQ: {e}")

# ===================================
# TAB 12: Data Dictionary (lazy)
# ===================================
with tabDict:
    st.header("📚 Data Dictionary (ленивый)")
    st.caption(f"Источник: **{selected_table}**")
    st.write("Формируется по кнопке (на больших таблицах — тяжёлый).")
    DICT_SAMPLE_LIMIT = st.number_input("Сэмпл строк (LIMIT)", 50_000, 2_000_000, 200_000, step=50_000)
    if st.button("📚 Сформировать словарь сейчас"):
        cols = [c[0] for c in ch_rows(f"DESCRIBE {selected_table}")]
        report_rows = []
        for name in cols:
            ex_rows = ch_rows(f"SELECT {name} FROM {selected_table} WHERE {name} IS NOT NULL LIMIT 1")
            ex = ex_rows[0][0] if ex_rows else None
            q = f"SELECT countIf({name} IS NULL OR toString({name})='') nulls, uniq({name}) uniques, count() total FROM (SELECT {name} FROM {selected_table} LIMIT {int(DICT_SAMPLE_LIMIT)})"
            try:
                n, u, t = ch_rows(q)[0]
                report_rows.append([name, ex, int(n), (n / (t or 1)), int(u)])
            except:
                pass
        dict_df = pd.DataFrame(report_rows, columns=["column", "example", "nulls", "null_rate", "uniques"]).sort_values(
            "null_rate", ascending=False)
        st.dataframe(dict_df, use_container_width=True)
        md_lines = ["# Data Dictionary — " + selected_table, ""]
        for _, r in dict_df.iterrows():
            md_lines.append(
                f"- **{r['column']}** | nulls: {int(r['nulls'])} ({r['null_rate']:.1%}) | uniques: {int(r['uniques'])} | example: `{str(r['example'])[:50]}`")
        md_text = "\n".join(md_lines)
        st.download_button("⬇️ CSV (Dictionary)", data=dict_df.to_csv(index=False).encode("utf-8"),
                           file_name="data_dictionary.csv")
        st.download_button("⬇️ Markdown (Dictionary)", data=md_text.encode("utf-8"),
                           file_name=f"data_dictionary_{selected_table}.md")

# ===================================
# TAB 13: NL Assistant
# ===================================
with tab8:
    st.header("😊 NL-ассистент (Claude)")
    st.caption(f"Источник по умолчанию: {selected_table}")
    import json as _json

    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    ANTHROPIC_KEY = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")


    def _safe_select(sql: str, limit: int = 200):
        s = sql.strip().lower()
        if not s.startswith("select") or any(x in s for x in
                                             (";", " insert ", " update ", " delete ", " drop ", " alter ", " rename ",
                                              " truncate ")):
            raise ValueError("Только SELECT без ';'")
        if " limit " not in s:
            sql = sql.strip() + f" LIMIT {limit}"
        return ch_rows(sql)


    def _profile_cols(table: str, top_k: int = 5):
        cols = [c[0] for c in ch_rows(f"DESCRIBE {table}")]
        prof = {}
        for c in cols:
            try:
                total, nulls, uniqs = \
                ch_rows(f"SELECT count(), countIf({c} IS NULL OR toString({c})=''), uniq({c}) FROM {table}")[0]
                ex = ch_rows(f"SELECT {c} FROM {table} WHERE {c} IS NOT NULL AND toString({c})!='' LIMIT {top_k}")
                ch_type = [t[1] for t in ch_rows(f"DESCRIBE {table}") if t[0] == c][0]
                prof[c] = {"type": ch_type, "total": int(total), "nulls": int(nulls),
                           "null_rate": (nulls / total if total else 0), "uniques": int(uniqs),
                           "examples": [r[0] for r in ex]}
            except Exception as e:
                prof[c] = {"error": str(e)}
        return prof


    if not ANTHROPIC_KEY:
        st.warning("Нет CLAUDE_API_KEY/ANTHROPIC_API_KEY в окружении.")
    else:
        try:
            from anthropic import Anthropic

            anthropic_client = Anthropic(api_key=ANTHROPIC_KEY)
        except Exception as e:
            anthropic_client = None
            st.error(f"Anthropic SDK error: {e}")

        if anthropic_client:
            if "nl_chat" not in st.session_state:
                st.session_state.nl_chat = []
            for role, text in st.session_state.nl_chat:
                with st.chat_message(role):
                    st.markdown(text)

            SYSTEM = ("Ты помогаешь работать с ClickHouse. Отвечай на языке пользователя. "
                      "Если нужен расчёт — верни JSON одной строкой: "
                      '{"action":"sql","sql":"SELECT ..."} | {"action":"profile","table":"..."} | {"action":"echo","text":"..."} '
                      "Только SELECT. Добавляй LIMIT 200.")

            user_q = st.chat_input("Например: «Покажи топ-10 MCC по сумме за прошлый месяц»")
            if user_q:
                st.session_state.nl_chat.append(("user", user_q))
                with st.chat_message("user"):
                    st.markdown(user_q)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    try:
                        resp = anthropic_client.messages.create(model=CLAUDE_MODEL, max_tokens=800, temperature=0,
                                                                system=SYSTEM,
                                                                messages=[{"role": "user", "content": user_q}])
                        text = resp.content[0].text if resp and resp.content else ""
                    except Exception as e:
                        text = f'{{"action":"echo","text":"Ошибка вызова модели: {e}"}}'
                    try:
                        block = _json.loads(text.strip())
                    except Exception:
                        placeholder.markdown(text)
                        st.session_state.nl_chat.append(("assistant", text))
                    else:
                        if block.get("action") == "sql":
                            sql = block.get("sql") or ""
                            try:
                                rows = _safe_select(sql)
                                df = pd.DataFrame(rows)
                                st.code(sql, language="sql")
                                st.dataframe(df, use_container_width=True)
                                placeholder.markdown("Готово ✅")
                                st.session_state.nl_chat.append(("assistant", f"```sql\n{sql}\n```\nГотово ✅"))
                            except Exception as e:
                                st.error(f"Ошибка SELECT: {e}")
                                st.session_state.nl_chat.append(("assistant", f"Ошибка SELECT: {e}"))
                        elif block.get("action") == "profile":
                            table = block.get("table") or selected_table
                            st.info(f"Профиль: **{table}**")
                            st.json(_profile_cols(table))
                            placeholder.markdown("Готово ✅")
                            st.session_state.nl_chat.append(("assistant", f"Профиль таблицы {table} показан ✅"))
                        elif block.get("action") == "echo":
                            msg = block.get("text", "")
                            placeholder.markdown(msg)
                            st.session_state.nl_chat.append(("assistant", msg))
                        else:
                            placeholder.markdown(text)
                            st.session_state.nl_chat.append(("assistant", text))

if __name__ == "__main__":
    st.write("Запустите:  streamlit run unified_dashboard.py")