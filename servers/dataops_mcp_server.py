#!/usr/bin/env python3
"""
servers/dataops_mcp_server.py — MCP server exposing safe DataOps tools
- Подхватывает .env (CLICKHOUSE_*, CLAUDE_API_KEY/ANTHROPIC_API_KEY)
- Аккуратные SELECT-only инструменты
- Гибкий запуск пайплайнов (импорт либо подпроцесс)
- --selftest для быстрой проверки связи
"""

import os, json, re, csv, sys, subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path

# ---- env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clickhouse_driver import Client
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from mcp.server import FastMCP  # type: ignore

APP_NAME = "dataops"
EXPORT_DIR = Path(os.environ.get("EXPORT_DIR", PROJECT_ROOT / "exports"))
EXPORT_DIR.mkdir(exist_ok=True, parents=True)


def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


def get_ch() -> Client:
    host = _env("CLICKHOUSE_HOST", "CH_HOST", default="localhost")
    port = int(_env("CLICKHOUSE_PORT", "CH_PORT", default="9000"))
    user = _env("CLICKHOUSE_USER", "CH_USER", default="analyst")
    password = _env("CLICKHOUSE_PASSWORD", "CH_PASSWORD", default="admin123")
    db = _env("CLICKHOUSE_DATABASE", "CH_DB", default="card_analytics")
    return Client(
        host=host, port=port, user=user, password=password, database=db,
        connect_timeout=3, send_receive_timeout=6, sync_request_timeout=6,
        settings={"max_execution_time": 8, "use_uncompressed_cache": 0}
    )


mcp = FastMCP(APP_NAME)


def _only_select(sql: str, default_limit: int = 1000) -> str:
    s = (sql or "").strip()
    s_low = f" {s.lower()} "
    if not s_low.strip().startswith("select"):
        raise ValueError("Only SELECT statements are allowed")
    banned = [" insert ", " update ", " delete ", " drop ", " alter ",
              " rename ", " truncate ", ";"]
    if any(b in s_low for b in banned):
        raise ValueError("Potentially unsafe SQL detected")
    if " limit " not in s_low:
        s += f" LIMIT {default_limit}"
    return s


@mcp.tool()
def list_tables(prefix: str = "transactions_") -> List[str]:
    """Return tables filtered by prefix."""
    rows = get_ch().execute("SHOW TABLES")
    return [r[0] for r in rows if not prefix or r[0].startswith(prefix)]


@mcp.tool()
def describe_table(table: str) -> Dict[str, str]:
    """Return {column: ch_type}."""
    rows = get_ch().execute(f"DESCRIBE {table}")
    return {r[0]: r[1] for r in rows}


@mcp.tool()
def sample_rows(table: str, limit: int = 100) -> Dict[str, Any]:
    """Return sample rows as {columns, rows}."""
    ch = get_ch()
    cols = [c[0] for c in ch.execute(f"DESCRIBE {table}")]
    data = ch.execute(f"SELECT * FROM {table} LIMIT {limit}")
    return {"columns": cols[:len(data[0])] if data else cols, "rows": data}


@mcp.tool()
def profile_table(table: str, top_k: int = 5) -> Dict[str, Any]:
    """Compute simple profile per column (nulls, uniques, examples)."""
    ch = get_ch()
    prof: Dict[str, Any] = {}
    cols = ch.execute(f"DESCRIBE {table}")
    for name, ch_type, *_ in cols:
        try:
            total, nulls, uniqs = ch.execute(
                f"SELECT count(), countIf({name} IS NULL OR toString({name}) = ''), uniq({name}) FROM {table}"
            )[0]
            examples = [r[0] for r in ch.execute(
                f"SELECT {name} FROM {table} WHERE {name} IS NOT NULL AND toString({name}) != '' LIMIT {top_k}"
            )]
            item = {
                "type": ch_type,
                "total": int(total),
                "nulls": int(nulls),
                "null_rate": (float(nulls) / float(total) if total else 0.0),
                "uniques": int(uniqs),
                "examples": examples
            }
            if any(x in ch_type for x in ("Int", "Float", "Decimal")):
                mn, mx, avg = ch.execute(
                    f"SELECT min(toFloat64OrNull(toString({name}))), "
                    f"max(toFloat64OrNull(toString({name}))), "
                    f"avg(toFloat64OrNull(toString({name}))) FROM {table}"
                )[0]
                item.update({"min": mn, "max": mx, "avg": avg})
            prof[name] = item
        except Exception as e:
            prof[name] = {"error": str(e)}
    return prof


@mcp.tool()
def sql_select(sql: str, limit: int = 1000) -> Dict[str, Any]:
    """Execute safe SELECT; return {columns, rows}."""
    safe = _only_select(sql, default_limit=limit)
    ch = get_ch()
    data = ch.execute(safe)
    columns = None
    try:
        low = safe.lower()
        if " from " in low:
            after = low.split(" from ", 1)[1].strip()
            tbl = after.split()[0].split(".")[-1]
            columns = [c[0] for c in ch.execute(f"DESCRIBE {tbl}")]
    except Exception:
        pass
    return {"columns": columns[:len(data[0])] if (columns and data) else None, "rows": data}


@mcp.tool()
def export_csv(sql: str, filename: str = "export.csv") -> str:
    """Execute safe SELECT and save to exports/<filename>; returns path."""
    safe = _only_select(sql)
    ch = get_ch()
    data = ch.execute(safe)
    path = EXPORT_DIR / Path(filename).name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for row in data:
            writer.writerow(row)
    return str(path)


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


def _run_subprocess(pyfile: Path) -> str:
    if not pyfile.exists():
        raise FileNotFoundError(f"Script not found: {pyfile}")
    cmd = [sys.executable, str(pyfile)]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed:\n{proc.stderr}")
    return proc.stdout.strip() or "ok"


@mcp.tool()
def run_pipeline(name: str) -> str:
    """Run one of: fraud|segmentation|forecast|shap|restore_dates|safe_load|analysis."""
    name = (name or "").strip().lower()

    if name == "fraud":
        mod = _try_import("fraud_detection")
        if mod and hasattr(mod, "main"):
            mod.main(); return "fraud: done"
        return _run_subprocess(PROJECT_ROOT / "fraud_detection.py")

    if name in ("segmentation", "segments"):
        mod = _try_import("customer_segmentation")
        if mod:
            if hasattr(mod, "CustomerSegmentation"):
                mod.CustomerSegmentation().run_full_pipeline(); return "segmentation: done"
            if hasattr(mod, "main"):
                mod.main(); return "segmentation: done"
        return _run_subprocess(PROJECT_ROOT / "customer_segmentation.py")

    if name in ("forecast", "prophet"):
        mod = _try_import("volume_forecasting")
        if mod:
            if hasattr(mod, "VolumeForecaster"):
                mod.VolumeForecaster().run_full_pipeline(); return "forecast: done"
            if hasattr(mod, "main"):
                mod.main(); return "forecast: done"
        return _run_subprocess(PROJECT_ROOT / "volume_forecasting.py")

    if name in ("shap", "explain"):
        mod = _try_import("shap_explainer")
        if mod and hasattr(mod, "main"):
            mod.main(); return "shap: done"
        return _run_subprocess(PROJECT_ROOT / "shap_explainer.py")

    if name in ("restore_dates", "restore"):
        mod = _try_import("restore_correct_dates")
        if mod:
            if hasattr(mod, "restore_correct_dates"):
                mod.restore_correct_dates(); return "restore_dates: done"
            if hasattr(mod, "main"):
                mod.main(); return "restore_dates: done"
        return _run_subprocess(PROJECT_ROOT / "restore_correct_dates.py")

    if name in ("safe_load", "load"):
        mod = _try_import("safe_load")
        if mod:
            if hasattr(mod, "safe_load_to_clickhouse"):
                mod.safe_load_to_clickhouse(); return "safe_load: done"
            if hasattr(mod, "main"):
                mod.main(); return "safe_load: done"
        return _run_subprocess(PROJECT_ROOT / "safe_load.py")

    if name in ("analysis", "queries"):
        mod = _try_import("run_analysis")
        if mod:
            if hasattr(mod, "run_analysis"):
                mod.run_analysis(); return "analysis: done"
            if hasattr(mod, "main"):
                mod.main(); return "analysis: done"
        return _run_subprocess(PROJECT_ROOT / "run_analysis.py")

    raise ValueError("Unknown pipeline. Use: fraud|segmentation|forecast|shap|restore_dates|safe_load|analysis")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        try:
            print("CH ping ->", get_ch().execute("SELECT 1"))
        except Exception as e:
            print("CH ping failed:", e)
        try:
            print("Tables (first 5):", list_tables()[:5])
        except Exception as e:
            print("list_tables failed:", e)
        sys.exit(0)

    print("MCP server started (stdio). Waiting for an MCP client (e.g., Claude)...")
    mcp.run()
