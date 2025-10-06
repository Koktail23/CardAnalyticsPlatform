#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fraud_graph_pipeline.py — оркестратор пайплайна фрода с граф-фичами:
1) build_graph_features.py  (генерация graph_* за окно)
2) fraud_detection.py       (обучение XGB с граф-фичами, Optuna, MLflow)
3) calibrate_thresholds.py  (Isotonic + пороги FPR≈1/2% и EV-оптимум, MLflow)
4) Итоговая сводка в консоль + сохранение в ml/fraud_summary.json

Запуск:
  python ml\\run_fraud_graph_pipeline.py --window_days 90 --n_trials 20 --with_mlflow
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


def parse_args():
    ap = argparse.ArgumentParser("run_fraud_graph_pipeline")
    ap.add_argument("--window_days", type=int, default=90, help="окно для graph_*")
    ap.add_argument("--n_trials", type=int, default=20, help="число проб Optuna")
    ap.add_argument("--with_mlflow", action="store_true", help="логировать в MLflow")
    ap.add_argument("--python", default=sys.executable, help="интерпретатор Python")
    ap.add_argument("--models_path", default=os.getenv("MODELS_PATH", "ml"))
    return ap.parse_args()


def run_cmd(cmd, cwd: Path, env=None):
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=env)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
        raise SystemExit(p.returncode)
    return p.stdout


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1] if (Path(__file__).name == "run_fraud_graph_pipeline.py") else Path.cwd()
    models_dir = Path(args.models_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # чтобы matplotlib/prophet/plotting не блокировали
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    # 1) GRAPH FEATURES
    out1 = run_cmd(
        [args.python, str(root / "ml" / "build_graph_features.py"), "--window_days", str(args.window_days)],
        cwd=root, env=env
    )

    # 2) TRAIN with graph features (XGB + Optuna)
    train_cmd = [args.python, str(root / "ml" / "fraud_detection.py"),
                 "--mlflow" if args.with_mlflow else "",
                 "--tune", "--n_trials", str(args.n_trials)]
    train_cmd = [c for c in train_cmd if c]  # убрать пустые элементы
    out2 = run_cmd(train_cmd, cwd=root, env=env)

    # 3) CALIBRATION (Isotonic + thresholds)
    calib_cmd = [args.python, str(root / "ml" / "calibrate_thresholds.py")]
    if args.with_mlflow:
        calib_cmd.append("--mlflow")
    out3 = run_cmd(calib_cmd, cwd=root, env=env)

    # 4) COLLECT SUMMARY
    thresholds_path = models_dir / "thresholds.json"
    metrics = {}
    try:
        # вытащим метрики из stdout calibrate_thresholds (там печатается JSON)
        j = json.loads(out3.splitlines()[0])
        metrics["brier_raw"] = j.get("brier_raw")
        metrics["brier_calibrated"] = j.get("brier_calibrated")
        metrics["thr_fpr1"] = j.get("thresholds", {}).get("fpr_1pct", {}).get("thr")
        metrics["recall_fpr1"] = j.get("thresholds", {}).get("fpr_1pct", {}).get("recall")
        metrics["thr_fpr2"] = j.get("thresholds", {}).get("fpr_2pct", {}).get("thr")
        metrics["recall_fpr2"] = j.get("thresholds", {}).get("fpr_2pct", {}).get("recall")
        ev = j.get("thresholds", {}).get("ev_optimal", {})
        metrics["thr_ev"] = ev.get("thr")
        metrics["ev_best"] = ev.get("ev")
    except Exception:
        # если что-то пойдёт не так — попробуем прочитать thresholds.json
        if thresholds_path.exists():
            j = json.loads(thresholds_path.read_text(encoding="utf-8"))
            metrics["brier_raw"] = j.get("brier_raw")
            metrics["brier_calibrated"] = j.get("brier_calibrated")
            metrics["thr_fpr1"] = j.get("thresholds", {}).get("fpr_1pct", {}).get("thr")
            metrics["recall_fpr1"] = j.get("thresholds", {}).get("fpr_1pct", {}).get("recall")
            metrics["thr_fpr2"] = j.get("thresholds", {}).get("fpr_2pct", {}).get("thr")
            metrics["recall_fpr2"] = j.get("thresholds", {}).get("fpr_2pct", {}).get("recall")
            ev = j.get("thresholds", {}).get("ev_optimal", {})
            metrics["thr_ev"] = ev.get("thr")
            metrics["ev_best"] = ev.get("ev")

    # сохраним краткую сводку в ml/fraud_summary.json
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "window_days": args.window_days,
        "n_trials": args.n_trials,
        "mlflow": bool(args.with_mlflow),
        "metrics": metrics
    }
    summary_path = models_dir / "fraud_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== FRAUD GRAPH PIPELINE SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("========================================")
    print(f"[OK] summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
