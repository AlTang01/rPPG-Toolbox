"""Enregistre les métriques d'un run pour comparaison entre méthodes."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = ROOT / "runs" / "comparison"
RESULTS_CSV = COMPARISON_DIR / "results.csv"

CSV_FIELDS = [
    "timestamp",
    "run_name",
    "toolbox_mode",
    "method",
    "roi",
    "dataset",
    "eval_method",
    "n_windows",
    "mae",
    "rmse",
    "mape",
    "pearson",
    "snr",
    "config_file",
    "cached_path",
    "results_dir",
    "model_file_name",
]


def _compute_metrics(gt_hr: np.ndarray, pred_hr: np.ndarray) -> dict:
    gt_hr = np.asarray(gt_hr, dtype=np.float64)
    pred_hr = np.asarray(pred_hr, dtype=np.float64)
    err = pred_hr - gt_hr
    pearson = float(np.corrcoef(gt_hr, pred_hr)[0, 1]) if len(gt_hr) > 1 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(err / gt_hr)) * 100)
    return {
        "n_windows": int(len(gt_hr)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "mape": mape,
        "pearson": pearson,
    }


def _resolve_config_file() -> str:
    raw = os.environ.get("RPPG_CONFIG_FILE", "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sanitize_run_name(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name.strip())
    return name[:200] if name else "unnamed_run"


def _ensure_unique_run_name(run_name: str) -> str:
    """Avoid overwriting runs/comparison/runs/<name>.json on rerun."""
    run_name = _sanitize_run_name(run_name)
    json_path = COMPARISON_DIR / "runs" / f"{run_name}.json"
    if not json_path.exists():
        return run_name
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return _sanitize_run_name(f"{run_name}_rerun_{suffix}")


def _paths_from_config(config) -> dict:
    if config.TOOLBOX_MODE == "unsupervised_method":
        data = config.UNSUPERVISED.DATA
        log_path = config.LOG.PATH
        exp_name = data.EXP_DATA_NAME
        cached = data.CACHED_PATH
        results_dir = getattr(
            config.UNSUPERVISED,
            "OUTPUT_SAVE_DIR",
            os.path.join(log_path, exp_name, "saved_outputs"),
        )
        model_file = ""
    elif config.TOOLBOX_MODE in ("train_and_test", "only_test"):
        data = config.TEST.DATA
        log_path = config.LOG.PATH
        exp_name = config.TRAIN.DATA.EXP_DATA_NAME
        cached = data.CACHED_PATH
        results_dir = config.TEST.OUTPUT_SAVE_DIR
        model_file = getattr(config.TRAIN, "MODEL_FILE_NAME", "")
    else:
        return {
            "cached_path": "",
            "results_dir": "",
            "model_file_name": "",
        }

    return {
        "cached_path": str(cached),
        "results_dir": str(results_dir),
        "model_file_name": str(model_file),
    }


def _migrate_csv_if_needed() -> None:
    if not RESULTS_CSV.exists():
        return
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        rows = list(reader)
    missing = [c for c in CSV_FIELDS if c not in old_fields]
    if not missing:
        return
    backup = RESULTS_CSV.with_suffix(".csv.bak")
    if not backup.exists():
        RESULTS_CSV.rename(backup)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in CSV_FIELDS}
            writer.writerow(out)
    print(f"results.csv migrated ({', '.join(missing)} added). Backup: {backup}")


def save_run_summary(
    *,
    config,
    gt_hr,
    pred_hr,
    snr_mean=None,
    method: str | None = None,
    dataset: str | None = None,
    run_name: str | None = None,
) -> Path:
    """
    Append une ligne dans runs/comparison/results.csv et sauve un JSON par run.

    Variables d'environnement :
      RPPG_RUN_NAME — nom du run (sinon dérivé du modèle / méthode)
      RPPG_CONFIG_FILE — chemin du YAML utilisé
      RPPG_ROI — ROI appliquée (défaut full_face)
    """
    if config.TOOLBOX_MODE == "unsupervised_method":
        method = method or str(getattr(config.UNSUPERVISED, "METHOD", ""))
        if isinstance(method, (list, tuple)):
            method = method[0] if method else ""
        dataset = dataset or config.UNSUPERVISED.DATA.DATASET
        eval_method = config.INFERENCE.EVALUATION_METHOD
        filename_id = f"{method}_{dataset}"
    else:
        method = method or config.MODEL.NAME
        dataset = dataset or config.TEST.DATA.DATASET
        eval_method = config.INFERENCE.EVALUATION_METHOD
        if config.TOOLBOX_MODE == "train_and_test":
            filename_id = config.TRAIN.MODEL_FILE_NAME
        else:
            model_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = f"{model_root}_{dataset}"

    run_name = run_name or os.environ.get("RPPG_RUN_NAME") or filename_id
    run_name = _ensure_unique_run_name(run_name)
    roi = os.environ.get("RPPG_ROI", "full_face") or "full_face"
    config_file = _resolve_config_file()
    paths = _paths_from_config(config)

    stats = _compute_metrics(gt_hr, pred_hr)
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_name": run_name,
        "toolbox_mode": config.TOOLBOX_MODE,
        "method": method,
        "roi": roi,
        "dataset": dataset,
        "eval_method": eval_method,
        "n_windows": stats["n_windows"],
        "mae": round(stats["mae"], 4),
        "rmse": round(stats["rmse"], 4),
        "mape": round(stats["mape"], 4),
        "pearson": round(stats["pearson"], 4),
        "snr": round(float(np.nanmean(snr_mean)), 4)
        if snr_mean is not None and len(snr_mean)
        else "",
        "config_file": config_file,
        "cached_path": paths["cached_path"],
        "results_dir": paths["results_dir"],
        "model_file_name": paths["model_file_name"],
    }

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_csv_if_needed()
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    json_path = COMPARISON_DIR / "runs" / f"{run_name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**row, "gt_hr": gt_hr.tolist(), "pred_hr": pred_hr.tolist()}
    if snr_mean is not None:
        payload["snr_per_window"] = np.asarray(snr_mean).tolist()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Run summary saved: {RESULTS_CSV}")
    print(f"  run_name={run_name}")
    print(f"  config_file={config_file or '(empty — set RPPG_CONFIG_FILE)'}")
    print(f"  roi={roi}  method={method}")
    print(f"  cached_path={paths['cached_path']}")
    print(f"  results_dir={paths['results_dir']}")
    print(f"  json={json_path}")
    return json_path
