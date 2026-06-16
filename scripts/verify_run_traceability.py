#!/usr/bin/env python3
"""Vérifie que runs/comparison/results.csv contient les colonnes de traçabilité."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "runs" / "comparison" / "results.csv"

ALL_FIELDS = [
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

REQUIRED = [
    "timestamp",
    "run_name",
    "method",
    "roi",
    "config_file",
    "mae",
    "rmse",
    "pearson",
    "cached_path",
    "results_dir",
]

METRICS = ["mae", "rmse", "mape", "pearson", "snr", "n_windows"]


def migrate_csv() -> None:
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in ALL_FIELDS if c not in old_fields]
    if not missing:
        print("CSV déjà à jour.")
        return
    backup = CSV_PATH.with_suffix(".csv.bak")
    if not backup.exists():
        CSV_PATH.rename(backup)
        print(f"Backup: {backup}")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ALL_FIELDS})
    print(f"Migré — colonnes ajoutées: {', '.join(missing)}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--migrate":
        migrate_csv()
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    if not CSV_PATH.exists():
        print(f"Missing: {CSV_PATH}")
        return 1

    with CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    missing_cols = [c for c in REQUIRED if c not in fields]
    if missing_cols:
        print("Colonnes manquantes:", ", ".join(missing_cols))
        print("Colonnes actuelles:", ", ".join(fields))
        return 1

    print(f"OK — {len(rows)} ligne(s), colonnes: {', '.join(fields)}")
    print()

    issues = 0
    for i, row in enumerate(rows, start=1):
        name = row.get("run_name", "")
        problems = []
        if not row.get("config_file", "").strip():
            problems.append("config_file vide")
        if not row.get("roi", "").strip():
            problems.append("roi vide")
        if not row.get("method", "").strip():
            problems.append("method vide")
        if not row.get("timestamp", "").strip():
            problems.append("timestamp vide")
        json_path = ROOT / "runs" / "comparison" / "runs" / f"{name}.json"
        if name and not json_path.exists():
            problems.append(f"JSON absent: {json_path.name}")

        status = "OK" if not problems else "; ".join(problems)
        print(f"  [{i}] {name}: {status}")
        if problems:
            issues += 1

    if issues:
        print(f"\n{issues} run(s) avec avertissements (souvent runs anciens avant RPPG_CONFIG_FILE).")
        return 0
    print("\nTous les runs sont traçables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
