#!/usr/bin/env python3
"""Génère des graphiques de comparaison à partir de runs/comparison/results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "runs" / "comparison" / "results.csv"
OUT_DIR = ROOT / "runs" / "comparison" / "plots"


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    labels = df["run_name"].astype(str)
    values = df[metric].astype(float)
    order = values.sort_values().index
    labels = labels.loc[order]
    values = values.loc[order]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 5))
    bars = ax.bar(range(len(labels)), values, color="steelblue", edgecolor="black")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Comparaison des méthodes — {metric.upper()}")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bland_altman_overlay(df: pd.DataFrame, out_path: Path) -> None:
    """Superpose les paires HR prédites / GT de chaque run (fichiers JSON)."""
    runs_dir = ROOT / "runs" / "comparison" / "runs"
    if not runs_dir.is_dir():
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for run_name in df["run_name"]:
        json_path = runs_dir / f"{run_name}.json"
        if not json_path.exists():
            continue
        import json

        data = json.loads(json_path.read_text(encoding="utf-8"))
        gt = data.get("gt_hr", [])
        pred = data.get("pred_hr", [])
        if not gt or not pred:
            continue
        ax.scatter(gt, pred, alpha=0.5, s=12, label=str(run_name))

    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Identité")
    ax.set_xlabel("HR référence (bpm)")
    ax.set_ylabel("HR estimée (bpm)")
    ax.set_title("Comparaison scatter (tous runs)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparer les runs rPPG-Toolbox")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Fichier results.csv (défaut: runs/comparison/results.csv)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Ne garder que les N dernières lignes par run_name (0 = tout)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Aucun résultat : {args.csv}")
        print("Lancez des expériences avec RPPG_RUN_NAME défini, ex. :")
        print("  RPPG_RUN_NAME=POS_full RPPG_CONFIG_FILE=configs/... python main.py ...")
        return

    df = pd.read_csv(args.csv)
    if df.empty:
        print("CSV vide.")
        return

    if args.last > 0:
        df = df.groupby("run_name", as_index=False).tail(args.last)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in ("mae", "rmse", "mape", "pearson"):
        if metric in df.columns:
            plot_metric(df, metric, OUT_DIR / f"compare_{metric}.png")

    plot_bland_altman_overlay(df, OUT_DIR / "compare_scatter_all.png")
    print(f"Graphiques enregistrés dans {OUT_DIR}")


if __name__ == "__main__":
    main()
