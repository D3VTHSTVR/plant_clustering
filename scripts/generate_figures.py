#!/usr/bin/env python3
"""
Regenerate figures from a saved experiment run (reproducibility / report revisions).

Usage:
  python scripts/generate_figures.py results/experiments/run_20260324_120000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import plot_metric_comparison_table_rows, plot_silhouette_vs_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="Directory containing metrics.json")
    args = ap.parse_args()
    d = Path(args.run_dir)
    data = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    plot_silhouette_vs_k(data["main"], d / "fig_silhouette_vs_k.png")
    rows = []
    for k, r in data["main"].items():
        rows.append(
            {
                "feature": k,
                "silhouette": r["silhouette"],
                "best_k": r["best_k"],
                "ari": r["supervised"].get("ari"),
                "nmi": r["supervised"].get("nmi"),
            }
        )
    plot_metric_comparison_table_rows(rows, d / "fig_feature_silhouette_bar.png")
    print(f"Updated figures in {d.resolve()}")


if __name__ == "__main__":
    main()
