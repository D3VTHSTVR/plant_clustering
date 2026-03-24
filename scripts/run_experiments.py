#!/usr/bin/env python3
"""
Reproducible experiment driver for CS 572 final project.
Loads data per config.py, runs feature extraction + K-Means + metrics, saves JSON/CSV + figures.

Usage:
  python scripts/run_experiments.py
  python scripts/run_experiments.py --quick          # small subset, faster
  python scripts/run_experiments.py --no-ablations   # skip raw_no_pca ablation
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

import config
from src.data_loader import load_all_image_paths, load_inaturalist2021_plants
from src.experiment_runner import run_full_experiment_bundle, save_run
from src.visualization import plot_metric_comparison_table_rows, plot_silhouette_vs_k


def _snapshot_config(quick: bool) -> dict:
    return {
        "DATASET_SOURCE": config.DATASET_SOURCE,
        "INATURALIST_VERSION": config.INATURALIST_VERSION,
        "INATURALIST_MAX_IMAGES": config.INATURALIST_MAX_IMAGES,
        "RANDOM_SEED": config.RANDOM_SEED,
        "K_RANGE": list(config.K_RANGE),
        "FEATURE_TYPES": list(config.FEATURE_TYPES),
        "CNN_BACKBONE": config.CNN_BACKBONE,
        "USE_PCA_FOR_RAW": config.USE_PCA_FOR_RAW,
        "quick_mode": quick,
    }


def main():
    ap = argparse.ArgumentParser(description="Run plant clustering experiments")
    ap.add_argument("--quick", action="store_true", help="Use fewer images and smaller k range for smoke tests")
    ap.add_argument("--no-ablations", action="store_true", dest="no_ablations", help="Skip ablation runs")
    ap.add_argument("--output", type=str, default=None, help="Output directory (default: results/experiments/run_TIMESTAMP)")
    args = ap.parse_args()

    if args.quick:
        if config.DATASET_SOURCE == "inaturalist2021":
            mi = config.INATURALIST_MAX_IMAGES
            config.INATURALIST_MAX_IMAGES = min(800, mi) if mi is not None else 800
        else:
            config.MAX_IMAGES = min(800, config.MAX_IMAGES) if config.MAX_IMAGES else 800
        config.K_RANGE = (2, 8)
        config.K_MEANS_N_INIT = 5

    if config.DATASET_SOURCE == "inaturalist2021":
        paths, labels, _ = load_inaturalist2021_plants(
            root=config.INATURALIST_DIR,
            version=config.INATURALIST_VERSION,
            max_images=config.INATURALIST_MAX_IMAGES,
            download=True,
            random_state=config.RANDOM_SEED,
        )
    else:
        paths, labels, _ = load_all_image_paths(
            config.RAW_DATA_DIR,
            max_images=config.MAX_IMAGES,
        )

    if len(paths) == 0:
        print("No images found. Use local data/ or iNaturalist download.", file=sys.stderr)
        sys.exit(1)

    y_true = np.asarray(labels) if labels is not None else None

    bundle = run_full_experiment_bundle(
        paths,
        y_true,
        config,
        run_ablations=not args.no_ablations,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output) if args.output else (config.RESULTS_DIR / "experiments" / f"run_{run_id}")
    snap = _snapshot_config(args.quick)
    save_run(out, bundle, snap)

    # Figures
    plot_silhouette_vs_k(bundle["main"], out / "fig_silhouette_vs_k.png")
    rows = []
    for k, r in bundle["main"].items():
        rows.append(
            {
                "feature": k,
                "silhouette": r["silhouette"],
                "best_k": r["best_k"],
                "ari": r["supervised"].get("ari"),
                "nmi": r["supervised"].get("nmi"),
            }
        )
    plot_metric_comparison_table_rows(rows, out / "fig_feature_silhouette_bar.png")

    (out / "RUN_INFO.txt").write_text(
        f"run_id={run_id}\n"
        f"n_images={len(paths)}\n"
        f"output_dir={out.resolve()}\n",
        encoding="utf-8",
    )
    print(f"Saved results to {out.resolve()}")
    print("See metrics.json, metrics_table.csv, and fig_*.png")


if __name__ == "__main__":
    main()
