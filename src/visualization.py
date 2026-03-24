"""Figures for experiments (used by scripts/generate_figures.py and notebook)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_silhouette_vs_k(bundle_main: dict, out_path: Path, title: str = "Silhouette vs k"):
    """bundle_main: feature_name -> result dict from experiment_runner."""
    n = len(bundle_main)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, bundle_main.items()):
        kt = res["k_tune"]
        ax.plot(kt["k_values"], kt["silhouette_scores"], "o-", label="silhouette")
        ax.axvline(res["best_k"], color="C1", ls="--", label=f"best k={res['best_k']}")
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison_table_rows(rows: list, out_path: Path):
    """Bar chart of silhouette by feature (from metrics_table rows)."""
    names = [r["feature"] for r in rows if not str(r["feature"]).startswith("raw_pixels_no")]
    sils = [r["silhouette"] for r in rows if not str(r["feature"]).startswith("raw_pixels_no")]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x, sils, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Silhouette (best k)")
    ax.set_title("Feature comparison (main runs)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_sizes(cluster_stats: dict, out_path: Path, title: str = "Cluster sizes"):
    sizes = cluster_stats.get("cluster_sizes", [])
    if not sizes:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(sizes)), sizes, color="seagreen")
    ax.set_xlabel("Cluster id")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
