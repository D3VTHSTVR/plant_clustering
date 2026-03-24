"""
End-to-end experiment orchestration for reproducible results and report tables.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.clustering import apply_pca, run_kmeans, tune_k
from src.evaluation import cluster_statistics, limitation_notes, silhouette_breakdown, supervised_metrics
from src.features import (
    extract_cnn_embeddings,
    extract_handcrafted_from_paths,
    extract_raw_pixels_from_paths,
)


def _prepare_matrix(
    X: np.ndarray,
    feature_name: str,
    use_pca_for_raw: bool,
    pca_variance: float,
    random_state: int,
):
    """Scale and optionally PCA-reduce features."""
    if feature_name == "raw_pixels" and use_pca_for_raw:
        X_out, _, _ = apply_pca(X, variance_retained=pca_variance, random_state=random_state)
        return X_out, {"preprocess": "standardize_pca", "input_dim": X.shape[1], "output_dim": X_out.shape[1]}
    scaler = StandardScaler()
    X_out = scaler.fit_transform(X)
    return X_out, {"preprocess": "standardize_only", "dim": X_out.shape[1]}


def run_single_feature_pipeline(
    paths: list,
    y_true: np.ndarray | None,
    feature_name: str,
    config_module: Any,
    use_pca_for_raw: bool | None = None,
) -> dict:
    """
    Extract features, tune k, fit K-Means at best k, compute metrics.
    """
    cfg = config_module
    rs = cfg.RANDOM_SEED
    use_pca = cfg.USE_PCA_FOR_RAW if use_pca_for_raw is None else use_pca_for_raw

    t_start = time.perf_counter()
    if feature_name == "raw_pixels":
        X = extract_raw_pixels_from_paths(paths, size=cfg.IMG_SIZE)
    elif feature_name == "handcrafted":
        X = extract_handcrafted_from_paths(paths, size=cfg.IMG_SIZE)
    elif feature_name == "cnn_embeddings":
        X = extract_cnn_embeddings(
            paths, backbone=cfg.CNN_BACKBONE, size=cfg.IMG_SIZE_CNN
        )
        X = StandardScaler().fit_transform(X)
    else:
        raise ValueError(f"Unknown feature: {feature_name}")

    t_extract = time.perf_counter() - t_start

    Xp, prep_meta = _prepare_matrix(X, feature_name, use_pca, cfg.PCA_VARIANCE_RETAINED, rs)

    kmeans_kw = dict(
        init=cfg.K_MEANS_INIT,
        n_init=cfg.K_MEANS_N_INIT,
        max_iter=cfg.K_MEANS_MAX_ITER,
        random_state=rs,
    )
    tune = tune_k(Xp, cfg.K_RANGE, **kmeans_kw)
    best_k = tune["best_k"]
    _, labels, inertia = run_kmeans(Xp, best_k, **kmeans_kw)

    sil = float(silhouette_score(Xp, labels))
    sup = supervised_metrics(y_true, labels) if y_true is not None else {"ari": None, "nmi": None}
    stats = cluster_statistics(labels)
    sil_bd = silhouette_breakdown(Xp, labels)
    notes = limitation_notes(stats, sup)

    return {
        "feature": feature_name,
        "use_pca_for_raw": use_pca if feature_name == "raw_pixels" else None,
        "prep": prep_meta,
        "n_samples": int(X.shape[0]),
        "feature_dim_raw": int(X.shape[1]),
        "seconds_extract": round(t_extract, 2),
        "seconds_total": round(time.perf_counter() - t_start, 2),
        "k_tune": tune,
        "best_k": int(best_k),
        "silhouette": sil,
        "inertia": float(inertia),
        "supervised": sup,
        "cluster_stats": stats,
        "silhouette_per_cluster": sil_bd["per_cluster"],
        "limitation_notes": notes,
    }


def run_full_experiment_bundle(
    paths: list,
    y_true: np.ndarray | None,
    config_module: Any,
    run_ablations: bool = True,
) -> dict:
    """
    Main + optional ablations for sensitivity analysis.
    """
    cfg = config_module
    results = {
        "main": {},
        "ablations": [],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    for name in cfg.FEATURE_TYPES:
        results["main"][name] = run_single_feature_pipeline(
            paths, y_true, name, cfg, use_pca_for_raw=None
        )

    if run_ablations:
        if "raw_pixels" in cfg.FEATURE_TYPES:
            results["ablations"].append(
                {
                    "name": "raw_pixels_no_pca",
                    "result": run_single_feature_pipeline(
                        paths, y_true, "raw_pixels", cfg, use_pca_for_raw=False
                    ),
                }
            )
    return results


def _jsonable(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_run(out_dir: Path, bundle: dict, config_snapshot: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(_jsonable(bundle), indent=2), encoding="utf-8"
    )
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(_jsonable(config_snapshot), indent=2), encoding="utf-8"
    )

    # Flat CSV for paper Table
    rows = []
    for fname, r in bundle["main"].items():
        row = {
            "feature": fname,
            "best_k": r["best_k"],
            "silhouette": r["silhouette"],
            "ari": r["supervised"].get("ari"),
            "nmi": r["supervised"].get("nmi"),
            "n_samples": r["n_samples"],
        }
        rows.append(row)
    for ab in bundle.get("ablations", []):
        r = ab["result"]
        rows.append(
            {
                "feature": ab["name"],
                "best_k": r["best_k"],
                "silhouette": r["silhouette"],
                "ari": r["supervised"].get("ari"),
                "nmi": r["supervised"].get("nmi"),
                "n_samples": r["n_samples"],
            }
        )

    import csv

    csv_path = out_dir / "metrics_table.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
