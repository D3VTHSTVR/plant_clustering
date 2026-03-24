"""
Clustering evaluation: unsupervised (silhouette) and supervised (ARI, NMI) when labels exist.
Diagnostic statistics for limitation analysis.
"""

import numpy as np

try:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
        silhouette_samples,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def supervised_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compare cluster labels to reference labels (e.g. species id or folder class).
    ARI and NMI in [0, 1]; higher is better alignment.
    """
    if y_true is None or len(y_true) == 0:
        return {"ari": None, "nmi": None}
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return {"ari": None, "nmi": None}
    yt = y_true[:n]
    yp = y_pred[:n]
    mask = yt >= 0 if np.issubdtype(yt.dtype, np.integer) or np.issubdtype(yt.dtype, np.floating) else np.ones(n, dtype=bool)
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {"ari": None, "nmi": None}
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required: pip install scikit-learn")
    return {
        "ari": float(adjusted_rand_score(yt, yp)),
        "nmi": float(normalized_mutual_info_score(yt, yp)),
    }


def cluster_statistics(y_pred: np.ndarray) -> dict:
    """Cluster sizes and imbalance (for limitation discussion)."""
    y_pred = np.asarray(y_pred).ravel()
    unique, counts = np.unique(y_pred, return_counts=True)
    return {
        "n_clusters": int(len(unique)),
        "cluster_sizes": counts.tolist(),
        "size_min": int(counts.min()),
        "size_max": int(counts.max()),
        "size_std": float(counts.std()),
        "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
    }


def silhouette_breakdown(X: np.ndarray, labels: np.ndarray) -> dict:
    """Mean silhouette per cluster."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required")
    labels = np.asarray(labels).ravel()
    if len(np.unique(labels)) < 2:
        return {"per_cluster": {}, "overall": -1.0}
    samples = silhouette_samples(X, labels)
    overall = float(silhouette_score(X, labels))
    per_cluster = {}
    for c in np.unique(labels):
        mask = labels == c
        per_cluster[int(c)] = float(samples[mask].mean())
    return {"per_cluster": per_cluster, "overall": overall}


def limitation_notes(stats: dict, sup_metrics: dict) -> list:
    """Bullet points for report Discussion (auto-generated hints)."""
    notes = []
    if stats.get("imbalance_ratio", 1) > 3:
        notes.append(
            f"Cluster size imbalance (ratio {stats['imbalance_ratio']:.1f}) may indicate "
            "K-Means favoring large modes or outliers."
        )
    if sup_metrics.get("ari") is not None and sup_metrics["ari"] < 0.05:
        notes.append(
            "Low ARI vs taxonomy labels suggests clusters do not align with species or "
            "visual classes; unsupervised structure may differ from human taxonomy."
        )
    if sup_metrics.get("nmi") is not None and sup_metrics["nmi"] < 0.1:
        notes.append(
            "Low NMI indicates partial overlap at best between clusters and reference labels."
        )
    if not notes:
        notes.append("Review silhouette per-cluster and cluster galleries for qualitative failure modes.")
    return notes
