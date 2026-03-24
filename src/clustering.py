"""
K-Means clustering and evaluation for plant image features.
"""

import numpy as np
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def run_kmeans(X: np.ndarray, k: int, init="k-means++", n_init=10, max_iter=300, random_state=42):
    """Run K-Means and return model, labels, and inertia."""
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return kmeans, labels, kmeans.inertia_


def evaluate_silhouette(X: np.ndarray, labels: np.ndarray) -> tuple:
    """Compute overall silhouette score and per-sample scores."""
    score = silhouette_score(X, labels)
    samples = silhouette_samples(X, labels)
    return score, samples


def tune_k(X: np.ndarray, k_range: tuple, **kmeans_kw):
    """
    Try multiple k values, return best k by silhouette score.
    k_range: (min_k, max_k) inclusive
    Returns: dict with k_values, silhouette_scores, inertia_values, best_k
    """
    k_min, k_max = k_range
    k_values = list(range(k_min, k_max))
    sil_scores = []
    inertias = []

    for k in k_values:
        _, labels, inertia = run_kmeans(X, k, **kmeans_kw)
        inertias.append(inertia)
        if k > 1 and len(np.unique(labels)) > 1:
            sil = silhouette_score(X, labels)
            sil_scores.append(sil)
        else:
            sil_scores.append(-1)

    best_idx = np.argmax(sil_scores)
    best_k = k_values[best_idx]

    return {
        "k_values": k_values,
        "silhouette_scores": sil_scores,
        "inertias": inertias,
        "best_k": best_k,
    }


def apply_pca(X: np.ndarray, n_components=None, variance_retained=0.95, random_state=42):
    """
    Apply PCA for dimensionality reduction.
    n_components: fixed number, or None to use variance_retained.
    Returns: X_reduced, pca_model
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=random_state)
    else:
        pca = PCA(n_components=min(X.shape[0], X.shape[1]), random_state=random_state)
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n = np.searchsorted(cumvar, variance_retained) + 1
        n = min(n, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n, random_state=random_state)

    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca, scaler
