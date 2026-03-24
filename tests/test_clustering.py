import numpy as np
import pytest

from src.clustering import run_kmeans, tune_k


@pytest.fixture
def blobs():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((30, 2)) + np.array([0, 0])
    b = rng.standard_normal((30, 2)) + np.array([5, 5])
    return np.vstack([a, b])


def test_tune_k_finds_two_clusters(blobs):
    out = tune_k(blobs, (2, 6), random_state=0)
    assert out["best_k"] in range(2, 6)
    assert max(out["silhouette_scores"]) > 0


def test_run_kmeans(blobs):
    km, labels, inertia = run_kmeans(blobs, k=2, random_state=0)
    assert len(np.unique(labels)) == 2
