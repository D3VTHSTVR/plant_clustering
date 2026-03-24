"""Smoke tests for evaluation metrics."""
import numpy as np
import pytest

from src.evaluation import cluster_statistics, supervised_metrics


def test_supervised_metrics_perfect_match():
    y = np.array([0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 2])
    m = supervised_metrics(y, pred)
    assert m["ari"] == pytest.approx(1.0)
    assert m["nmi"] == pytest.approx(1.0)


def test_cluster_statistics():
    pred = np.array([0, 0, 1, 1, 1])
    s = cluster_statistics(pred)
    assert s["n_clusters"] == 2
    assert s["cluster_sizes"] == [2, 3]
