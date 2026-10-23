"""Impurity measures and information gain for decision trees.

Classification impurities work from class counts, so they handle any number of
classes and any label type. With p_k the share of class k in a node:

    gini(y)    = 1 - sum_k p_k^2
    entropy(y) = -sum_k p_k * log2(p_k)          (0 * log2(0) is taken as 0)

For regression targets the impurity is the population variance:

    variance(y) = mean((y - mean(y))^2)

A split sends the samples in `left_mask` left and the rest right. Its gain is
the parent impurity minus the children's impurity, weighted by their sizes:

    gain = I(y) - (n_left / n) * I(y_left) - (n_right / n) * I(y_right)

The *_from_counts helpers take a (..., n_classes) array of counts, so a tree can
score every candidate threshold in one vectorized call.
"""
import numpy as np


def gini_from_counts(counts):
    """Gini impurity of each row of class counts; rows with no samples give 0."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(n > 0, n, 1)
    return np.where(n[..., 0] > 0, 1.0 - np.sum(p * p, axis=-1), 0.0)


def entropy_from_counts(counts):
    """Entropy in bits of each row of class counts; rows with no samples give 0."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(n > 0, n, 1)
    # log2 of 1 is 0, so empty classes contribute nothing and log2(0) never runs.
    return -np.sum(p * np.log2(np.where(p > 0, p, 1.0)), axis=-1)


def _class_counts(y):
    _, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
    return counts


def gini(y) -> float:
    """Gini impurity of a label array."""
    if np.asarray(y).size == 0:
        return 0.0
    return float(gini_from_counts(_class_counts(y)))


def entropy(y) -> float:
    """Entropy (in bits) of a label array."""
    if np.asarray(y).size == 0:
        return 0.0
    return float(entropy_from_counts(_class_counts(y)))


def variance(y) -> float:
    """Population variance of a regression target."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    return float(np.var(y))


IMPURITIES = {"gini": gini, "entropy": entropy, "variance": variance}


def information_gain(y, left_mask, impurity) -> float:
    """Impurity decrease when the samples in `left_mask` go left and the rest go right.

    `impurity` is one of gini, entropy, variance (or any function of a target
    array), or the name of one of them.
    """
    if isinstance(impurity, str):
        if impurity not in IMPURITIES:
            raise ValueError(f"impurity must be one of {sorted(IMPURITIES)}, got {impurity!r}")
        impurity = IMPURITIES[impurity]
    y = np.asarray(y)
    left_mask = np.asarray(left_mask, dtype=bool)
    if left_mask.shape != y.shape[:1]:
        raise ValueError("left_mask must hold one boolean per sample")
    n = len(y)
    if n == 0:
        return 0.0
    y_left, y_right = y[left_mask], y[~left_mask]
    weighted = (len(y_left) * impurity(y_left) + len(y_right) * impurity(y_right)) / n
    return float(impurity(y) - weighted)
