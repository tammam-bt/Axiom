"""Per-feature scaling. Every statistic is taken per column (axis 0).

    min-max:   (x - min) / (max - min)     maps each feature to [0, 1]
    standard:  (x - mean) / std            zero mean, unit variance (population std)
    max-abs:   x / max(|x|)                maps each feature to [-1, 1], keeps zeros at 0

A constant column would divide by zero, so its divisor is set to 1: min-max
and standard scaling turn it into 0, and max-abs scaling into its sign.

The classes store the training statistics, so transform() applies the training
set's scaling to new data. The functions scale an array by its own statistics.
A 1-D array is treated as a single feature.
"""
import numpy as np


def _columns(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1), True
    if x.ndim != 2:
        raise ValueError("expected a 1-D array or a 2-D array of shape (n_samples, n_features)")
    return x, False


def _restore(x, was_1d):
    return x.ravel() if was_1d else x


def _min_max_stats(X):
    data_min = X.min(axis=0)
    data_range = X.max(axis=0) - data_min
    return data_min, np.where(data_range == 0, 1.0, data_range)


def _standard_stats(X):
    constant = X.min(axis=0) == X.max(axis=0)
    # For a constant column use the value itself, so x - mean is exactly 0.
    mean = np.where(constant, X[0], X.mean(axis=0))
    return mean, np.where(constant, 1.0, X.std(axis=0))


def _max_abs_stats(X):
    max_abs = np.abs(X).max(axis=0)
    return np.zeros(X.shape[1]), np.where(max_abs == 0, 1.0, max_abs)


def _scale_by_own_stats(stats, x):
    X, was_1d = _columns(x)
    shift, divisor = stats(X)
    return _restore((X - shift) / divisor, was_1d)


def min_max_scale(x):
    """Scale each feature to [0, 1] by its own min and max."""
    return _scale_by_own_stats(_min_max_stats, x)


def standard_scale(x):
    """Scale each feature to zero mean and unit variance by its own mean and std."""
    return _scale_by_own_stats(_standard_stats, x)


def max_abs_scale(x):
    """Divide each feature by its own largest absolute value."""
    return _scale_by_own_stats(_max_abs_stats, x)


class _Scaler:
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _check(self, X):
        if getattr(self, "n_features_", None) is None:
            raise RuntimeError("call fit before transform")
        X, was_1d = _columns(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, the scaler was fitted on {self.n_features_}")
        return X, was_1d


class MinMaxScaler(_Scaler):
    """Maps each feature to [0, 1] using the min and max seen in fit.

    Attributes after fit: data_min_, data_max_, data_range_ (1 for a constant column).
    """

    def fit(self, X):
        X, _ = _columns(X)
        self.n_features_ = X.shape[1]
        self.data_min_, self.data_range_ = _min_max_stats(X)
        self.data_max_ = X.max(axis=0)
        return self

    def transform(self, X):
        X, was_1d = self._check(X)
        return _restore((X - self.data_min_) / self.data_range_, was_1d)

    def inverse_transform(self, X):
        X, was_1d = self._check(X)
        return _restore(X * self.data_range_ + self.data_min_, was_1d)


class StandardScaler(_Scaler):
    """Gives each feature zero mean and unit variance using the mean and std seen in fit.

    Attributes after fit: mean_, scale_ (the population std, 1 for a constant column).
    """

    def fit(self, X):
        X, _ = _columns(X)
        self.n_features_ = X.shape[1]
        self.mean_, self.scale_ = _standard_stats(X)
        return self

    def transform(self, X):
        X, was_1d = self._check(X)
        return _restore((X - self.mean_) / self.scale_, was_1d)

    def inverse_transform(self, X):
        X, was_1d = self._check(X)
        return _restore(X * self.scale_ + self.mean_, was_1d)
