"""Gradient boosting with regression trees.

The model is an additive sum F(x) = F_0 + lr * (h_1(x) + ... + h_M(x)). Each new
tree h_m is a DecisionTreeRegressor fitted to the pseudo-residuals, the negative
gradient of the loss at the current F:

    GBMRegressor   squared error, F_0 = mean(y),            residual = y - F
    GBMClassifier  log-loss,      F_0 = log(p / (1 - p)),   residual = y - sigmoid(F)

where p is the share of the positive class.

The tree's splits come from fitting the residuals. For squared error each leaf
then predicts the mean residual of its samples. For log-loss each leaf instead
takes one Newton step on the samples it holds, with s = sigmoid(F):

    leaf value = sum(y - s) / sum(s * (1 - s))

which is the minimizer of a second-order expansion of the log-loss in that leaf
(Friedman 2001; scikit-learn's GradientBoostingClassifier does the same).
"""
import numpy as np
from ..core.base import BaseEstimator
from .decision_tree import DecisionTreeRegressor


def _sigmoid(z):
    # exp of a negative number only, so large |z| cannot overflow
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class _BaseGBM(BaseEstimator):
    def __init__(self, n_estimators = 100, learning_rate = 0.1, max_depth = 3, min_samples_split = 2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _boost(self, X, y, init):
        """Fit `n_estimators` trees to the pseudo-residuals of y, starting from the constant `init`."""
        self.init_ = float(init)
        self.estimators_ = []
        F = np.full(len(y), self.init_)
        losses = [self._loss(y, F)]
        for _ in range(self.n_estimators):
            residual = self._negative_gradient(y, F)
            tree = DecisionTreeRegressor(min_samples_split = self.min_samples_split, max_depth = self.max_depth)
            tree.fit(X, residual)
            self._set_leaf_values(tree, X, y, F)
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            losses.append(self._loss(y, F))
        # train_loss_[0] is the loss of the constant start, train_loss_[m] the loss after m trees.
        self.train_loss_ = np.array(losses)
        return self

    def _set_leaf_values(self, tree, X, y, F):
        """Hook to replace the mean-residual leaves of a freshly fitted tree."""

    def _raw_predict(self, X):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("call fit before predict")
        X = np.asarray(X, dtype = float)
        F = np.full(len(X), self.init_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    @staticmethod
    def _check_X_y(X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features)")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one target per row of X")
        return X, y


class GBMRegressor(_BaseGBM):
    """Gradient boosting for regression with squared-error loss."""

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        y = y.astype(float)
        return self._boost(X, y, init = np.mean(y))

    def predict(self, X):
        return self._raw_predict(X)

    @staticmethod
    def _loss(y, F):
        return float(np.mean((y - F) ** 2))

    @staticmethod
    def _negative_gradient(y, F):
        return y - F


class GBMClassifier(_BaseGBM):
    """Gradient boosting for binary classification with log-loss.

    classes_[1] is the positive class; predict_proba returns its probability.
    """

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"GBMClassifier is binary: y must hold exactly 2 classes, got {len(self.classes_)}")
        y01 = (y == self.classes_[1]).astype(float)
        p = np.mean(y01)
        return self._boost(X, y01, init = np.log(p / (1 - p)))

    def decision_function(self, X):
        """Raw boosted score F(x), the log-odds of the positive class."""
        return self._raw_predict(X)

    def predict_proba(self, X):
        """Probability of classes_[1], shape (n_samples,)."""
        return _sigmoid(self._raw_predict(X))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X) >= 0.5).astype(int)]

    @staticmethod
    def _loss(y, F):
        # mean of -[y log s(F) + (1 - y) log(1 - s(F))], written to stay finite for large |F|
        return float(np.mean(np.logaddexp(0.0, F) - y * F))

    @staticmethod
    def _negative_gradient(y, F):
        return y - _sigmoid(F)

    def _set_leaf_values(self, tree, X, y, F):
        s = _sigmoid(F)
        for leaf, rows in tree._leaf_rows(X):
            hessian = np.sum(s[rows] * (1.0 - s[rows]))
            leaf.predicted_value = float(np.sum(y[rows] - s[rows]) / hessian) if hessian > 1e-150 else 0.0


# docs/documentation.md promises `from axiom.trees.gbm import GBM`.
GBM = GBMRegressor
