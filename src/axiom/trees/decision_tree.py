import numpy as np
from ..core.base import BaseEstimator
from .gains import entropy_from_counts, gini_from_counts


class Node:
    def __init__(self, left = None, right = None, feature_index = None,feature_type = None,threshold= None, predicted_value = None, info_gain = None):
        # Decision Node
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.info_gain = info_gain
        #Leaf Node
        self.predicted_value = predicted_value

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


class _BaseTree(BaseEstimator):
    """Binary tree grown depth-first with splits of the form x[feature] <= threshold.

    A node becomes a leaf when it is pure, holds fewer than `min_samples_split`
    samples, sits at depth `max_depth` (the root is depth 0; None means no
    limit), or has no feature with two distinct values. Otherwise it takes the
    split with the largest impurity decrease, even when that decrease is 0, the
    same rule scikit-learn uses.

    Subclasses define the target encoding, the impurity and the leaf value.
    """

    def __init__(self, min_samples_split = 2, max_depth = 3, verbose = False):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.verbose = verbose
        self.depth = 0
        self.root = None

    # ---- fitting -------------------------------------------------------------

    def fit(self, X, y):
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one target per row of X")
        if len(X) == 0:
            raise ValueError("cannot fit a tree on 0 samples")
        self._check_params()
        self.n_features_ = X.shape[1]
        self.root = self._grow(X, self._encode_target(y), depth = 0)
        self.depth = self._depth_of(self.root)
        if(self.verbose):
            print(f"Tree building completed. Depth: {self.depth}, root feature_index: {self.root.feature_index}, threshold: {self.root.threshold}, info_gain: {self.root.info_gain}")
        return self

    def _grow(self, X, y, depth):
        n_samples = len(y)
        if(self.verbose):
            print(f"Building tree at depth {depth} with {n_samples} samples")
        can_split = (n_samples >= self.min_samples_split
                     and (self.max_depth is None or depth < self.max_depth)
                     and not np.all(y == y[0]))
        if can_split:
            split = self._best_split(X, y)
            if split is not None:
                feature, threshold, gain = split
                if(self.verbose):
                    print(f"Best split at depth {depth}: feature_index={feature}, threshold={threshold}, info_gain={gain}")
                mask = X[:, feature] <= threshold
                left = self._grow(X[mask], y[mask], depth + 1)
                right = self._grow(X[~mask], y[~mask], depth + 1)
                return Node(left = left, right = right, feature_index = feature, threshold = threshold, info_gain = gain)
        return Node(predicted_value = self._leaf_value(y))

    # Upper bound on the elements of the (n_samples, n_features, n_classes) work
    # arrays built per block of features in _best_split.
    _BLOCK_ELEMENTS = 2_000_000

    def _best_split(self, X, y):
        """Best (feature, threshold, gain), or None when no feature can be split.

        Each column is sorted once. Cumulative sums over the sorted targets then
        give the impurity of both children for every split position of every
        feature at the same time, so there is no Python loop over thresholds.
        Candidate thresholds are the midpoints between consecutive distinct values.
        Ties go to the lowest feature index, then to the lowest threshold.
        """
        n_samples, n_features = X.shape
        parent = self._node_impurity(y)
        block = max(1, self._BLOCK_ELEMENTS // (n_samples * self._target_width()))
        best, best_children = None, np.inf
        for start in range(0, n_features, block):
            columns = X[:, start:start + block]
            order = np.argsort(columns, axis = 0, kind = "stable")
            xs = np.take_along_axis(columns, order, axis = 0)
            distinct = xs[:-1] < xs[1:]  # row i splits between xs[i] and xs[i + 1]
            if not distinct.any():
                continue
            children = np.where(distinct, self._children_impurity(y[order]), np.inf)
            feature, i = divmod(int(np.argmin(children.T)), n_samples - 1)
            if children[i, feature] < best_children:
                low, high = xs[i, feature], xs[i + 1, feature]
                threshold = (low + high) / 2.0
                if not low < threshold < high:  # adjacent floats: the midpoint rounds onto an end
                    threshold = low
                best_children = children[i, feature]
                best = (start + feature, float(threshold), max(0.0, float(parent - best_children)))
        return best

    # ---- prediction ----------------------------------------------------------

    def predict(self, X):
        X = self._check_fitted_X(X)
        predictions = np.empty(len(X), dtype = self._output_dtype())
        for leaf, rows in self._leaf_rows(X):
            predictions[rows] = leaf.predicted_value
        return predictions

    def _leaf_rows(self, X):
        """Yield (leaf, row indices of X that land in it) for every leaf that gets rows.

        All rows that reach a node are routed together with one comparison,
        instead of walking the tree once per row.
        """
        stack = [(self.root, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if rows.size == 0:
                continue
            if node.is_leaf:
                yield node, rows
                continue
            go_left = X[rows, node.feature_index] <= node.threshold
            stack.append((node.right, rows[~go_left]))
            stack.append((node.left, rows[go_left]))

    def _check_fitted_X(self, X):
        if self.root is None:
            raise RuntimeError("call fit before predict")
        X = self._check_X(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, the tree was fitted on {self.n_features_}")
        return X

    # ---- inspection ----------------------------------------------------------

    def get_depth(self):
        """Depth of the fitted tree: 0 for a single leaf."""
        return self.depth

    def _depth_of(self, node):
        if node.is_leaf:
            return 0
        return 1 + max(self._depth_of(node.left), self._depth_of(node.right))

    def print_tree(self, node=None, level=0, direction = "Root"):
        if node is None:
            node = self.root
        indent = "  " * level
        if node.is_leaf:
            print(f"{indent} {direction} Leaf: Predict {node.predicted_value}")
        else:
            print(f"{indent} {direction} Node: feature_index={node.feature_index}, threshold={node.threshold}, info_gain={node.info_gain}")
            self.print_tree(node.left, level + 1, direction = "Left")
            self.print_tree(node.right, level + 1, direction = "Right")

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype = float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features)")
        return X

    def _check_params(self):
        pass

    # Subclass hooks
    def _encode_target(self, y):
        raise NotImplementedError

    def _node_impurity(self, y):
        raise NotImplementedError

    def _children_impurity(self, y_sorted):
        """Size-weighted impurity of the two children for every split position.

        y_sorted has shape (n, k): column j holds the targets in the order of
        feature j. The result has shape (n - 1, k); row i is the split that sends
        the first i + 1 sorted samples left.
        """
        raise NotImplementedError

    def _target_width(self):
        """Values per sample in the _children_impurity work arrays."""
        return 1

    def _leaf_value(self, y):
        raise NotImplementedError

    def _output_dtype(self):
        raise NotImplementedError


class DecisionTreeClassifier(_BaseTree):
    """Classification tree for any number of classes and any label type.

    criterion: "entropy" (information gain, the default) or "gini".
    Leaves predict the majority class of their samples.
    """

    CRITERIA = {"gini": gini_from_counts, "entropy": entropy_from_counts}

    def __init__(self, criterion = "entropy", min_samples_split = 2, max_depth = 3, verbose = False):
        super().__init__(min_samples_split = min_samples_split, max_depth = max_depth, verbose = verbose)
        self.criterion = criterion
        self._check_params()

    def _check_params(self):
        if self.criterion not in self.CRITERIA:
            raise ValueError(f"criterion must be one of {sorted(self.CRITERIA)}, got {self.criterion!r}")

    def _encode_target(self, y):
        self.classes_, y_index = np.unique(y, return_inverse = True)
        self.n_classes_ = len(self.classes_)
        self._impurity = self.CRITERIA[self.criterion]
        return y_index.ravel()

    def _node_impurity(self, y):
        return float(self._impurity(np.bincount(y, minlength = self.n_classes_)))

    def _children_impurity(self, y_sorted):
        n = len(y_sorted)
        one_hot = (y_sorted[..., None] == np.arange(self.n_classes_)).astype(float)  # (n, k, classes)
        left_counts = np.cumsum(one_hot, axis = 0)[:-1]
        right_counts = one_hot.sum(axis = 0) - left_counts
        n_left = np.arange(1, n, dtype = float)[:, None]
        return (n_left * self._impurity(left_counts) + (n - n_left) * self._impurity(right_counts)) / n

    def _target_width(self):
        return self.n_classes_

    def _leaf_value(self, y):
        return self.classes_[np.argmax(np.bincount(y, minlength = self.n_classes_))]

    def _output_dtype(self):
        return self.classes_.dtype


class DecisionTreeRegressor(_BaseTree):
    """Regression tree: splits minimize the children's variance, leaves predict the mean."""

    criterion = "variance"

    def _encode_target(self, y):
        return np.asarray(y, dtype = float)

    def _node_impurity(self, y):
        return float(np.var(y))

    def _children_impurity(self, y_sorted):
        n = len(y_sorted)
        # Every column holds the same targets in a different order, so the node's
        # mean and totals come from column 0 and are shared by all features.
        centered = y_sorted - y_sorted[:, 0].mean()  # keeps the sums of squares well conditioned
        left_sum = np.cumsum(centered, axis = 0)[:-1]
        total_sum, total_sq = centered[:, 0].sum(), np.sum(centered[:, 0] ** 2)
        n_left = np.arange(1, n, dtype = float)[:, None]
        # Sum of squared errors around each child's mean is sum(y^2) - (sum y)^2 / n_child,
        # and the two children's sum(y^2) add up to the node's.
        sse = total_sq - left_sum ** 2 / n_left - (total_sum - left_sum) ** 2 / (n - n_left)
        return sse / n

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _output_dtype(self):
        return float
