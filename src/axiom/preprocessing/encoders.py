import numpy as np


def _codes(categories, values, what):
    """Index of each value in the sorted `categories`; ValueError for unseen values."""
    values = np.asarray(values)
    if values.size == 0:
        return np.zeros(0, dtype=int)
    try:
        codes = np.searchsorted(categories, values)
        found = codes < len(categories)
        found[found] = categories[codes[found]] == values[found]
    except TypeError as error:  # e.g. ints looked up among strings
        raise ValueError(f"{what} of a different type than the fitted ones: {error}") from None
    if not found.all():
        unknown = np.unique(values[~found]).tolist()
        raise ValueError(f"unknown {what}: {unknown}")
    return codes


class LabelEncoder:
    """Maps labels to the integers 0..n_classes-1, in sorted label order.

    enc = LabelEncoder()
    enc.fit_transform(["cat", "dog", "cat", "bird"])   # -> [1, 2, 1, 0]
    enc.inverse_transform([1, 2])                       # -> ["cat", "dog"]
    """

    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("LabelEncoder expects a 1-D array of labels")
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        self._check_fitted()
        return _codes(self.classes_, np.asarray(y).ravel(), "labels")

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, codes):
        self._check_fitted()
        codes = np.asarray(codes)
        if codes.size == 0:
            codes = codes.astype(int)  # np.asarray([]) is float, which cannot index
        elif (not np.issubdtype(codes.dtype, np.integer)
              or codes.min() < 0 or codes.max() >= len(self.classes_)):
            raise ValueError(f"codes must be integers in [0, {len(self.classes_) - 1}]")
        return self.classes_[codes]

    def _check_fitted(self):
        if self.classes_ is None:
            raise RuntimeError("call fit before transform")


class OneHotEncoder:
    """One column of 0/1 per category, for each input column.

    A 1-D input is treated as a single column, and inverse_transform gives a 1-D
    array back. Categories are sorted; an unseen category raises ValueError.

    ohe = OneHotEncoder()
    ohe.fit_transform([1, 2, 1, 0])   # -> [[0,1,0], [0,0,1], [0,1,0], [1,0,0]]
    """

    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        X, self._fitted_1d = self._as_columns(X)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        self._check_fitted()
        X, _ = self._as_columns(X)
        if X.shape[1] != len(self.categories_):
            raise ValueError(f"X has {X.shape[1]} columns, the encoder was fitted on {len(self.categories_)}")
        blocks = []
        for j, categories in enumerate(self.categories_):
            codes = _codes(categories, X[:, j], f"categories in column {j}")
            block = np.zeros((len(X), len(categories)))
            block[np.arange(len(X)), codes] = 1.0
            blocks.append(block)
        return np.hstack(blocks)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, onehot):
        self._check_fitted()
        onehot = np.asarray(onehot, dtype=float)
        widths = [len(c) for c in self.categories_]
        if onehot.ndim != 2 or onehot.shape[1] != sum(widths):
            raise ValueError(f"expected a 2-D array with {sum(widths)} columns")
        columns, start = [], 0
        for categories, width in zip(self.categories_, widths):
            block = onehot[:, start:start + width]
            start += width
            if not (np.isin(block, (0.0, 1.0)).all() and (block.sum(axis=1) == 1).all()):
                raise ValueError("each row needs exactly one 1 per encoded column")
            columns.append(categories[np.argmax(block, axis=1)])
        dtypes = {c.dtype for c in columns}
        out = np.empty((len(onehot), len(columns)), dtype=dtypes.pop() if len(dtypes) == 1 else object)
        for j, column in enumerate(columns):
            out[:, j] = column
        return out[:, 0] if self._fitted_1d else out

    @staticmethod
    def _as_columns(X):
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1), True
        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects a 1-D or 2-D array")
        return X, False

    def _check_fitted(self):
        if self.categories_ is None:
            raise RuntimeError("call fit before transform")
