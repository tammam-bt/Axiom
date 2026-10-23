import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from axiom.linear.linear_regression import LinearRegression
from axiom.linear.logistic_regression import LogisticRegression


def linear_data(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 3.0 + rng.normal(0, 0.1, size=200)
    return X, y


def scaled_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    return (Xtr - mu) / sd, (Xte - mu) / sd, ytr, yte


# ---- LinearRegression -------------------------------------------------------

def test_linear_regression_coefficients_match_lstsq():
    X, y = linear_data()
    model = LinearRegression(alpha=0.1)
    model.fit(X, y, epochs=1000)
    A = np.c_[X, np.ones(len(X))]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    np.testing.assert_allclose(model.weights, coef[:-1], atol=1e-2)
    assert abs(model.bias - coef[-1]) < 1e-2


def test_linear_regression_gradient_is_the_batch_mean():
    X, y = linear_data()
    model = LinearRegression()
    model.weights, model.bias = np.array([0.3, -0.1, 0.2]), 0.5
    error = X @ model.weights + model.bias - y
    dw, db = model.compute_gradient(X, y)
    np.testing.assert_allclose(dw, X.T @ error / len(X))
    assert db == np.sum(error) / len(X)


def test_linear_regression_prints_only_when_verbose(capsys):
    X, y = linear_data()
    LinearRegression().fit(X, y, epochs=5)
    assert capsys.readouterr().out == ""
    LinearRegression(verbose=True).fit(X, y, epochs=5)
    assert "cost" in capsys.readouterr().out


# ---- LogisticRegression -----------------------------------------------------

def test_logistic_regression_accuracy_on_breast_cancer():
    Xtr, Xte, ytr, yte = scaled_breast_cancer()
    model = LogisticRegression(alpha=0.1)
    model.fit(Xtr, ytr, epochs=1000)
    assert np.mean(model.predict(Xte) == yte) >= 0.9


def test_logistic_predict_returns_labels_and_predict_proba_returns_probabilities():
    Xtr, Xte, ytr, _ = scaled_breast_cancer()
    model = LogisticRegression(alpha=0.1)
    model.fit(Xtr, ytr, epochs=200)
    proba = model.predict_proba(Xte)
    labels = model.predict(Xte)
    assert proba.shape == labels.shape == (len(Xte),)
    assert np.all((proba > 0) & (proba < 1))
    assert set(np.unique(labels)) <= {0, 1}
    np.testing.assert_array_equal(labels, (proba >= 0.5).astype(int))


def test_logistic_gradient_weights_the_whole_error_and_averages_the_bias_term():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])  # 2 positives, 8 negatives
    model = LogisticRegression()
    model.w, model.b = np.zeros(2), 0.0  # every prediction is 0.5
    m = len(y)
    class_weight = np.where(y == 1, m / (2 * 2), m / (2 * 8))
    error = class_weight * (0.5 - y)
    dw, db = model.calculate_gradient(X, y)
    np.testing.assert_allclose(dw, X.T @ error / m)
    np.testing.assert_allclose(db, np.sum(error) / m)


def test_logistic_regression_prints_only_when_verbose(capsys):
    Xtr, _, ytr, _ = scaled_breast_cancer()
    LogisticRegression().fit(Xtr, ytr, epochs=5)
    assert capsys.readouterr().out == ""
    LogisticRegression(verbose=True).fit(Xtr, ytr, epochs=5)
    assert "cost" in capsys.readouterr().out


def test_logistic_logging_keeps_one_weight_snapshot_per_epoch():
    Xtr, _, ytr, _ = scaled_breast_cancer()
    log = LogisticRegression(alpha=0.1).fit(Xtr, ytr, epochs=3, with_logging=True)
    assert not np.array_equal(log["Weights"][0], log["Weights"][2])


def test_logistic_regression_refit_starts_from_scratch():
    X, _, y, _ = scaled_breast_cancer()
    model = LogisticRegression(alpha=0.1, b=0.5)
    model.fit(X, y, epochs=50)
    w1, b1 = model.w.copy(), model.b
    model.fit(X, y, epochs=50)
    assert np.array_equal(model.w, w1)
    assert model.b == b1
