import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from axiom.trees.decision_tree import DecisionTreeRegressor
from axiom.trees.gbm import GBM, GBMClassifier, GBMRegressor


def wavy_regression(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.1, size=n)
    return X, y


def breast_cancer_split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


def test_regressor_training_loss_decreases_every_round():
    X, y = wavy_regression()
    model = GBMRegressor(n_estimators=30).fit(X, y)
    loss = np.asarray(model.train_loss_)
    assert loss.shape == (31,)  # the constant start, then one entry per tree
    assert loss[0] == pytest.approx(np.var(y))
    assert np.all(np.diff(loss) < 0)


def test_classifier_training_log_loss_decreases_every_round():
    Xtr, _, ytr, _ = breast_cancer_split()
    model = GBMClassifier(n_estimators=30).fit(Xtr, ytr)
    loss = np.asarray(model.train_loss_)
    assert loss.shape == (31,)
    assert np.all(np.diff(loss) < 0)


def test_one_round_at_learning_rate_1_equals_a_single_tree():
    X, y = wavy_regression()
    X_new, _ = wavy_regression(n=50, seed=1)
    gbm = GBMRegressor(n_estimators=1, learning_rate=1.0, max_depth=3).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=3).fit(X, y)
    np.testing.assert_allclose(gbm.predict(X_new), tree.predict(X_new), rtol=0, atol=1e-10)


def test_regressor_beats_the_mean_baseline_on_held_out_data():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    pred = GBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3).fit(Xtr, ytr).predict(Xte)
    gbm_mse = np.mean((yte - pred) ** 2)
    baseline_mse = np.mean((yte - ytr.mean()) ** 2)
    assert gbm_mse < baseline_mse


def test_classifier_reaches_093_on_breast_cancer():
    Xtr, Xte, ytr, yte = breast_cancer_split()
    model = GBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    assert np.mean(model.predict(Xte) == yte) >= 0.93


def test_classifier_starts_from_the_log_odds_and_fits_y_minus_sigmoid():
    Xtr, _, ytr, _ = breast_cancer_split()
    model = GBMClassifier(n_estimators=1, learning_rate=1.0).fit(Xtr, ytr)
    p = ytr.mean()
    assert model.init_ == pytest.approx(np.log(p / (1 - p)))
    # The first tree is fitted to y - sigmoid(F_0) = y - p. Its leaves then take a
    # Newton step, sum(y - s) / sum(s(1 - s)), which with s = p everywhere is the
    # mean residual divided by p(1 - p).
    tree = DecisionTreeRegressor(max_depth=3).fit(Xtr, ytr - p)
    expected = model.init_ + tree.predict(Xtr) / (p * (1 - p))
    np.testing.assert_allclose(model.decision_function(Xtr), expected, rtol=1e-9)


def test_classifier_probabilities_and_original_labels():
    Xtr, Xte, ytr, _ = breast_cancer_split()
    names = np.array(["malignant", "benign"])
    model = GBMClassifier(n_estimators=20).fit(Xtr, names[ytr])
    proba = model.predict_proba(Xte)
    assert proba.shape == (len(Xte),)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_array_equal(model.predict(Xte), np.where(proba >= 0.5, model.classes_[1], model.classes_[0]))
    assert set(model.predict(Xte)) <= {"malignant", "benign"}


def test_classifier_rejects_more_than_two_classes():
    X = np.arange(9, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError):
        GBMClassifier().fit(X, np.array([0, 1, 2] * 3))


def test_gbm_promised_in_the_docs_is_the_regressor():
    assert GBM is GBMRegressor
    X, y = wavy_regression()
    assert GBM(n_estimators=5, learning_rate=0.1, max_depth=3).fit(X, y).predict(X).shape == (len(X),)
