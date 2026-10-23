import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

from axiom.trees.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from axiom.trees.gains import entropy, gini, information_gain, variance

# The 32-point set from examples/decision_tree_test.py: four jittered quadrants,
# class 1 when exactly one coordinate is above 0.5.
JITTERED_XOR_X = np.array([
    [0.10, 0.12], [0.18, 0.20], [0.22, 0.28], [0.30, 0.35],
    [0.40, 0.18], [0.12, 0.42], [0.33, 0.10], [0.45, 0.45],
    [0.08, 0.62], [0.15, 0.75], [0.25, 0.88], [0.35, 0.70],
    [0.42, 0.58], [0.20, 0.95], [0.47, 0.82], [0.05, 0.55],
    [0.55, 0.05], [0.62, 0.22], [0.70, 0.40], [0.82, 0.12],
    [0.95, 0.30], [0.78, 0.48], [0.58, 0.35], [0.88, 0.18],
    [0.55, 0.60], [0.63, 0.72], [0.74, 0.84], [0.86, 0.66],
    [0.92, 0.94], [0.68, 0.55], [0.97, 0.58], [0.79, 0.90],
])
JITTERED_XOR_Y = np.repeat([0, 1, 1, 0], 8)

# The four XOR corners, eight copies each: 32 points.
CORNER_XOR_X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (8, 1))
CORNER_XOR_Y = np.tile(np.array([0, 1, 1, 0]), 8)

CRITERIA = ["gini", "entropy"]


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# ---- gains.py ---------------------------------------------------------------

def test_gini_of_balanced_binary_labels_is_one_half():
    assert gini([0, 0, 1, 1]) == 0.5


def test_entropy_of_balanced_binary_labels_is_one_bit():
    assert entropy([0, 0, 1, 1]) == 1.0


def test_gini_of_a_pure_node_is_zero():
    assert gini([2, 2, 2]) == 0


def test_impurities_count_any_number_of_classes_and_any_label_type():
    three_balanced = ["a", "b", "c", "a", "b", "c"]
    assert gini(three_balanced) == pytest.approx(2 / 3)
    assert entropy(three_balanced) == pytest.approx(np.log2(3))
    assert entropy([7, 7, 7]) == 0


def test_variance_is_the_population_variance():
    assert variance([1.0, 3.0]) == 1.0
    assert variance([5.0, 5.0, 5.0]) == 0


def test_information_gain_of_a_perfect_and_a_useless_split():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([True, True, False, False])
    useless = np.array([True, False, True, False])
    assert information_gain(y, perfect, entropy) == 1.0
    assert information_gain(y, perfect, gini) == 0.5
    assert information_gain(y, useless, entropy) == 0.0


def test_information_gain_with_variance_for_regression_targets():
    y = np.array([1.0, 1.0, 3.0, 3.0])
    assert information_gain(y, np.array([True, True, False, False]), variance) == 1.0


# ---- DecisionTreeClassifier -------------------------------------------------

@pytest.mark.parametrize("criterion", CRITERIA)
def test_corner_xor_reaches_train_accuracy_1_at_depth_3(criterion):
    # Every first split of this set has zero gain, so it only works if the tree
    # still splits an impure node when the best gain is exactly 0.
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=3).fit(CORNER_XOR_X, CORNER_XOR_Y)
    assert accuracy(CORNER_XOR_Y, tree.predict(CORNER_XOR_X)) == 1.0


@pytest.mark.parametrize("criterion", CRITERIA)
def test_jittered_xor_matches_sklearn_at_depth_3_and_separates_when_unlimited(criterion):
    # A greedy tree cannot see that a zero-gain split at 0.5 pays off one level
    # down, so at depth 3 it chips small pure groups off the edges instead.
    # scikit-learn does the same, so match it rather than expect 1.0.
    sk = SkDecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=0)
    sk_acc = accuracy(JITTERED_XOR_Y, sk.fit(JITTERED_XOR_X, JITTERED_XOR_Y).predict(JITTERED_XOR_X))
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=3).fit(JITTERED_XOR_X, JITTERED_XOR_Y)
    assert abs(accuracy(JITTERED_XOR_Y, tree.predict(JITTERED_XOR_X)) - sk_acc) <= 0.03

    deep = DecisionTreeClassifier(criterion=criterion, max_depth=None).fit(JITTERED_XOR_X, JITTERED_XOR_Y)
    assert accuracy(JITTERED_XOR_Y, deep.predict(JITTERED_XOR_X)) == 1.0


@pytest.mark.parametrize("criterion", CRITERIA)
def test_a_threshold_of_zero_still_sends_x_le_0_left(criterion):
    X = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([0, 0, 1])
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    assert tree.root.threshold == 0.5
    np.testing.assert_array_equal(tree.predict(X), [0, 0, 1])

    # The split between -1 and 0 is the one the old `if node.threshold and ...` broke.
    X2 = np.array([[-1.0], [1.0]])
    tree2 = DecisionTreeClassifier(criterion=criterion).fit(X2, np.array([0, 1]))
    assert tree2.root.threshold == 0.0
    np.testing.assert_array_equal(tree2.predict(np.array([[-1.0], [0.0], [1.0]])), [0, 0, 1])


def test_every_midpoint_is_a_candidate_threshold():
    # The best cut sits between the first two distinct values. The old search
    # skipped the first and last `min_samples_split` values and never tried it.
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([1, 0, 0, 0, 0, 0])
    tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert tree.root.threshold == 0.5
    np.testing.assert_array_equal(tree.predict(X), y)


@pytest.mark.parametrize("max_depth", [0, 1, 2, 3, 4])
def test_fitted_depth_never_exceeds_max_depth(max_depth):
    X, y = load_breast_cancer(return_X_y=True)
    tree = DecisionTreeClassifier(max_depth=max_depth).fit(X, y)
    assert tree.get_depth() <= max_depth
    assert tree.get_depth() == max_depth  # this data is not separable at depth 4


def test_unknown_criterion_raises_value_error():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="mse")


def test_fit_and_predict_print_nothing_unless_verbose(capsys):
    X, y = load_breast_cancer(return_X_y=True)
    DecisionTreeClassifier(max_depth=2).fit(X, y).predict(X)
    assert capsys.readouterr().out == ""


def test_three_class_gini_tree_on_separable_blobs():
    # cluster_std=0.6 keeps the three blobs linearly separable (at 1.0 they overlap).
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    tree = DecisionTreeClassifier(criterion="gini", max_depth=3).fit(Xtr, ytr)
    assert set(np.unique(tree.predict(Xte))) == {0, 1, 2}
    assert accuracy(yte, tree.predict(Xte)) >= 0.95


def test_string_labels_come_back_as_strings():
    X, y = make_blobs(n_samples=90, centers=3, cluster_std=0.5, random_state=1)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    tree = DecisionTreeClassifier(criterion="gini").fit(X, names)
    np.testing.assert_array_equal(tree.predict(X), names)


@pytest.mark.parametrize("criterion", CRITERIA)
@pytest.mark.parametrize("max_depth", [2, 3, 5])
def test_breast_cancer_accuracy_within_003_of_sklearn(criterion, max_depth):
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    ours = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth).fit(Xtr, ytr)
    sk = SkDecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0).fit(Xtr, ytr)
    ours_acc, sk_acc = accuracy(yte, ours.predict(Xte)), accuracy(yte, sk.predict(Xte))
    assert abs(ours_acc - sk_acc) <= 0.03, (ours_acc, sk_acc)


# ---- DecisionTreeRegressor --------------------------------------------------

def test_regressor_fits_a_step_function_exactly():
    X = np.linspace(0, 10, 101).reshape(-1, 1)
    y = np.select([X[:, 0] < 2.5, X[:, 0] < 6.0], [1.0, -2.0], default=4.0)
    tree = DecisionTreeRegressor(max_depth=2).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)
    assert tree.get_depth() == 2


def test_regressor_leaf_is_the_mean_of_its_samples():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 10.0, 14.0])
    tree = DecisionTreeRegressor(max_depth=1).fit(X, y)
    np.testing.assert_allclose(tree.predict(X), [2.0, 2.0, 12.0, 12.0])
