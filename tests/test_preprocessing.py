import numpy as np
import pytest
from sklearn.preprocessing import MaxAbsScaler as SkMaxAbsScaler
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from axiom.preprocessing.encoders import LabelEncoder, OneHotEncoder
from axiom.preprocessing.scalers import (
    MinMaxScaler,
    StandardScaler,
    max_abs_scale,
    min_max_scale,
    standard_scale,
)


def test_label_encoder_round_trip():
    labels = ["cat", "dog", "cat", "bird"]
    enc = LabelEncoder()
    codes = enc.fit_transform(labels)
    np.testing.assert_array_equal(codes, [1, 2, 1, 0])
    np.testing.assert_array_equal(enc.classes_, ["bird", "cat", "dog"])
    np.testing.assert_array_equal(enc.inverse_transform(codes), labels)
    assert enc.transform([]).shape == enc.inverse_transform([]).shape == (0,)


def test_label_encoder_unknown_label_raises():
    enc = LabelEncoder().fit(["cat", "dog"])
    with pytest.raises(ValueError, match="fish"):
        enc.transform(["cat", "fish"])


def test_label_encoder_unknown_code_raises():
    enc = LabelEncoder().fit(["cat", "dog"])
    with pytest.raises(ValueError):
        enc.inverse_transform([0, 2])


def test_one_hot_round_trip_on_a_1d_input():
    y = np.array([1, 2, 1, 0])
    ohe = OneHotEncoder()
    onehot = ohe.fit_transform(y)
    np.testing.assert_array_equal(onehot, [[0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(ohe.inverse_transform(onehot), y)


def test_one_hot_on_a_column_vector_keeps_its_shape():
    y = np.array([0, 2, 1, 2, 0]).reshape(-1, 1)
    ohe = OneHotEncoder()
    assert ohe.fit_transform(y).shape == (5, 3)
    np.testing.assert_array_equal(ohe.inverse_transform(ohe.transform(y)), y)


def test_one_hot_round_trip_on_several_columns():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"], ["green", "S"]])
    ohe = OneHotEncoder().fit(X)
    onehot = ohe.transform(X)
    assert onehot.shape == (4, 3 + 2)
    np.testing.assert_array_equal(onehot.sum(axis=1), [2, 2, 2, 2])
    np.testing.assert_array_equal(ohe.inverse_transform(onehot), X)


def test_one_hot_unknown_category_raises():
    ohe = OneHotEncoder().fit(np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        ohe.transform(np.array([1, 3]))


def test_one_hot_inverse_rejects_rows_that_are_not_one_hot():
    ohe = OneHotEncoder().fit(np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        ohe.inverse_transform(np.array([[1.0, 1.0, 0.0]]))


# ---- scalers ----------------------------------------------------------------

def feature_matrix(n=60, seed=0):
    # Four features on very different scales and offsets, so whole-array
    # statistics would give visibly different answers from per-feature ones.
    rng = np.random.default_rng(seed)
    return rng.normal(loc=[0.0, 5.0, -3.0, 100.0], scale=[1.0, 0.1, 10.0, 50.0], size=(n, 4))


def with_constant_columns(seed=0):
    X = feature_matrix(n=20, seed=seed)
    X[:, 1] = 3.3  # constant, non-zero
    X[:, 2] = 0.0  # constant zero
    return X


SCALE_FUNCTIONS = [
    (min_max_scale, SkMinMaxScaler),
    (standard_scale, SkStandardScaler),
    (max_abs_scale, SkMaxAbsScaler),
]
SCALER_CLASSES = [(MinMaxScaler, SkMinMaxScaler), (StandardScaler, SkStandardScaler)]


@pytest.mark.parametrize("scale, sk_scaler", SCALE_FUNCTIONS)
def test_scale_functions_match_sklearn_per_feature(scale, sk_scaler):
    X = feature_matrix()
    np.testing.assert_allclose(scale(X), sk_scaler().fit_transform(X), rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale, sk_scaler", SCALE_FUNCTIONS)
def test_scale_functions_handle_constant_columns(scale, sk_scaler):
    X = with_constant_columns()
    out = scale(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, sk_scaler().fit_transform(X), rtol=0, atol=1e-12)
    if scale is not max_abs_scale:  # max-abs keeps the sign and size: 3.3 / 3.3 = 1
        np.testing.assert_array_equal(out[:, 1:3], 0.0)


def test_scale_functions_on_a_1d_array():
    np.testing.assert_allclose(min_max_scale([1.0, 2.0, 3.0]), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(max_abs_scale([-4.0, 2.0]), [-1.0, 0.5])
    np.testing.assert_allclose(standard_scale([1.0, 3.0]), [-1.0, 1.0])


@pytest.mark.parametrize("ours, sk_scaler", SCALER_CLASSES)
def test_scaler_classes_match_sklearn_and_transform_with_the_train_statistics(ours, sk_scaler):
    X = feature_matrix()
    X_train, X_test = X[:40], X[40:]
    scaler, sk = ours().fit(X_train), sk_scaler().fit(X_train)
    np.testing.assert_allclose(scaler.transform(X_train), sk.transform(X_train), rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaler.transform(X_test), sk.transform(X_test), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ours().fit_transform(X_train), sk.transform(X_train), rtol=0, atol=1e-12)
    # Refitting on the test rows would give different numbers.
    assert not np.allclose(scaler.transform(X_test), ours().fit_transform(X_test))


@pytest.mark.parametrize("ours, sk_scaler", SCALER_CLASSES)
def test_scaler_classes_handle_constant_columns(ours, sk_scaler):
    X = with_constant_columns()
    scaler = ours().fit(X)
    out = scaler.transform(X)
    np.testing.assert_array_equal(out[:, 1:3], 0.0)
    np.testing.assert_allclose(out, sk_scaler().fit_transform(X), rtol=0, atol=1e-12)
    np.testing.assert_allclose(scaler.inverse_transform(out), X, rtol=0, atol=1e-12)


@pytest.mark.parametrize("ours", [MinMaxScaler, StandardScaler])
def test_scaler_inverse_transform_round_trips(ours):
    X = feature_matrix()
    scaler = ours().fit(X[:40])
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X, rtol=0, atol=1e-12)


def test_scaler_rejects_a_different_number_of_features():
    scaler = StandardScaler().fit(feature_matrix())
    with pytest.raises(ValueError):
        scaler.transform(feature_matrix()[:, :3])
