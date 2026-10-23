import numpy as np
import pytest

from axiom.preprocessing.encoders import LabelEncoder, OneHotEncoder


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
