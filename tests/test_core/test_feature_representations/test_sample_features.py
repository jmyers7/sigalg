import pandas as pd
import pytest

import sigalg as sa


class TestBasicConstruction:

    def test_construction_with_list(self):
        features = [1, 2, 3]
        sf = sa.SampleFeatures(features)
        expected_series = pd.Series(features, name="omega", index=["X0", "X1", "X2"])
        pd.testing.assert_series_equal(sf.to_pandas(), expected_series)

    def test_construction_with_dict(self):
        features = {"X0": 1, "X1": 2, "X2": 3}
        sf = sa.SampleFeatures(features)
        expected_series = pd.Series(features, name="omega", index=["X0", "X1", "X2"])
        pd.testing.assert_series_equal(sf.to_pandas(), expected_series)

    def test_construction_with_series(self):
        features = pd.Series([1, 2, 3], name="s", index=["a", "b", "c"])
        sf = sa.SampleFeatures(features)
        expected_series = pd.Series([1, 2, 3], name="s", index=["a", "b", "c"])
        pd.testing.assert_series_equal(sf.to_pandas(), expected_series)

    def test_construction_with_single_value(self):
        features = [42]
        sf = sa.SampleFeatures(features)
        expected_series = pd.Series([42], name="omega", index=["X"])
        pd.testing.assert_series_equal(sf.to_pandas(), expected_series)

    def test_construction_with_custom_sample_index(self):
        features = [1, 2, 3]
        sf = sa.SampleFeatures(features, sample_index="omega5")
        expected_series = pd.Series([1, 2, 3], name="omega5", index=["X0", "X1", "X2"])
        pd.testing.assert_series_equal(sf.to_pandas(), expected_series)

    def test_construction_with_default_sample_index(self):
        features = [1, 2, 3]
        sf = sa.SampleFeatures(features)
        expected_series = pd.Series([1, 2, 3], name="omega", index=["X0", "X1", "X2"])
        pd.testing.assert_series_equal(sf.to_pandas(), expected_series)

    def test_construction_with_custom_feature_index(self):
        features = [1, 2, 3]
        feature_index = ["feature_a", "feature_b", "feature_c"]
        sf = sa.SampleFeatures(features, feature_index=feature_index)
        expected_series = pd.Series([1, 2, 3], name="omega", index=feature_index)
        pd.testing.assert_series_equal(sf.to_pandas(), expected_series)

    def test_construction_with_default_feature_index_single_feature(self):
        features = [42]
        sf = sa.SampleFeatures(features)
        assert list(sf.feature_index) == ["X"]

    def test_construction_with_custom_feature_prefix(self):
        features = [1, 2, 3]
        sf = sa.SampleFeatures(features, feature_prefix="Y")

        assert list(sf.feature_index) == ["Y0", "Y1", "Y2"]

    def test_construction_with_custom_initial_feature_index(self):
        features = [1, 2, 3]
        sf = sa.SampleFeatures(features, initial_feature_index=5)

        assert list(sf.feature_index) == ["X5", "X6", "X7"]

    def test_construction_with_custom_prefix_and_initial_index(self):
        features = [1, 2, 3]
        sf = sa.SampleFeatures(
            features, feature_prefix="Feature", initial_feature_index=10
        )

        assert list(sf.feature_index) == ["Feature10", "Feature11", "Feature12"]

    def test_construction_overwrite_does_not_overwrite_user_provided_feature_index(
        self,
    ):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"])
        sf = sa.SampleFeatures(features, overwrite_default_feature_index=True)
        assert list(sf.feature_index) == ["a", "b", "c"]

    def test_construction_overwrite_does_not_overwite_user_provided_sample_index(
        self,
    ):
        features = pd.Series([1, 2, 3], name="my_sample")
        sf = sa.SampleFeatures(features, overwrite_default_sample_index=True)
        assert sf.sample_index == "my_sample"

    def test_construction_copies_input_series(self):
        features = pd.Series([1, 2, 3])
        sf = sa.SampleFeatures(features)
        features.iloc[0] = 999
        assert sf._data["X0"] == 1

    def test_construction_with_complex_sample_index(self):
        features = [1, 2, 3]
        sample_index = ("tuple", "index")
        sf = sa.SampleFeatures(features, sample_index=sample_index)

        assert sf.sample_index == ("tuple", "index")

    def test_construction_with_numeric_sample_index(self):
        features = [1, 2, 3]
        sf = sa.SampleFeatures(features, sample_index=42)

        assert sf.sample_index == 42

    def test_construction_with_mixed_type_features(self):
        features = [1, "two", 3.0]
        sf = sa.SampleFeatures(features)

        assert len(sf._data) == 3
        assert sf._data["X0"] == 1
        assert sf._data["X1"] == "two"
        assert sf._data["X2"] == 3.0

    def test_construction_with_boolean_features(self):
        features = [True, False, True]
        sf = sa.SampleFeatures(features)

        assert sf._data.tolist() == [True, False, True]
        assert sf._data.dtype == bool

    def test_construction_with_series_preserves_dtype(self):
        features = pd.Series([1, 2, 3], dtype="int32")
        sf = sa.SampleFeatures(features)

        assert sf._data.dtype == "int32"

    def test_construction_with_series_and_dtype_override(self):
        features = pd.Series([1, 2, 3], dtype="int32")
        sf = sa.SampleFeatures(features, dtype="float64")

        assert sf._data.dtype == "float64"

    def test_construction_all_parameters(self):
        features = [10, 20, 30, 40]
        sf = sa.SampleFeatures(
            features=features,
            sample_index="sample_alpha",
            feature_index=["a", "b", "c", "d"],
            initial_feature_index=99,
            feature_prefix="Z",
            dtype=float,
        )

        assert sf.sample_index == "sample_alpha"
        assert list(sf.feature_index) == ["a", "b", "c", "d"]
        assert sf._data.dtype == float
        assert sf._data.tolist() == [10.0, 20.0, 30.0, 40.0]


class TestGetItem:

    @pytest.fixture
    def sample_features(self):
        features = [10, 20, 30]
        return sa.SampleFeatures(
            features=features, sample_index="s1", feature_index=["F0", "F1", "F2"]
        )

    def test_getitem_with_one_string_index(self, sample_features):
        val = sample_features["F1"]
        assert val == 20

    def test_getitem_with_list_of_string_indices(self, sample_features):
        vals = sample_features[["F0", "F2"]]
        expected_series = pd.Series([10, 30], name="s1", index=["F0", "F2"])
        pd.testing.assert_series_equal(vals, expected_series)


class TestFeatureAt:

    @pytest.fixture
    def sample_features(self):
        features = [10, 20, 30, 40, 50]
        return sa.SampleFeatures(
            features=features,
            sample_index="s1",
            feature_index=["F0", "F1", "F2", "F3", "F4"],
        )

    def test_feature_at_with_integer_index(self, sample_features):
        val = sample_features.feature_at[2]
        assert val == 30

    def test_feature_at_with_slice(self, sample_features):
        vals = sample_features.feature_at[1:4]
        expected_series = pd.Series([20, 30, 40], name="s1", index=["F1", "F2", "F3"])
        pd.testing.assert_series_equal(vals, expected_series)

    def test_feature_at_with_list_of_integer_indices(self, sample_features):
        vals = sample_features.feature_at[[0, 3, 4]]
        expected_series = pd.Series([10, 40, 50], name="s1", index=["F0", "F3", "F4"])
        pd.testing.assert_series_equal(vals, expected_series)
