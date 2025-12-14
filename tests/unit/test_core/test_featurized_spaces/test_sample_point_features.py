import pandas as pd
import pytest

import sigalg as sa

pytestmark = pytest.mark.unit


class TestConstructor:
    def test_basic_construction(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        expected_series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        pd.testing.assert_series_equal(spf.values, expected_series)


class TestProperties:

    def test_name_property(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test_name")
        spf = sa.SamplePointFeatures(name="test_name", values=features)
        assert spf.name == "test_name"

    def test_feature_embedding_property_initially_none(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        assert spf.feature_embedding is None


class TestFeatureAt:

    def test_feature_at_single_index(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        assert spf.feature_at[0] == 1
        assert spf.feature_at[1] == 2
        assert spf.feature_at[2] == 3

    def test_feature_at_negative_index(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        assert spf.feature_at[-1] == 3
        assert spf.feature_at[-2] == 2

    def test_feature_at_slice(self):
        features = pd.Series(
            [1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"], name="omega"
        )
        spf = sa.SamplePointFeatures(name="omega", values=features)
        result = spf.feature_at[1:4]
        expected_series = pd.Series([2, 3, 4], index=["b", "c", "d"], name="omega")
        pd.testing.assert_series_equal(result, expected_series)

    def test_feature_at_list(self):
        features = pd.Series(
            [1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"], name="omega"
        )
        spf = sa.SamplePointFeatures(name="omega", values=features)
        result = spf.feature_at[[0, 2, 4]]
        expected_series = pd.Series([1, 3, 5], index=["a", "c", "e"], name="omega")
        pd.testing.assert_series_equal(result, expected_series)


class TestIteration:
    def test_iter(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        values = list(spf)
        assert values == [1, 2, 3]

    def test_iter_empty(self):
        features = pd.Series([], dtype=int, name="empty")
        spf = sa.SamplePointFeatures(name="empty", values=features)
        values = list(spf)
        assert values == []


class TestLen:
    def test_len(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        assert len(spf) == 3

    def test_len_single_feature(self):
        features = pd.Series([42], index=["x"], name="single")
        spf = sa.SamplePointFeatures(name="single", values=features)
        assert len(spf) == 1

    def test_len_empty(self):
        features = pd.Series([], dtype=int, name="empty")
        spf = sa.SamplePointFeatures(name="empty", values=features)
        assert len(spf) == 0


class TestSum:
    def test_sum_integers(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        assert spf.sum() == 6

    def test_sum_floats(self):
        features = pd.Series([1.5, 2.5, 3.0], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        assert abs(spf.sum() - 7.0) < 1e-10

    def test_sum_single_value(self):
        features = pd.Series([42], index=["x"], name="single")
        spf = sa.SamplePointFeatures(name="single", values=features)
        assert spf.sum() == 42

    def test_sum_empty(self):
        features = pd.Series([], dtype=float, name="empty")
        spf = sa.SamplePointFeatures(name="empty", values=features)
        assert spf.sum() == 0


class TestValidation:
    def test_name_not_hashable(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        with pytest.raises(TypeError, match="name must be a Hashable"):
            sa.SamplePointFeatures(name=["not_hashable"], values=features)

    def test_features_not_series(self):
        with pytest.raises(TypeError, match="values must be a pandas Series"):
            sa.SamplePointFeatures(name="omega", values=[1, 2, 3])

    def test_features_name_mismatch(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="wrong_name")
        with pytest.raises(ValueError, match="values.name must match the given name"):
            sa.SamplePointFeatures(name="omega", values=features)

    def test_features_name_none_mismatch(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name=None)
        with pytest.raises(ValueError, match="values.name must match the given name"):
            sa.SamplePointFeatures(name="omega", values=features)


class TestEquality:
    def test_equal_spf(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf1 = sa.SamplePointFeatures(name="omega", values=features)
        spf2 = sa.SamplePointFeatures(name="omega", values=features)
        assert spf1 == spf2

    def test_equal_spf_different_objects_same_values(self):
        features1 = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        features2 = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf1 = sa.SamplePointFeatures(name="omega", values=features1)
        spf2 = sa.SamplePointFeatures(name="omega", values=features2)
        assert spf1 == spf2

    def test_not_equal_different_values(self):
        features1 = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        features2 = pd.Series([1, 2, 4], index=["a", "b", "c"], name="omega")
        spf1 = sa.SamplePointFeatures(name="omega", values=features1)
        spf2 = sa.SamplePointFeatures(name="omega", values=features2)
        assert spf1 != spf2

    def test_not_equal_different_names(self):
        features1 = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega0")
        features2 = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega1")
        spf1 = sa.SamplePointFeatures(name="omega0", values=features1)
        spf2 = sa.SamplePointFeatures(name="omega1", values=features2)
        assert spf1 != spf2

    def test_not_equal_different_indices(self):
        features1 = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        features2 = pd.Series([1, 2, 3], index=["x", "y", "z"], name="omega")
        spf1 = sa.SamplePointFeatures(name="omega", values=features1)
        spf2 = sa.SamplePointFeatures(name="omega", values=features2)
        assert spf1 != spf2

    def test_not_equal_different_type(self):
        features = pd.Series([1, 2, 3], index=["a", "b", "c"], name="omega")
        spf = sa.SamplePointFeatures(name="omega", values=features)
        assert spf != "not sample point features"
        assert spf != 42
        assert spf is not None

    def test_equal_from_feature_embedding(self):
        values = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf1 = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=0, feature_embedding=feature_embedding
        )
        spf2 = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=0, feature_embedding=feature_embedding
        )
        assert spf1 == spf2

    def test_not_equal_from_feature_embedding_different_rows(self):
        values = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf1 = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=0, feature_embedding=feature_embedding
        )
        spf2 = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=1, feature_embedding=feature_embedding
        )
        assert spf1 != spf2


class TestFromFeatureEmbedding:

    def test_from_feature_embedding_basic(self):
        values = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=0, feature_embedding=feature_embedding
        )
        expected_series = pd.Series([1, 2], index=pd.Index([0, 1]), name=0)
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == 0
        assert spf.feature_embedding is feature_embedding

    def test_from_feature_embedding_with_index(self):
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]], columns=pd.Index([0, 1], name="xyz")
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=0, feature_embedding=feature_embedding
        )
        expected_series = pd.Series([1, 2], index=pd.Index([0, 1], name="xyz"), name=0)
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == 0
        assert spf.feature_embedding is feature_embedding

    def test_from_feature_embedding_second_row(self):
        values = pd.DataFrame([[10, 20], [30, 40], [50, 60]])
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=1, feature_embedding=feature_embedding
        )
        expected_series = pd.Series([30, 40], index=pd.Index([0, 1]), name=1)
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == 1
        assert spf.feature_embedding is feature_embedding

    def test_from_feature_embedding_custom_names(self):
        values = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        feature_embedding = sa.FeatureEmbedding(values=values, name="Y")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=0, feature_embedding=feature_embedding
        )
        expected_series = pd.Series([1, 2, 3], index=pd.Index([0, 1, 2]), name=0)
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == 0
        assert spf.feature_embedding is feature_embedding

    def test_from_feature_embedding_single_feature(self):
        values = pd.DataFrame([[100], [200], [300]])
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=2, feature_embedding=feature_embedding
        )
        expected_series = pd.Series([300], index=pd.Index([0]), name=2)
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == 2
        assert spf.feature_embedding is feature_embedding

    def test_from_feature_embedding_many_features(self):
        values = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=1, feature_embedding=feature_embedding
        )
        expected_series = pd.Series(
            [6, 7, 8, 9, 10], index=pd.Index([0, 1, 2, 3, 4]), name=1
        )
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == 1
        assert spf.feature_embedding is feature_embedding

    def test_from_feature_embedding_with_string_indices(self):
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=["row1", "row2"], columns=["col1", "col2"]
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index="row1", feature_embedding=feature_embedding
        )
        expected_series = pd.Series(
            [1, 2], index=pd.Index(["col1", "col2"]), name="row1"
        )
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == "row1"
        assert spf.feature_embedding is feature_embedding

    def test_from_feature_embedding_string_index_second_row(self):
        values = pd.DataFrame(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            index=["alpha", "beta", "gamma"],
            columns=["X", "Y", "Z"],
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="features")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index="beta", feature_embedding=feature_embedding
        )
        expected_series = pd.Series(
            [40, 50, 60], index=pd.Index(["X", "Y", "Z"]), name="beta"
        )
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == "beta"

    def test_from_feature_embedding_mixed_string_indices(self):
        values = pd.DataFrame(
            [[1.5, 2.5], [3.5, 4.5]],
            index=["sample_a", "sample_b"],
            columns=["feature_1", "feature_2"],
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index="sample_b", feature_embedding=feature_embedding
        )
        expected_series = pd.Series(
            [3.5, 4.5],
            index=pd.Index(["feature_1", "feature_2"]),
            name="sample_b",
        )
        pd.testing.assert_series_equal(spf.values, expected_series)

    def test_from_feature_embedding_custom_integer_indices(self):
        values = pd.DataFrame([[100, 200], [300, 400]], index=[10, 20], columns=[5, 15])
        feature_embedding = sa.FeatureEmbedding(values=values, name="Z")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index=20, feature_embedding=feature_embedding
        )
        expected_series = pd.Series([300, 400], index=pd.Index([5, 15]), name=20)
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == 20

    def test_from_feature_embedding_string_indices_with_custom_names(self):
        values = pd.DataFrame(
            [[7, 8, 9], [10, 11, 12]],
            index=["obs1", "obs2"],
            columns=["var_x", "var_y", "var_z"],
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="data")
        spf = sa.SamplePointFeatures.from_feature_embedding(
            sample_index="obs1", feature_embedding=feature_embedding
        )
        expected_series = pd.Series(
            [7, 8, 9],
            index=pd.Index(["var_x", "var_y", "var_z"]),
            name="obs1",
        )
        pd.testing.assert_series_equal(spf.values, expected_series)
        assert spf.name == "obs1"
