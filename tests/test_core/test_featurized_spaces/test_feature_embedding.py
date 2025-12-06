import numpy as np
import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:

    def test_basic_construction(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="features")
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["s0", "s1", "s2"], name="S"),
            columns=pd.Index(["f0", "f1"], name="features"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space,
            feature_index=feature_index,
            values=df,
            name="X",
        )
        assert feature_embedding.sample_space == sample_space
        assert feature_embedding.feature_index == feature_index
        assert feature_embedding.name == "X"
        expected_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["s0", "s1", "s2"], name="S"),
            columns=pd.Index(["f0", "f1"], name="features"),
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_construction_with_default_name(self):
        sample_space = sa.SampleSpace(["a", "b"], name="Sample")
        feature_index = sa.FeatureIndex([0, 1], values_name="idx")
        df = pd.DataFrame(
            [[10, 20], [30, 40]],
            index=pd.Index(["a", "b"], name="Sample"),
            columns=pd.Index([0, 1], name="idx"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df
        )
        assert feature_embedding.name == "X"


class TestProperties:

    def test_values_property_returns_copy(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        values1 = feature_embedding.values
        values2 = feature_embedding.values
        assert values1 is not values2
        pd.testing.assert_frame_equal(values1, values2)

    def test_sample_space_property(self):
        sample_space = sa.SampleSpace(["x", "y", "z"], name="XYZ")
        feature_index = sa.FeatureIndex(["a"], values_name="A")
        df = pd.DataFrame(
            [[1], [2], [3]],
            index=pd.Index(["x", "y", "z"], name="XYZ"),
            columns=pd.Index(["a"], name="A"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="Y"
        )
        assert feature_embedding.sample_space == sample_space

    def test_feature_index_property(self):
        sample_space = sa.SampleSpace([1, 2], name="Nums")
        feature_index = sa.FeatureIndex(["col1", "col2"], values_name="cols")
        df = pd.DataFrame(
            [[5, 6], [7, 8]],
            index=pd.Index([1, 2], name="Nums"),
            columns=pd.Index(["col1", "col2"], name="cols"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="Z"
        )
        assert feature_embedding.feature_index == feature_index

    def test_name_property(self):
        sample_space = sa.SampleSpace(["a"], name="A")
        feature_index = sa.FeatureIndex(["b"], values_name="B")
        df = pd.DataFrame(
            [[100]],
            index=pd.Index(["a"], name="A"),
            columns=pd.Index(["b"], name="B"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space,
            feature_index=feature_index,
            values=df,
            name="TestName",
        )
        assert feature_embedding.name == "TestName"

    def test_shape_property(self):
        sample_space = sa.SampleSpace(["r1", "r2", "r3"], name="R")
        feature_index = sa.FeatureIndex(["c1", "c2"], values_name="C")
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["r1", "r2", "r3"], name="R"),
            columns=pd.Index(["c1", "c2"], name="C"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        assert feature_embedding.shape == (3, 2)

    def test_len_method(self):
        sample_space = sa.SampleSpace(["a", "b", "c", "d"], name="ABCD")
        feature_index = sa.FeatureIndex(["x"], values_name="X")
        df = pd.DataFrame(
            [[1], [2], [3], [4]],
            index=pd.Index(["a", "b", "c", "d"], name="ABCD"),
            columns=pd.Index(["x"], name="X"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="Y"
        )
        assert len(feature_embedding) == 4


class TestFromDF:

    def test_from_df_basic(self):
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        feature_embedding = sa.FeatureEmbedding.from_df(df, name="Test")
        assert feature_embedding.name == "Test"
        assert feature_embedding.sample_space.name == "Test_sample_space"
        assert len(feature_embedding.sample_space) == 3
        assert len(feature_embedding.feature_index) == 2
        expected_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index([0, 1], name="feature"),
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_from_df_with_custom_names(self):
        df = pd.DataFrame([[10, 20]], index=["row"], columns=["colA", "colB"])
        feature_embedding = sa.FeatureEmbedding.from_df(
            df, name="Custom", sample_values_name="samp", feature_index_name="feat"
        )
        assert feature_embedding.name == "Custom"
        expected_df = pd.DataFrame(
            [[10, 20]],
            index=pd.Index(["row"], name="samp"),
            columns=pd.Index(["colA", "colB"], name="feat"),
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_from_df_preserves_original_indices(self):
        df = pd.DataFrame(
            [[7, 8], [9, 10]], index=["idx1", "idx2"], columns=["c1", "c2"]
        )
        feature_embedding = sa.FeatureEmbedding.from_df(df)
        assert list(feature_embedding.sample_space.values) == ["idx1", "idx2"]
        assert list(feature_embedding.feature_index.values) == ["c1", "c2"]


class TestFromNumpy:

    def test_from_numpy_basic(self):
        array = np.array([[1, 2], [3, 4]])
        feature_embedding = sa.FeatureEmbedding.from_numpy(array, name="NumPy")
        assert feature_embedding.name == "NumPy"
        assert feature_embedding.sample_space.name == "NumPy_sample_space"
        assert len(feature_embedding.sample_space) == 2
        assert len(feature_embedding.feature_index) == 2
        expected_df = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="sample"),
            columns=pd.Index([0, 1], name="feature"),
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_from_numpy_with_custom_names(self):
        array = np.array([[5.5, 6.6, 7.7]])
        feature_embedding = sa.FeatureEmbedding.from_numpy(
            array, name="NP", sample_values_name="rows", feature_index_name="cols"
        )
        expected_df = pd.DataFrame(
            [[5.5, 6.6, 7.7]],
            index=pd.Index([0], name="rows"),
            columns=pd.Index([0, 1, 2], name="cols"),
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_from_numpy_large_array(self):
        array = np.ones((5, 3))
        feature_embedding = sa.FeatureEmbedding.from_numpy(array)
        assert feature_embedding.shape == (5, 3)
        assert len(feature_embedding.sample_space) == 5
        assert len(feature_embedding.feature_index) == 3


class TestGetSampleFeatures:

    def test_get_sample_features_basic(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        sample_features = feature_embedding.get_sample_features("s0")
        assert sample_features.name == "s0"
        expected_series = pd.Series(
            [1, 2], index=pd.Index(["f0", "f1"], name="F"), name="s0"
        )
        pd.testing.assert_series_equal(sample_features.values, expected_series)

    def test_get_sample_features_at_indexer(self):
        sample_space = sa.SampleSpace(["a", "b", "c"], name="ABC")
        feature_index = sa.FeatureIndex(["x", "y"], values_name="XY")
        df = pd.DataFrame(
            [[10, 20], [30, 40], [50, 60]],
            index=pd.Index(["a", "b", "c"], name="ABC"),
            columns=pd.Index(["x", "y"], name="XY"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="Z"
        )
        sample_features = feature_embedding.get_sample_features_at[1]
        assert sample_features.name == "b"
        expected_series = pd.Series(
            [30, 40], index=pd.Index(["x", "y"], name="XY"), name="b"
        )
        pd.testing.assert_series_equal(sample_features.values, expected_series)


class TestGetEventFeatures:

    def test_get_event_features_basic(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["s0", "s1", "s2"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        event_features = feature_embedding.get_event_features(["s0", "s2"], name="E")
        assert event_features.sample_space.name == "E"
        assert set(event_features.sample_space.values) == {"s0", "s2"}
        expected_df = pd.DataFrame(
            [[1, 2], [5, 6]],
            index=pd.Index(["s0", "s2"], name="E"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        pd.testing.assert_frame_equal(event_features.values, expected_df)

    def test_get_event_features_default_name(self):
        sample_space = sa.SampleSpace(["a", "b"], name="AB")
        feature_index = sa.FeatureIndex(["x"], values_name="X")
        df = pd.DataFrame(
            [[100], [200]],
            index=pd.Index(["a", "b"], name="AB"),
            columns=pd.Index(["x"], name="X"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="Y"
        )
        event_features = feature_embedding.get_event_features(["a"])
        assert event_features.sample_space.name == "A"

    def test_get_event_features_at_indexer(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"], name="S")
        feature_index = sa.FeatureIndex(["c0", "c1"], values_name="C")
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=pd.Index(["s0", "s1", "s2", "s3"], name="S"),
            columns=pd.Index(["c0", "c1"], name="C"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        event_features = feature_embedding.get_event_features_at[[0, 2], "Event"]
        assert event_features.sample_space.name == "Event"
        expected_df = pd.DataFrame(
            [[1, 2], [5, 6]],
            index=pd.Index(["s0", "s2"], name="Event"),
            columns=pd.Index(["c0", "c1"], name="C"),
        )
        pd.testing.assert_frame_equal(event_features.values, expected_df)


class TestGetFeatureRV:

    def test_get_feature_rv_basic(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["s0", "s1", "s2"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        rv = feature_embedding.get_feature_rv("f0")
        assert rv.name == "f0"
        expected_values = pd.Series(
            [1, 3, 5], index=pd.Index(["s0", "s1", "s2"], name="S"), name="f0"
        )
        pd.testing.assert_series_equal(rv.values, expected_values)


class TestGetSubFeatures:

    def test_get_sub_features_single_column(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1", "f2"], values_name="F")
        df = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0", "f1", "f2"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        sub_features = feature_embedding.get_sub_features(["f1"])
        assert sub_features.name == "X_sub"
        assert sub_features.sample_space == sample_space
        expected_df = pd.DataFrame(
            [[2], [5]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f1"], name="F"),
        )
        pd.testing.assert_frame_equal(sub_features.values, expected_df)

    def test_get_sub_features_multiple_columns(self):
        sample_space = sa.SampleSpace(["a", "b"], name="AB")
        feature_index = sa.FeatureIndex(["c0", "c1", "c2"], values_name="C")
        df = pd.DataFrame(
            [[10, 20, 30], [40, 50, 60]],
            index=pd.Index(["a", "b"], name="AB"),
            columns=pd.Index(["c0", "c1", "c2"], name="C"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="Y"
        )
        sub_features = feature_embedding.get_sub_features(["c0", "c2"])
        expected_df = pd.DataFrame(
            [[10, 30], [40, 60]],
            index=pd.Index(["a", "b"], name="AB"),
            columns=pd.Index(["c0", "c2"], name="C"),
        )
        pd.testing.assert_frame_equal(sub_features.values, expected_df)


class TestIterSampleFeatures:

    def test_iter_sample_features(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame(
            [[100], [200]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        indices = []
        for sample_index, sample_features in feature_embedding.iter_sample_features():
            indices.append(sample_index)
            assert sample_features.name == sample_index
        assert indices == ["s0", "s1"]


class TestApplyToFeatures:

    def test_apply_to_features_sum(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["s0", "s1", "s2"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        result = feature_embedding.apply_to_features(lambda sf: sf.values.sum())
        expected_series = pd.Series(
            [3, 7, 11], index=pd.Index(["s0", "s1", "s2"], name="S")
        )
        pd.testing.assert_series_equal(result, expected_series)

    def test_apply_to_features_max(self):
        sample_space = sa.SampleSpace(["a", "b"], name="AB")
        feature_index = sa.FeatureIndex(["x", "y", "z"], values_name="XYZ")
        df = pd.DataFrame(
            [[10, 5, 15], [3, 8, 2]],
            index=pd.Index(["a", "b"], name="AB"),
            columns=pd.Index(["x", "y", "z"], name="XYZ"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="W"
        )
        result = feature_embedding.apply_to_features(lambda sf: sf.values.max())
        expected_series = pd.Series([15, 8], index=pd.Index(["a", "b"], name="AB"))
        pd.testing.assert_series_equal(result, expected_series)


class TestEquality:

    def test_equal_feature_embeddings(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        fe1 = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        fe2 = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        assert fe1 == fe2

    def test_not_equal_different_values(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df1 = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        df2 = pd.DataFrame(
            [[1, 2], [3, 5]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        fe1 = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df1, name="X"
        )
        fe2 = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df2, name="X"
        )
        assert fe1 != fe2

    def test_not_equal_different_sample_space(self):
        sample_space1 = sa.SampleSpace(["s0", "s1"], name="S1")
        sample_space2 = sa.SampleSpace(["s0", "s1"], name="S2")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df1 = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S1"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        df2 = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S2"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        fe1 = sa.FeatureEmbedding(
            sample_space=sample_space1,
            feature_index=feature_index,
            values=df1,
            name="X",
        )
        fe2 = sa.FeatureEmbedding(
            sample_space=sample_space2,
            feature_index=feature_index,
            values=df2,
            name="X",
        )
        assert fe1 != fe2

    def test_not_equal_different_name(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0", "f1"], name="F"),
        )
        fe1 = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        fe2 = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="Y"
        )
        assert fe1 != fe2

    def test_not_equal_to_non_feature_embedding(self):
        sample_space = sa.SampleSpace(["s0"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame(
            [[1]], index=pd.Index(["s0"], name="S"), columns=pd.Index(["f0"], name="F")
        )
        fe = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        assert fe != "not a feature embedding"
        assert fe != 42
        assert fe is not None


class TestSetters:

    def test_name_setter(self):
        sample_space = sa.SampleSpace(["s0"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame(
            [[1]], index=pd.Index(["s0"], name="S"), columns=pd.Index(["f0"], name="F")
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        feature_embedding.name = "NewName"
        assert feature_embedding.name == "NewName"


class TestValidation:

    def test_invalid_sample_space_type(self):
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame([[1]], columns=pd.Index(["f0"], name="F"))
        with pytest.raises(TypeError, match="sample_space must be a SampleSpace"):
            sa.FeatureEmbedding(
                sample_space="invalid", feature_index=feature_index, values=df
            )

    def test_invalid_feature_index_type(self):
        sample_space = sa.SampleSpace(["s0"], name="S")
        df = pd.DataFrame([[1]], index=pd.Index(["s0"], name="S"))
        with pytest.raises(TypeError, match="feature_index must be a FeatureIndex"):
            sa.FeatureEmbedding(
                sample_space=sample_space, feature_index="invalid", values=df
            )

    def test_invalid_values_type(self):
        sample_space = sa.SampleSpace(["s0"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        with pytest.raises(TypeError, match="values must be a pandas DataFrame"):
            sa.FeatureEmbedding(
                sample_space=sample_space, feature_index=feature_index, values=[1, 2, 3]
            )

    def test_mismatched_sample_space_indices(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame(
            [[1], [2]],
            index=pd.Index(["a", "b"], name="Wrong"),
            columns=pd.Index(["f0"], name="F"),
        )
        with pytest.raises(ValueError, match="indices of `values` must match"):
            sa.FeatureEmbedding(
                sample_space=sample_space, feature_index=feature_index, values=df
            )

    def test_mismatched_feature_index_columns(self):
        sample_space = sa.SampleSpace(["s0"], name="S")
        feature_index = sa.FeatureIndex(["f0", "f1"], values_name="F")
        df = pd.DataFrame(
            [[1, 2, 3]],
            index=pd.Index(["s0"], name="S"),
            columns=pd.Index(["a", "b", "c"], name="Wrong"),
        )
        with pytest.raises(ValueError, match="columns of `values` must match"):
            sa.FeatureEmbedding(
                sample_space=sample_space, feature_index=feature_index, values=df
            )

    def test_from_df_invalid_type(self):
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            sa.FeatureEmbedding.from_df([[1, 2], [3, 4]])

    def test_from_numpy_invalid_type(self):
        with pytest.raises(TypeError, match="array must be a numpy ndarray"):
            sa.FeatureEmbedding.from_numpy([[1, 2], [3, 4]])

    def test_get_sample_features_invalid_index(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame(
            [[1], [2]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        with pytest.raises(ValueError, match="not found in sample_space"):
            feature_embedding.get_sample_features("invalid_index")

    def test_get_event_features_invalid_index(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame(
            [[1], [2]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=pd.Index(["f0"], name="F"),
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        with pytest.raises(ValueError, match="not found in sample_space"):
            feature_embedding.get_event_features(["s0", "invalid_index"])

    def test_name_setter_invalid_type(self):
        sample_space = sa.SampleSpace(["s0"], name="S")
        feature_index = sa.FeatureIndex(["f0"], values_name="F")
        df = pd.DataFrame(
            [[1]], index=pd.Index(["s0"], name="S"), columns=pd.Index(["f0"], name="F")
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, feature_index=feature_index, values=df, name="X"
        )
        with pytest.raises(TypeError, match="name must be a string"):
            feature_embedding.name = 123
