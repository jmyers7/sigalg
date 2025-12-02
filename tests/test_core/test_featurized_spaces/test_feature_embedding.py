import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:

    def test_basic_construction(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=sample_space.values, columns=["X0", "X1"]
        )
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, values=df, name="X"
        )
        assert feature_embedding.sample_space == sample_space
        expected_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=sample_space.values, columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)
        assert feature_embedding.name == "X"


class TestGenerateFromDF:

    def test_default_generation(self):
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        feature_embedding = sa.FeatureEmbedding.from_df(df)
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["omega0", "omega1", "omega2"], name="Omega"),
            columns=["E0", "E1"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)
        assert feature_embedding.sample_space.name == "Omega"
        assert feature_embedding.name == "E"

    def test_with_custom_names(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding = sa.FeatureEmbedding.from_df(
            df, name="Y", sample_prefix="s", sample_space_name="S"
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=["Y0", "Y1"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)
        assert feature_embedding.sample_space.name == "S"
        assert feature_embedding.name == "Y"

    def test_with_initial_indices(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding = sa.FeatureEmbedding.from_df(
            df, name="E", initial_sample_index=5, initial_feature_index=3
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index(["omega5", "omega6"], name="Omega"),
            columns=["E3", "E4"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_without_overwrite_defaults(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding = sa.FeatureEmbedding.from_df(
            df,
            overwrite_default_sample_index=False,
            overwrite_default_feature_index=False,
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="Omega"),
            columns=[0, 1],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_with_existing_index(self):
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["f1", "f2"])
        feature_embedding = sa.FeatureEmbedding.from_df(
            df,
            name="X",
            overwrite_default_sample_index=True,
            overwrite_default_feature_index=True,
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index(["a", "b"], name="Omega"),
            columns=["f1", "f2"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)


class TestFromSequences:

    def test_binary_sequences_length_2(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1], [1, 0], [1, 1]],
            index=pd.Index(["omega0", "omega1", "omega2", "omega3"], name="Omega"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_binary_sequences_length_3(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 3, name="X")
        expected_df = pd.DataFrame(
            data=[
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            index=pd.Index(
                [
                    "omega0",
                    "omega1",
                    "omega2",
                    "omega3",
                    "omega4",
                    "omega5",
                    "omega6",
                    "omega7",
                ],
                name="Omega",
            ),
            columns=["X0", "X1", "X2"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_ternary_sequences(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1, 2], 2, name="X")
        expected_df = pd.DataFrame(
            data=[
                [0, 0],
                [0, 1],
                [0, 2],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
                [2, 2],
            ],
            index=pd.Index(
                [
                    "omega0",
                    "omega1",
                    "omega2",
                    "omega3",
                    "omega4",
                    "omega5",
                    "omega6",
                    "omega7",
                    "omega8",
                ],
                name="Omega",
            ),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

    def test_with_custom_names(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences(
            [0, 1], 2, name="Y", sample_space_name="S", sample_prefix="s"
        )
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1], [1, 0], [1, 1]],
            index=pd.Index(["s0", "s1", "s2", "s3"], name="S"),
            columns=["Y0", "Y1"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)
        assert feature_embedding.sample_space.name == "S"
        assert feature_embedding.name == "Y"

    def test_with_initial_indices(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences(
            [0, 1], 2, name="X", initial_sample_index=10, initial_feature_index=5
        )
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1], [1, 0], [1, 1]],
            index=pd.Index(["omega10", "omega11", "omega12", "omega13"], name="Omega"),
            columns=["X5", "X6"],
        )
        pd.testing.assert_frame_equal(feature_embedding.values, expected_df)


class TestGetSampleFeatures:

    def test_get_sample_features(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        sf = feature_embedding.get_sample_features("omega1")
        assert sf.name == "omega1"
        expected_series = pd.Series([0, 1], index=["X0", "X1"], name="omega1")
        pd.testing.assert_series_equal(sf.values, expected_series)

    def test_get_sample_features_at(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        sf = feature_embedding.get_sample_features_at[1]
        expected_series = pd.Series([0, 1], index=["X0", "X1"], name="omega1")
        pd.testing.assert_series_equal(sf.values, expected_series)


class TestGetEventFeatures:

    def test_get_event_features(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        ef = feature_embedding.get_event_features(["omega0", "omega2"])
        expected_df = pd.DataFrame(
            data=[[0, 0], [1, 0]],
            index=pd.Index(["omega0", "omega2"], name="A"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.values, expected_df)
        assert ef.sample_space.name == "A"
        assert set(ef.sample_space.values) == {"omega0", "omega2"}

    def test_get_event_features_with_name(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        ef = feature_embedding.get_event_features(["omega0", "omega1"], name="B")
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1]],
            index=pd.Index(["omega0", "omega1"], name="B"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.values, expected_df)
        assert ef.sample_space.name == "B"
        assert set(ef.sample_space.values) == {"omega0", "omega1"}

    def test_get_event_features_at(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        ef = feature_embedding.get_event_features_at[[0, 2]]
        expected_df = pd.DataFrame(
            data=[[0, 0], [1, 0]],
            index=pd.Index(["omega0", "omega2"], name="A"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.values, expected_df)
        assert set(ef.sample_space.values) == {"omega0", "omega2"}

    def test_get_event_features_at_with_name(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        ef = feature_embedding.get_event_features_at[[1, 3], "C"]
        expected_df = pd.DataFrame(
            data=[[0, 1], [1, 1]],
            index=pd.Index(["omega1", "omega3"], name="C"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.values, expected_df)
        assert ef.sample_space.name == "C"


class TestGetFeatureRV:

    def test_get_feature_rv(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        X0 = feature_embedding.get_feature_rv("X0")
        assert X0.name == "X0"
        expected_series = pd.Series(
            [0, 0, 1, 1], index=["omega0", "omega1", "omega2", "omega3"], name="X0"
        )
        expected_series.index.name = "Omega"
        pd.testing.assert_series_equal(X0.values, expected_series)


class TestGetSubFeatures:

    def test_get_sub_features_single_column(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 3, name="X")
        sub = feature_embedding.get_sub_features(["X0"])
        expected_df = pd.DataFrame(
            data=[[0], [0], [0], [0], [1], [1], [1], [1]],
            index=pd.Index(
                [
                    "omega0",
                    "omega1",
                    "omega2",
                    "omega3",
                    "omega4",
                    "omega5",
                    "omega6",
                    "omega7",
                ],
                name="Omega",
            ),
            columns=["X0"],
        )
        pd.testing.assert_frame_equal(sub.values, expected_df)
        assert sub.sample_space == feature_embedding.sample_space

    def test_get_sub_features_multiple_columns(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 3, name="X")
        sub = feature_embedding.get_sub_features(["X0", "X2"])
        expected_df = pd.DataFrame(
            data=[
                [0, 0],
                [0, 1],
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [1, 0],
                [1, 1],
            ],
            index=pd.Index(
                [
                    "omega0",
                    "omega1",
                    "omega2",
                    "omega3",
                    "omega4",
                    "omega5",
                    "omega6",
                    "omega7",
                ],
                name="Omega",
            ),
            columns=["X0", "X2"],
        )
        pd.testing.assert_frame_equal(sub.values, expected_df)

    def test_sub_features_name(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="Y")
        sub = feature_embedding.get_sub_features(["Y0"])
        expected_df = pd.DataFrame(
            data=[[0], [0], [1], [1]],
            index=pd.Index(["omega0", "omega1", "omega2", "omega3"], name="Omega"),
            columns=["Y0"],
        )
        pd.testing.assert_frame_equal(sub.values, expected_df)
        assert sub.name == "Y_sub"


class TestApplyToFeatures:

    def test_apply_sum(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        result = feature_embedding.apply_to_features(lambda sf: sf.values.sum())
        expected_series = pd.Series(
            [0, 1, 1, 2], index=["omega0", "omega1", "omega2", "omega3"]
        )
        expected_series.index.name = "Omega"
        pd.testing.assert_series_equal(result, expected_series)

    def test_apply_product(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([1, 2], 2, name="X")
        result = feature_embedding.apply_to_features(lambda sf: sf.values.prod())
        expected_series = pd.Series(
            [1, 2, 2, 4], index=["omega0", "omega1", "omega2", "omega3"]
        )
        expected_series.index.name = "Omega"
        pd.testing.assert_series_equal(result, expected_series)


class TestIterSampleFeatures:

    def test_iter_sample_features(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        indices = []
        for idx, _ in feature_embedding.iter_sample_features():
            indices.append(idx)
        assert indices == ["omega0", "omega1", "omega2", "omega3"]


class TestEquality:

    def test_equal_feature_embedding(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding1 = sa.FeatureEmbedding.from_df(df, name="X")
        feature_embedding2 = sa.FeatureEmbedding.from_df(df, name="X")
        assert feature_embedding1 == feature_embedding2

    def test_equal_feature_embedding_custom_names(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding1 = sa.FeatureEmbedding.from_df(
            df, name="Y", sample_prefix="s", sample_space_name="S"
        )
        feature_embedding2 = sa.FeatureEmbedding.from_df(
            df, name="Y", sample_prefix="s", sample_space_name="S"
        )
        assert feature_embedding1 == feature_embedding2

    def test_not_equal_different_features(self):
        df1 = pd.DataFrame([[1, 2], [3, 4]])
        df2 = pd.DataFrame([[1, 2], [3, 5]])
        feature_embedding1 = sa.FeatureEmbedding.from_df(df1, name="X")
        feature_embedding2 = sa.FeatureEmbedding.from_df(df2, name="X")
        assert feature_embedding1 != feature_embedding2

    def test_not_equal_different_sample_space(self):
        df1 = pd.DataFrame([[1, 2], [3, 4]])
        df2 = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding1 = sa.FeatureEmbedding.from_df(
            df1, name="X", sample_space_name="Omega"
        )
        feature_embedding2 = sa.FeatureEmbedding.from_df(
            df2, name="X", sample_space_name="S"
        )
        assert feature_embedding1 != feature_embedding2

    def test_not_equal_different_embedding_name(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding1 = sa.FeatureEmbedding.from_df(df, name="X")
        feature_embedding2 = sa.FeatureEmbedding.from_df(df, name="Y")
        assert feature_embedding1 != feature_embedding2

    def test_not_equal_different_type(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding = sa.FeatureEmbedding.from_df(df, name="X")
        assert feature_embedding != "not a feature embedding"
        assert feature_embedding != 42
        assert feature_embedding is not None


class TestSetters:

    def test_set_name(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        df = pd.DataFrame([[1, 2], [3, 4]], index=["s0", "s1"], columns=["X0", "X1"])
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, values=df, name="X"
        )
        feature_embedding.name = "Y"
        assert feature_embedding.name == "Y"


class TestAddProbabilityMeasure:

    def test_add_uniform_measure(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        fps = feature_embedding.add_probability_measure_from_features(lambda sf: 0.25)
        assert isinstance(fps, sa.FeaturizedProbabilitySpace)
        assert abs(fps.P("omega0") - 0.25) < 1e-10
        assert abs(fps.P("omega1") - 0.25) < 1e-10

    def test_add_custom_measure(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 3, name="X")

        def pmf(sf):
            num_ones = sf.sum()
            return 0.5**num_ones * 0.5 ** (3 - num_ones)

        fps = feature_embedding.add_probability_measure_from_features(pmf)
        assert abs(fps.P("omega0") - 0.125) < 1e-10


class TestValidation:

    def test_invalid_sample_space_type(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        with pytest.raises(TypeError, match="sample_space must be a SampleSpace"):
            sa.FeatureEmbedding(sample_space="invalid", values=df, name="X")

    def test_mismatched_indices(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"])
        with pytest.raises(ValueError, match="indices of `values` must match"):
            sa.FeatureEmbedding(sample_space=sample_space, values=df, name="X")

    def test_from_sequences_non_list(self):
        with pytest.raises(TypeError, match="state_space must be a list"):
            sa.FeatureEmbedding.from_sequences(123, 2, name="X")

    def test_from_sequences_empty_state_space(self):
        with pytest.raises(ValueError, match="state_space must be non-empty"):
            sa.FeatureEmbedding.from_sequences([], 2, name="X")

    def test_from_sequences_invalid_sequence_length(self):
        with pytest.raises(
            ValueError, match="sequence_length must be a positive integer"
        ):
            sa.FeatureEmbedding.from_sequences([0, 1], 0, name="X")

    def test_from_sequences_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be a positive integer"):
            sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X", threshold=0)

    def test_from_sequences_exceeds_threshold(self):
        with pytest.raises(ValueError, match="exceeds threshold"):
            sa.FeatureEmbedding.from_sequences([0, 1], 20, name="X", threshold=10)

    def test_get_sample_features_invalid_index(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        with pytest.raises(ValueError, match="not found in sample_space"):
            feature_embedding.get_sample_features("invalid")

    def test_get_event_features_invalid_index(self):
        feature_embedding = sa.FeatureEmbedding.from_sequences([0, 1], 2, name="X")
        with pytest.raises(ValueError, match="not found in sample_space"):
            feature_embedding.get_event_features(["omega0", "invalid"])

    def test_invalid_name_type(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        df = pd.DataFrame([[1, 2], [3, 4]], index=["s0", "s1"], columns=["X0", "X1"])
        feature_embedding = sa.FeatureEmbedding(
            sample_space=sample_space, values=df, name="X"
        )
        with pytest.raises(TypeError, match="name must be a string"):
            feature_embedding.name = 123
