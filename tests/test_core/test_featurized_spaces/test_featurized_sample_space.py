import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:
    def test_basic_construction(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]], index=sample_space.values, columns=["X0", "X1"]
        )
        feature_embedding = sa.FeatureEmbedding(features=df, name="X")
        fss = sa.FeaturizedSampleSpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        assert fss.sample_space == sample_space
        assert fss.feature_embedding == feature_embedding


class TestGenerateFromDF:
    def test_default_generation(self):
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        fss = sa.FeaturizedSampleSpace.from_df(df)
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["omega0", "omega1", "omega2"], name="Omega"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)
        assert fss.sample_space.name == "Omega"
        assert fss.feature_embedding.name == "X"

    def test_with_custom_names(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fss = sa.FeaturizedSampleSpace.from_df(
            df, embedding_name="Y", sample_prefix="s", sample_space_name="S"
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index(["s0", "s1"], name="S"),
            columns=["Y0", "Y1"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)
        assert fss.sample_space.name == "S"
        assert fss.feature_embedding.name == "Y"

    def test_with_initial_indices(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fss = sa.FeaturizedSampleSpace.from_df(
            df, initial_sample_index=5, initial_feature_index=3
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index(["omega5", "omega6"], name="Omega"),
            columns=["X3", "X4"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)

    def test_without_overwrite_defaults(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fss = sa.FeaturizedSampleSpace.from_df(
            df,
            overwrite_default_sample_index=False,
            overwrite_default_feature_index=False,
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index([0, 1], name="Omega"),
            columns=[0, 1],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)

    def test_with_existing_index(self):
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["f1", "f2"])
        fss = sa.FeaturizedSampleSpace.from_df(
            df,
            overwrite_default_sample_index=False,
            overwrite_default_feature_index=False,
        )
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index(["a", "b"], name="Omega"),
            columns=["f1", "f2"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)

    def test_single_row(self):
        df = pd.DataFrame([[1, 2]])
        fss = sa.FeaturizedSampleSpace.from_df(df)
        expected_df = pd.DataFrame(
            data=[[1, 2]],
            index=pd.Index(["omega"], name="Omega"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)

    def test_single_column(self):
        df = pd.DataFrame([[1], [2], [3]])
        fss = sa.FeaturizedSampleSpace.from_df(df)
        expected_df = pd.DataFrame(
            data=[[1], [2], [3]],
            index=pd.Index(["omega0", "omega1", "omega2"], name="Omega"),
            columns=["X"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)


class TestFromSequences:
    def test_binary_sequences_length_2(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1], [1, 0], [1, 1]],
            index=pd.Index(["omega0", "omega1", "omega2", "omega3"], name="Omega"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)

    def test_binary_sequences_length_3(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 3)
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
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)

    def test_ternary_sequences(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1, 2], 2)
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
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)

    def test_with_custom_names(self):
        fss = sa.FeaturizedSampleSpace.from_sequences(
            [0, 1], 2, embedding_name="Y", sample_space_name="S", sample_prefix="s"
        )
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1], [1, 0], [1, 1]],
            index=pd.Index(["s0", "s1", "s2", "s3"], name="S"),
            columns=["Y0", "Y1"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)
        assert fss.sample_space.name == "S"
        assert fss.feature_embedding.name == "Y"

    def test_with_initial_indices(self):
        fss = sa.FeaturizedSampleSpace.from_sequences(
            [0, 1], 2, initial_sample_index=10, initial_feature_index=5
        )
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1], [1, 0], [1, 1]],
            index=pd.Index(["omega10", "omega11", "omega12", "omega13"], name="Omega"),
            columns=["X5", "X6"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)


class TestGetSampleFeatures:
    def test_get_sample_features(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        sf = fss.get_sample_features("omega1")
        assert sf.name == "omega1"
        expected_series = pd.Series([0, 1], index=["X0", "X1"], name="omega1")
        pd.testing.assert_series_equal(sf.values, expected_series)

    def test_get_sample_features_at(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        sf = fss.get_sample_features_at[1]
        expected_series = pd.Series([0, 1], index=["X0", "X1"], name="omega1")
        pd.testing.assert_series_equal(sf.values, expected_series)


class TestGetEventFeatures:
    def test_get_event_features(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        ef = fss.get_event_features(["omega0", "omega2"])
        expected_df = pd.DataFrame(
            data=[[0, 0], [1, 0]],
            index=pd.Index(["omega0", "omega2"], name="A"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.feature_embedding.values, expected_df)
        assert ef.event.name == "A"

    def test_get_event_features_with_name(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        ef = fss.get_event_features(["omega0", "omega1"], name="B")
        expected_df = pd.DataFrame(
            data=[[0, 0], [0, 1]],
            index=pd.Index(["omega0", "omega1"], name="B"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.feature_embedding.values, expected_df)
        assert ef.event.name == "B"

    def test_get_event_features_at(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        ef = fss.get_event_features_at[[0, 2]]
        expected_df = pd.DataFrame(
            data=[[0, 0], [1, 0]],
            index=pd.Index(["omega0", "omega2"], name="A"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.feature_embedding.values, expected_df)
        assert set(ef.event.values) == {"omega0", "omega2"}

    def test_get_event_features_at_with_name(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        ef = fss.get_event_features_at[[1, 3], "C"]
        expected_df = pd.DataFrame(
            data=[[0, 1], [1, 1]],
            index=pd.Index(["omega1", "omega3"], name="C"),
            columns=["X0", "X1"],
        )
        pd.testing.assert_frame_equal(ef.feature_embedding.values, expected_df)
        assert ef.event.name == "C"


class TestGetFeatureRV:
    def test_get_feature_rv(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        X0 = fss.get_feature_rv("X0")
        assert X0.name == "X0"
        expected_series = pd.Series(
            [0, 0, 1, 1], index=["omega0", "omega1", "omega2", "omega3"], name="X0"
        )
        expected_series.index.name = "Omega"
        pd.testing.assert_series_equal(X0.values, expected_series)


class TestGetSubFeatures:
    def test_get_sub_features_single_column(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 3)
        sub = fss.get_sub_features(["X0"])
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
        pd.testing.assert_frame_equal(sub.feature_embedding.values, expected_df)
        assert sub.sample_space == fss.sample_space

    def test_get_sub_features_multiple_columns(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 3)
        sub = fss.get_sub_features(["X0", "X2"])
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
        pd.testing.assert_frame_equal(sub.feature_embedding.values, expected_df)

    def test_sub_features_name(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2, embedding_name="Y")
        sub = fss.get_sub_features(["Y0"])
        expected_df = pd.DataFrame(
            data=[[0], [0], [1], [1]],
            index=pd.Index(["omega0", "omega1", "omega2", "omega3"], name="Omega"),
            columns=["Y0"],
        )
        pd.testing.assert_frame_equal(sub.feature_embedding.values, expected_df)
        assert sub.feature_embedding.name == "Y_sub"


class TestApplyToFeatures:
    def test_apply_sum(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        result = fss.apply_to_features(lambda sf: sf.values.sum())
        expected_series = pd.Series(
            [0, 1, 1, 2], index=["omega0", "omega1", "omega2", "omega3"]
        )
        expected_series.index.name = "Omega"
        pd.testing.assert_series_equal(result, expected_series)

    def test_apply_product(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([1, 2], 2)
        result = fss.apply_to_features(lambda sf: sf.values.prod())
        expected_series = pd.Series(
            [1, 2, 2, 4], index=["omega0", "omega1", "omega2", "omega3"]
        )
        expected_series.index.name = "Omega"
        pd.testing.assert_series_equal(result, expected_series)


class TestIterSampleFeatures:
    def test_iter_sample_features(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        indices = []
        for idx, _ in fss.feature_embedding.iter_sample_features():
            indices.append(idx)
        assert indices == ["omega0", "omega1", "omega2", "omega3"]


class TestEquality:
    def test_equal_fss(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fss1 = sa.FeaturizedSampleSpace.from_df(df)
        fss2 = sa.FeaturizedSampleSpace.from_df(df)
        assert fss1 == fss2

    def test_equal_fss_custom_names(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fss1 = sa.FeaturizedSampleSpace.from_df(
            df, embedding_name="Y", sample_prefix="s", sample_space_name="S"
        )
        fss2 = sa.FeaturizedSampleSpace.from_df(
            df, embedding_name="Y", sample_prefix="s", sample_space_name="S"
        )
        assert fss1 == fss2

    def test_not_equal_different_features(self):
        df1 = pd.DataFrame([[1, 2], [3, 4]])
        df2 = pd.DataFrame([[1, 2], [3, 5]])
        fss1 = sa.FeaturizedSampleSpace.from_df(df1)
        fss2 = sa.FeaturizedSampleSpace.from_df(df2)
        assert fss1 != fss2

    def test_not_equal_different_sample_space(self):
        df1 = pd.DataFrame([[1, 2], [3, 4]])
        df2 = pd.DataFrame([[1, 2], [3, 4]])
        fss1 = sa.FeaturizedSampleSpace.from_df(df1, sample_space_name="Omega")
        fss2 = sa.FeaturizedSampleSpace.from_df(df2, sample_space_name="S")
        assert fss1 != fss2

    def test_not_equal_different_embedding_name(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fss1 = sa.FeaturizedSampleSpace.from_df(df, embedding_name="X")
        fss2 = sa.FeaturizedSampleSpace.from_df(df, embedding_name="Y")
        assert fss1 != fss2

    def test_not_equal_different_type(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        fss = sa.FeaturizedSampleSpace.from_df(df)
        assert fss != "not a featurized sample space"
        assert fss != 42
        assert fss is not None


class TestSetters:
    def test_set_feature_embedding(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        df1 = pd.DataFrame([[1, 2], [3, 4]], index=["s0", "s1"], columns=["X0", "X1"])
        feature_embedding1 = sa.FeatureEmbedding(features=df1, name="X")
        fss = sa.FeaturizedSampleSpace(
            sample_space=sample_space, feature_embedding=feature_embedding1
        )
        df2 = pd.DataFrame([[5, 6], [7, 8]], index=["s0", "s1"], columns=["Y0", "Y1"])
        feature_embedding2 = sa.FeatureEmbedding(features=df2, name="Y")
        fss.feature_embedding = feature_embedding2
        expected_df = pd.DataFrame(
            data=[[5, 6], [7, 8]],
            index=["s0", "s1"],
            columns=["Y0", "Y1"],
        )
        pd.testing.assert_frame_equal(fss.feature_embedding.values, expected_df)
        assert fss.feature_embedding.name == "Y"


class TestAddProbabilityMeasure:
    def test_add_uniform_measure(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        fps = fss.add_probability_measure_from_features(lambda sf: 0.25)
        assert isinstance(fps, sa.FeaturizedProbabilitySpace)
        assert abs(fps.P("omega0") - 0.25) < 1e-10
        assert abs(fps.P("omega1") - 0.25) < 1e-10

    def test_add_custom_measure(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 3)

        def pmf(sf):
            num_ones = sf.sum()
            return 0.5**num_ones * 0.5 ** (3 - num_ones)

        fps = fss.add_probability_measure_from_features(pmf)
        assert abs(fps.P("omega0") - 0.125) < 1e-10


class TestValidation:
    def test_invalid_sample_space_type(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        feature_embedding = sa.FeatureEmbedding(features=df, name="X")
        with pytest.raises(TypeError, match="sample_space must be a SampleSpace"):
            sa.FeaturizedSampleSpace(
                sample_space="invalid", feature_embedding=feature_embedding
            )

    def test_invalid_embedding_type(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        with pytest.raises(TypeError, match="embedding must be a FeatureEmbedding"):
            sa.FeaturizedSampleSpace(
                sample_space=sample_space, feature_embedding="invalid"
            )

    def test_mismatched_indices(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"])
        feature_embedding = sa.FeatureEmbedding(features=df, name="X")
        with pytest.raises(ValueError, match="indices of embedding must match"):
            sa.FeaturizedSampleSpace(
                sample_space=sample_space, feature_embedding=feature_embedding
            )

    def test_set_embedding_mismatched_indices(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        df1 = pd.DataFrame([[1, 2], [3, 4]], index=["s0", "s1"])
        feature_embedding1 = sa.FeatureEmbedding(features=df1, name="X")
        fss = sa.FeaturizedSampleSpace(
            sample_space=sample_space, feature_embedding=feature_embedding1
        )
        df2 = pd.DataFrame([[5, 6], [7, 8]], index=["a", "b"])
        feature_embedding2 = sa.FeatureEmbedding(features=df2, name="Y")
        with pytest.raises(ValueError, match="indices of embedding must match"):
            fss.feature_embedding = feature_embedding2

    def test_from_sequences_non_iterable(self):
        with pytest.raises(TypeError, match="state_space must be an iterable"):
            sa.FeaturizedSampleSpace.from_sequences(123, 2)

    def test_from_sequences_empty_state_space(self):
        with pytest.raises(ValueError, match="state_space must be non-empty"):
            sa.FeaturizedSampleSpace.from_sequences([], 2)

    def test_from_sequences_invalid_sequence_length(self):
        with pytest.raises(
            ValueError, match="sequence_length must be a positive integer"
        ):
            sa.FeaturizedSampleSpace.from_sequences([0, 1], 0)

    def test_from_sequences_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be a positive integer"):
            sa.FeaturizedSampleSpace.from_sequences([0, 1], 2, threshold=0)

    def test_from_sequences_exceeds_threshold(self):
        with pytest.raises(ValueError, match="exceeds threshold"):
            sa.FeaturizedSampleSpace.from_sequences([0, 1], 20, threshold=10)

    def test_get_sample_features_invalid_index(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        with pytest.raises(ValueError, match="not found in sample_space"):
            fss.get_sample_features("invalid")

    def test_get_event_features_invalid_index(self):
        fss = sa.FeaturizedSampleSpace.from_sequences([0, 1], 2)
        with pytest.raises(ValueError, match="not found in sample_space"):
            fss.get_event_features(["omega0", "invalid"])
