import pandas as pd
import pytest

import sigalg as sa


class TestConstructionAndBasicProperties:

    def test_construction_from_sample_space(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        features = [[10, 20], [30, 40], [50, 60]]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space
        )
        assert space_features._sample_space == sample_space
        expected_df = pd.DataFrame(
            data=features, index=sample_space.index, columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_from_data_with_generated_indices(self):
        data = [[1, 2], [3, 4]]
        space_features = sa.SampleSpaceFeatures(
            features=data,
            overwrite_default_sample_space=True,
            overwrite_default_feature_index=True,
        )
        expected_sample_space = sa.SampleSpace(["omega0", "omega1"])
        expected_df = pd.DataFrame(
            data=data,
            index=expected_sample_space.index,
            columns=["X0", "X1"],
        )
        assert space_features._sample_space == expected_sample_space
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_from_data_with_default_indices(self):
        data = [[1, 2], [3, 4]]
        space_features = sa.SampleSpaceFeatures(
            features=data,
            overwrite_default_sample_space=False,
            overwrite_default_feature_index=False,
        )
        expected_sample_space = sa.SampleSpace([0, 1])
        expected_df = pd.DataFrame(
            data=data,
            index=[0, 1],
            columns=[0, 1],
        )
        assert space_features._sample_space == expected_sample_space
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_with_custom_indices(self):
        data = [[1, 2], [3, 4]]
        sample_index = ["s0", "s1"]
        feature_index = ["T0", "T1"]
        space_features = sa.SampleSpaceFeatures(
            data,
            sample_space=sa.SampleSpace(sample_index),
            feature_index=feature_index,
        )
        expected_df = pd.DataFrame(
            data=data,
            index=sample_index,
            columns=feature_index,
        )
        assert space_features._sample_space == sa.SampleSpace(sample_index)
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_with_custom_prefixes_and_initial_indices(self):
        data = [[1, 2], [3, 4]]
        space_features = sa.SampleSpaceFeatures(
            features=data,
            sample_prefix="sample_",
            feature_prefix="feat_",
            initial_sample_index=1,
            initial_feature_index=1,
        )
        expected_sample_space = sa.SampleSpace(["sample_1", "sample_2"])
        expected_df = pd.DataFrame(
            data=data,
            index=expected_sample_space.index,
            columns=["feat_1", "feat_2"],
        )
        assert space_features._sample_space == expected_sample_space
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_from_sequences(self):
        state_space = [0, 1]
        sequence_length = 3
        space_features = sa.SampleSpaceFeatures.from_sequences(
            state_space=state_space,
            sequence_length=sequence_length,
            sample_prefix="s",
            feature_prefix="F",
            threshold=10,
        )
        expected_sample_space = sa.SampleSpace(
            ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        )
        expected_space_features = sa.SampleSpaceFeatures(
            features=[
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
            sample_space=expected_sample_space,
            feature_index=["F0", "F1", "F2"],
        )
        assert space_features == expected_space_features
        assert space_features.sample_space == expected_sample_space

    def test_from_sequences_with_initial_feature_index(self):
        state_space = ["a", "b"]
        sequence_length = 2
        space_features = sa.SampleSpaceFeatures.from_sequences(
            state_space=state_space,
            sequence_length=sequence_length,
            initial_sample_index=1,
            initial_feature_index=1,
            sample_prefix="sample_",
            feature_prefix="feature_",
        )
        expected_sample_space = sa.SampleSpace(
            ["sample_1", "sample_2", "sample_3", "sample_4"]
        )
        expected_space_features = sa.SampleSpaceFeatures(
            features=[
                ("a", "a"),
                ("a", "b"),
                ("b", "a"),
                ("b", "b"),
            ],
            sample_space=expected_sample_space,
            feature_index=["feature_1", "feature_2"],
        )
        assert space_features == expected_space_features
        assert space_features.sample_space == expected_sample_space


class TestGetItemAndGetSampleFeatures:

    @pytest.fixture
    def sample_space_features(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        features = [[10, 20], [30, 40], [50, 60]]
        feature_index = ["F0", "F1"]
        return sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space, feature_index=feature_index
        )

    def test_getitem_with_one_string_index(self, sample_space_features):
        sf = sample_space_features["s1"]
        expected_sf = sa.SampleFeatures(
            features=[30, 40], sample_index="s1", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_get_sample_features_with_one_string_index(self, sample_space_features):
        sf = sample_space_features.get_sample_features("s2")
        expected_sf = sa.SampleFeatures(
            features=[50, 60], sample_index="s2", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf
