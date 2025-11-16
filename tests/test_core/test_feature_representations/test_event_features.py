import pandas as pd
import pytest

import sigalg as sa


class TestConstructionAndBasicProperties:

    @pytest.fixture
    def sample_space_features(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3", "s4"])
        features = [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        feature_index = ["F0", "F1"]
        return sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space, feature_index=feature_index
        )

    def test_construction_with_single_index(self, sample_space_features):
        event_indices = ["s1"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        assert ef.sample_space == sample_space_features.sample_space
        assert ef.event == sa.Event(
            sample_space=sample_space_features.sample_space,
            event_indices=event_indices,
        )
        expected_df = pd.DataFrame(data=[[30, 40]], index=["s1"], columns=["F0", "F1"])
        pd.testing.assert_frame_equal(ef._values, expected_df)

    def test_construction_with_multiple_indices(self, sample_space_features):
        event_indices = ["s0", "s2", "s4"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        assert ef.sample_space == sample_space_features.sample_space
        assert ef.event == sa.Event(
            sample_space=sample_space_features.sample_space,
            event_indices=event_indices,
        )
        expected_df = pd.DataFrame(
            data=[[10, 20], [50, 60], [90, 100]],
            index=["s0", "s2", "s4"],
            columns=["F0", "F1"],
        )
        pd.testing.assert_frame_equal(ef._values, expected_df)

    def test_construction_with_all_indices(self, sample_space_features):
        event_indices = ["s0", "s1", "s2", "s3", "s4"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        assert ef.sample_space == sample_space_features.sample_space
        pd.testing.assert_frame_equal(ef._values, sample_space_features._values)

    def test_construction_with_duplicate_indices(self, sample_space_features):
        event_indices = ["s1", "s1", "s3"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        # Event object may deduplicate, so use event.index for expected data
        expected_df = sample_space_features._values.loc[ef.event.index].copy()
        pd.testing.assert_frame_equal(ef._values, expected_df)

    def test_construction_with_out_of_order_indices(self, sample_space_features):
        event_indices = ["s3", "s1", "s4"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        # Use event.index to get the actual order after Event construction
        expected_df = sample_space_features._values.loc[ef.event.index].copy()
        pd.testing.assert_frame_equal(ef._values, expected_df)

    def test_sample_space_property(self, sample_space_features):
        event_indices = ["s1", "s2"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        assert ef.sample_space == sample_space_features.sample_space
        assert isinstance(ef.sample_space, sa.SampleSpace)

    def test_event_property(self, sample_space_features):
        event_indices = ["s1", "s2"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        expected_event = sa.Event(
            sample_space=sample_space_features.sample_space,
            event_indices=event_indices,
        )
        assert ef.event == expected_event
        assert isinstance(ef.event, sa.Event)

    def test_sample_space_features_property(self, sample_space_features):
        event_indices = ["s1", "s2"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        assert isinstance(ef.sample_space_features, sa.SampleSpaceFeatures)
        assert ef.sample_space_features == sample_space_features

    def test_values_is_independent_copy(self, sample_space_features):
        event_indices = ["s1", "s2"]
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=event_indices,
        )
        # Modify the event features data
        ef._values.iloc[0, 0] = 999
        # Original should be unchanged
        assert sample_space_features._values.loc["s1", "F0"] == 30


class TestValidation:

    @pytest.fixture
    def sample_space_features(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        features = [[10, 20], [30, 40], [50, 60]]
        return sa.SampleSpaceFeatures(features=features, sample_space=sample_space)

    def test_invalid_sample_space_features_type(self):
        with pytest.raises(
            TypeError, match="sample_space_features must be an instance"
        ):
            sa.EventFeatures(
                sample_space_features="invalid_type",
                event_indices=["s0", "s1"],
            )

    def test_invalid_event_indices_type(self, sample_space_features):
        with pytest.raises(TypeError, match="must be a list of sample indices"):
            sa.EventFeatures(
                sample_space_features=sample_space_features,
                event_indices="s0",
            )

    def test_invalid_event_indices_type_tuple(self, sample_space_features):
        with pytest.raises(TypeError, match="must be a list of sample indices"):
            sa.EventFeatures(
                sample_space_features=sample_space_features,
                event_indices=("s0", "s1"),
            )

    def test_invalid_sample_index_not_in_sample_space(self, sample_space_features):
        with pytest.raises(ValueError, match="Sample index s5 not found"):
            sa.EventFeatures(
                sample_space_features=sample_space_features,
                event_indices=["s0", "s5"],
            )

    def test_empty_event_indices_list(self, sample_space_features):
        # Empty list should be valid - creates an empty event
        ef = sa.EventFeatures(
            sample_space_features=sample_space_features,
            event_indices=[],
        )
        assert len(ef._values) == 0


class TestGetItemAndGetSampleFeatures:

    @pytest.fixture
    def event_features(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3", "s4"])
        features = [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        feature_index = ["F0", "F1"]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space, feature_index=feature_index
        )
        event_indices = ["s1", "s2", "s4"]
        return sa.EventFeatures(
            sample_space_features=space_features,
            event_indices=event_indices,
        )

    def test_getitem_with_one_string_index(self, event_features):
        sf = event_features["s2"]
        expected_sf = sa.SampleFeatures(
            features=[50, 60], sample_index="s2", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_getitem_with_first_index(self, event_features):
        sf = event_features["s1"]
        expected_sf = sa.SampleFeatures(
            features=[30, 40], sample_index="s1", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_getitem_with_last_index(self, event_features):
        sf = event_features["s4"]
        expected_sf = sa.SampleFeatures(
            features=[90, 100], sample_index="s4", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_get_sample_features(self, event_features):
        sf = event_features.get_sample_features("s2")
        expected_sf = sa.SampleFeatures(
            features=[50, 60], sample_index="s2", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf


class TestGetSampleFeaturesAt:

    @pytest.fixture
    def event_features(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3", "s4"])
        features = [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]
        feature_index = ["F0", "F1"]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space, feature_index=feature_index
        )
        event_indices = ["s1", "s2", "s4"]
        return sa.EventFeatures(
            sample_space_features=space_features,
            event_indices=event_indices,
        )

    def test_iloc_indexer_with_integer(self, event_features):
        sf = event_features.get_sample_features_at[0]
        expected_sf = sa.SampleFeatures(
            features=[30, 40], sample_index="s1", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_iloc_indexer_with_integer_middle(self, event_features):
        sf = event_features.get_sample_features_at[1]
        expected_sf = sa.SampleFeatures(
            features=[50, 60], sample_index="s2", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_iloc_indexer_with_integer_last(self, event_features):
        sf = event_features.get_sample_features_at[2]
        expected_sf = sa.SampleFeatures(
            features=[90, 100], sample_index="s4", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_iloc_indexer_with_negative_integer(self, event_features):
        sf = event_features.get_sample_features_at[-1]
        expected_sf = sa.SampleFeatures(
            features=[90, 100], sample_index="s4", feature_index=["F0", "F1"]
        )
        assert sf == expected_sf

    def test_iloc_indexer_with_slice(self, event_features):
        ef_slice = event_features.get_sample_features_at[0:2]
        expected_ef = sa.EventFeatures(
            sample_space_features=event_features.sample_space_features,
            event_indices=["s1", "s2"],
        )
        assert ef_slice == expected_ef

    def test_iloc_indexer_with_slice_all(self, event_features):
        ef_slice = event_features.get_sample_features_at[:]
        assert ef_slice == event_features

    def test_iloc_indexer_with_list_of_integers(self, event_features):
        ef_subset = event_features.get_sample_features_at[[0, 2]]
        expected_ef = sa.EventFeatures(
            sample_space_features=event_features.sample_space_features,
            event_indices=["s1", "s4"],
        )
        assert ef_subset == expected_ef

    def test_iloc_indexer_with_single_element_list(self, event_features):
        ef_subset = event_features.get_sample_features_at[[1]]
        expected_ef = sa.EventFeatures(
            sample_space_features=event_features.sample_space_features,
            event_indices=["s2"],
        )
        assert ef_subset == expected_ef


class TestEdgeCases:

    def test_single_sample_event(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        features = [[10, 20], [30, 40], [50, 60]]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space
        )
        ef = sa.EventFeatures(
            sample_space_features=space_features,
            event_indices=["s1"],
        )
        assert len(ef._values) == 1
        assert ef._values.loc["s1", "X0"] == 30
        assert ef._values.loc["s1", "X1"] == 40

    def test_single_feature_dimension(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        features = [[10], [30], [50]]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space
        )
        ef = sa.EventFeatures(
            sample_space_features=space_features,
            event_indices=["s0", "s2"],
        )
        assert ef._values.shape == (2, 1)
        expected_df = pd.DataFrame(data=[[10], [50]], index=["s0", "s2"], columns=["X"])
        pd.testing.assert_frame_equal(ef._values, expected_df)

    def test_numeric_sample_indices(self):
        sample_space = sa.SampleSpace([0, 1, 2, 3])
        features = [[10, 20], [30, 40], [50, 60], [70, 80]]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space
        )
        ef = sa.EventFeatures(
            sample_space_features=space_features,
            event_indices=[1, 3],
        )
        expected_df = pd.DataFrame(
            data=[[30, 40], [70, 80]], index=[1, 3], columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(ef._values, expected_df)
