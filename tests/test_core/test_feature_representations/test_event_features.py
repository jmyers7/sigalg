import pytest
import sigalg as sa
import pandas as pd


class TestConstructionAndBasicProperties:

    @pytest.fixture
    def sample_space_features(self):
        data = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=["s0", "s1", "s2"],
            columns=["Y0", "Y1"],
        )
        return sa.SampleSpaceFeatures(features=data)

    def test_basic_construction(self, sample_space_features):
        event_features = sa.EventFeatures(
            sample_space_features=sample_space_features, event_indices=["s1", "s2"]
        )
        assert event_features.sample_space_features == sample_space_features
        assert event_features.n_samples == 2
        assert event_features.n_features == 2
        assert event_features.sample_index[0] == "s1"
        assert event_features.sample_index[1] == "s2"
        assert event_features.feature_index[0] == "Y0"
        assert event_features.feature_index[1] == "Y1"
        assert event_features.shape == (2, 2)
        assert len(event_features) == 2

    def test_from_sample_space_method(self, sample_space_features):
        event_features = sample_space_features.get_event(["s1", "s2"])
        assert isinstance(event_features, sa.EventFeatures)
        assert event_features.sample_space_features == sample_space_features
        assert event_features.n_samples == 2
        assert event_features.n_features == 2
        assert event_features.sample_index[0] == "s1"
        assert event_features.sample_index[1] == "s2"
        assert event_features.feature_index[0] == "Y0"
        assert event_features.feature_index[1] == "Y1"
        assert event_features.shape == (2, 2)
