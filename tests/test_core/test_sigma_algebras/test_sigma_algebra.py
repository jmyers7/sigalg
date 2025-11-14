import sigalg as sa
import pytest


class TestConstructionAndBasicProperties:

    @pytest.fixture
    def space_features(self):
        data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        return sa.SampleSpaceFeatures(data)

    def test_construction(self, space_features):
        atom_ids = dict(zip(space_features.sample_index, [0, 0, 1, 1, 2]))
        sigma_alg = sa.SigmaAlgebra(
            sample_space_features=space_features, atom_ids=atom_ids
        )
        assert sigma_alg._sample_space_features == space_features
        assert sigma_alg._atom_ids == atom_ids


class TestMethods:

    @pytest.fixture
    def sample_space_features(self):
        data = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        return sa.SampleSpaceFeatures(data)

    @pytest.fixture
    def sigma_algebra(self, sample_space_features):
        atom_ids = dict(zip(sample_space_features.sample_index, [0, 0, 1, 1, 2]))
        return sa.SigmaAlgebra(
            sample_space_features=sample_space_features, atom_ids=atom_ids
        )

    def test_to_events(self, sample_space_features, sigma_algebra):
        events = sigma_algebra.to_events()
        expected_events = {
            0: sa.EventFeatures(
                sample_space_features=sample_space_features,
                sample_index=["omega0", "omega1"],
            ),
            1: sa.EventFeatures(
                sample_space_features=sample_space_features,
                sample_index=["omega2", "omega3"],
            ),
            2: sa.EventFeatures(
                sample_space_features=sample_space_features, sample_index=["omega4"]
            ),
        }
        assert events == expected_events

    def test_is_measurable(self, sample_space_features, sigma_algebra):
        event = sa.EventFeatures(
            sample_space_features=sample_space_features,
            sample_index=["omega0", "omega1"],
        )
        assert sigma_algebra.is_measurable(event) is True

        non_measurable_event = sa.EventFeatures(
            sample_space_features=sample_space_features,
            sample_index=["omega0", "omega2"],
        )
        assert sigma_algebra.is_measurable(non_measurable_event) is False
