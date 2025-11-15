import sigalg as sa
import pytest


class TestConstructionAndBasicProperties:

    @pytest.fixture
    def sample_space(self):
        indices = [f"omega{i}" for i in range(5)]
        return sa.SampleSpace(indices)

    def test_construction(self, sample_space):
        atom_ids = dict(zip(sample_space, [0, 0, 1, 1, 2]))
        sigma_alg = sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids)
        assert sigma_alg._sample_space == sample_space
        assert sigma_alg._atom_ids == atom_ids


class TestMethods:

    @pytest.fixture
    def sample_space(self):
        indices = [f"omega{i}" for i in range(5)]
        return sa.SampleSpace(indices)

    @pytest.fixture
    def sigma_algebra(self, sample_space):
        atom_ids = dict(zip(sample_space, [0, 0, 1, 1, 2]))
        return sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids)

    def test_to_events(self, sample_space, sigma_algebra):
        events = sigma_algebra.to_events()
        expected_events = {
            0: sa.Event(
                sample_space=sample_space,
                event_indices=["omega0", "omega1"],
            ),
            1: sa.Event(
                sample_space=sample_space,
                event_indices=["omega2", "omega3"],
            ),
            2: sa.Event(sample_space=sample_space, event_indices=["omega4"]),
        }
        assert events == expected_events

    def test_is_measurable(self, sample_space, sigma_algebra):
        event = sa.Event(
            sample_space=sample_space,
            event_indices=["omega0", "omega1"],
        )
        assert sigma_algebra.is_measurable(event) is True

        non_measurable_event = sa.Event(
            sample_space=sample_space,
            event_indices=["omega0", "omega2"],
        )
        assert sigma_algebra.is_measurable(non_measurable_event) is False
