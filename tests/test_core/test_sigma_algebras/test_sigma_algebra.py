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

    def test_iter(self, sigma_algebra):
        events = dict(sigma_algebra)
        expected_events = sigma_algebra.to_events()
        assert events == expected_events

    def test_eq(self, sample_space):
        atom_ids1 = dict(zip(sample_space, [0, 0, 1, 1, 2]))
        sigma_alg1 = sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids1)

        atom_ids2 = dict(zip(sample_space, [0, 0, 1, 1, 2]))
        sigma_alg2 = sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids2)

        atom_ids3 = dict(zip(sample_space, [0, 1, 1, 1, 2]))
        sigma_alg3 = sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids3)

        assert sigma_alg1 == sigma_alg2
        assert sigma_alg1 != sigma_alg3
        assert sigma_alg1 != "not a sigma algebra"


class TestValidation:

    @pytest.fixture
    def sample_space(self):
        indices = ["omega0", "omega1", "omega2", "omega3", "omega4"]
        return sa.SampleSpace(indices)

    def test_invalid_sample_space(self):
        atom_ids = {f"omega{i}": i // 2 for i in range(5)}
        with pytest.raises(TypeError):
            sa.SigmaAlgebra(sample_space="not a sample space", atom_ids=atom_ids)

    def test_invalid_atom_ids_type(self, sample_space):
        with pytest.raises(TypeError):
            sa.SigmaAlgebra(sample_space=sample_space, atom_ids="not a dict")

    def test_invalid_atom_id_in_sample_space(self, sample_space):
        atom_ids = {f"omega{i}": i // 2 for i in range(4)}
        with pytest.raises(ValueError):
            sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids)

    def test_is_measurable_invalid_types(self, sample_space):
        atom_ids = dict(zip(sample_space, [0, 0, 1, 1, 2]))
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space,
            atom_ids=atom_ids,
        )
        with pytest.raises(TypeError):
            sigma_algebra.is_measurable(event="not an event")

        other_sample_space = sa.SampleSpace([f"omega{i}" for i in range(3)])
        event = sa.Event(sample_space=other_sample_space, event_indices=["omega0"])
        with pytest.raises(ValueError):
            sigma_algebra.is_measurable(event=event)

        with pytest.raises(TypeError):
            sa.is_measurable(sigma_algebra="not a sigma algebra", event=event)
