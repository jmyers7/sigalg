import pytest

import sigalg as sa


class TestConstructor:

    def test_construction_with_default_sigma_algebra(self):
        sample_space = sa.SampleSpace(["omega0", "omega1"])
        event_space = sa.EventSpace(sample_space=sample_space)
        expected_sigma_algebra = sa.SigmaAlgebra.power_set(sample_space)
        assert event_space.sample_space == sample_space
        assert event_space.sigma_algebra == expected_sigma_algebra

    def test_construction_with_user_provided_sigma_algebra(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma_algebra = sa.SigmaAlgebra(
            sample_id_to_atom_id={"omega0": 0, "omega1": 1, "omega2": 1},
            sample_space=sample_space,
        )
        event_space = sa.EventSpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        assert event_space.sample_space == sample_space
        assert event_space.sigma_algebra == sigma_algebra


class TestSetters:

    def test_set_sigma_algebra(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        event_space = sa.EventSpace(sample_space=sample_space)
        new_sigma_algebra = sa.SigmaAlgebra(
            sample_id_to_atom_id={"omega0": 0, "omega1": 1, "omega2": 1},
            sample_space=sample_space,
        )
        event_space.sigma_algebra = new_sigma_algebra
        assert event_space.sigma_algebra == new_sigma_algebra


class TestEquality:

    def test_event_spaces_with_same_parameters_are_equal(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma_algebra = sa.SigmaAlgebra.power_set(sample_space)
        event_space1 = sa.EventSpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        event_space2 = sa.EventSpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        assert event_space1 == event_space2

    def test_event_spaces_with_different_parameters_are_not_equal(self):
        sample_space1 = sa.SampleSpace(["omega0", "omega1"])
        sample_space2 = sa.SampleSpace(["omega0", "omega1", "omega2"])
        event_space1 = sa.EventSpace(sample_space=sample_space1)
        event_space2 = sa.EventSpace(sample_space=sample_space2)
        assert event_space1 != event_space2


class TestValidation:

    def test_with_mismatched_sample_spaces(self):
        sample_space1 = sa.SampleSpace(["omega0", "omega1"])
        sample_space2 = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma_algebra = sa.SigmaAlgebra.power_set(sample_space2)
        with pytest.raises(
            ValueError,
            match="sigma_algebra's sample_space must match the provided sample_space.",
        ):
            sa.EventSpace(sample_space=sample_space1, sigma_algebra=sigma_algebra)
