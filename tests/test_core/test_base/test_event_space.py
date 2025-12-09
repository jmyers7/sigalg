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


class TestDataAccessMethods:

    def test_get_event_returns_correct_event(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        event_space = sa.EventSpace(sample_space=sample_space)
        event = event_space.get_event(["omega1", "omega3"], name="TestEvent")
        expected_event = sa.Event(
            sample_space=sample_space,
            event_indices=["omega1", "omega3"],
            name="TestEvent",
        )
        assert event == expected_event


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


class TestConversionMethods:

    def test_make_probability_space_with_default_probability_measure(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space,
            sample_id_to_atom_id={"s0": "A", "s1": "A", "s2": "B"},
        )
        event_space = sa.EventSpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        prob_space = event_space.make_probability_space()
        assert isinstance(prob_space, sa.ProbabilitySpace)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == sigma_algebra
        assert abs(prob_space.P("s0") - 1 / 3) < 1e-10
        assert abs(prob_space.P("s1") - 1 / 3) < 1e-10
        assert abs(prob_space.P("s2") - 1 / 3) < 1e-10

    def test_make_probability_space_with_custom_probability_measure(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        sigma_algebra = sa.SigmaAlgebra.power_set(sample_space)
        event_space = sa.EventSpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        probabilities = {"s0": 0.5, "s1": 0.3, "s2": 0.2}
        probability_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = event_space.make_probability_space(
            probability_measure=probability_measure
        )
        assert isinstance(prob_space, sa.ProbabilitySpace)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == sigma_algebra
        assert prob_space.probability_measure == probability_measure
        assert abs(prob_space.P("s0") - 0.5) < 1e-10
        assert abs(prob_space.P("s1") - 0.3) < 1e-10
        assert abs(prob_space.P("s2") - 0.2) < 1e-10

    def test_make_probability_space_preserves_sigma_algebra(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space,
            sample_id_to_atom_id={
                "omega0": 0,
                "omega1": 0,
                "omega2": 1,
                "omega3": 1,
            },
        )
        event_space = sa.EventSpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        prob_space = event_space.make_probability_space()
        assert prob_space.sigma_algebra == sigma_algebra
        event_01 = sample_space.get_event(["omega0", "omega1"])
        assert prob_space.is_measurable(event_01)
        event_23 = sample_space.get_event(["omega2", "omega3"])
        assert prob_space.is_measurable(event_23)

    def test_make_probability_space_with_trivial_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        sigma_algebra = sa.SigmaAlgebra.trivial(sample_space)
        event_space = sa.EventSpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        prob_space = event_space.make_probability_space()
        assert isinstance(prob_space, sa.ProbabilitySpace)
        assert prob_space.sigma_algebra == sigma_algebra
        full_event = sample_space.get_event(["s0", "s1", "s2"])
        assert prob_space.is_measurable(full_event)
        partial_event = sample_space.get_event(["s0"])
        assert not prob_space.is_measurable(partial_event)

    def test_make_probability_space_multiple_conversions(self):
        sample_space = sa.SampleSpace(["a", "b"])
        event_space = sa.EventSpace(sample_space=sample_space)
        prob_space1 = event_space.make_probability_space()
        prob_space2 = event_space.make_probability_space()
        assert prob_space1 == prob_space2
        assert prob_space1 is not prob_space2


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
