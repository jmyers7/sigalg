import pytest

from sigalg.core import (
    Event,
    EventSpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="omega")

    def test_constructor_with_custom_sigma_algebra(self, sample_space):
        """Test constructor with custom sigma algebra."""
        custom_sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            {"omega_0": 0, "omega_1": 1, "omega_2": 1},
        )
        event_space = EventSpace(
            sample_space=sample_space, sig_alg=custom_sigma_algebra
        )

        assert event_space.sample_space == sample_space
        assert event_space.sig_alg == custom_sigma_algebra

    def test_constructor_with_default_sigma_algebra(self, sample_space):
        """Test constructor with default sigma algebra."""
        event_space = EventSpace(sample_space=sample_space)
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)

        assert event_space.sample_space == sample_space
        assert event_space.sig_alg == expected_sigma_algebra

    def test_invalid_wrong_type_raises(self, sample_space):
        """Test that invalid type for sigma_algebra raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            EventSpace(sample_space=sample_space, sig_alg="not a sigma algebra")

    def test_invalid_mismatched_sample_space_raises(self, sample_space):
        """Test that mismatched sample space raises ValueError."""
        mismatched_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        invalid_sigma_algebra = SigmaAlgebra(
            sample_space=mismatched_sample_space
        ).from_dict({"omega_0": 0, "omega_1": 0})
        with pytest.raises((TypeError, ValueError)):
            EventSpace(sample_space=sample_space, sig_alg=invalid_sigma_algebra)


def test_set_sigma_algebra():
    """Test that the sigma-algebra setter correctly updates the sigma-algebra."""
    sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="omega")
    event_space = EventSpace(sample_space=sample_space)
    new_sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
        {"omega_0": 0, "omega_1": 1, "omega_2": 1},
    )
    event_space.sig_alg = new_sigma_algebra
    assert event_space.sig_alg == new_sigma_algebra


class TestGetEventMethod:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4, initial_index=0, prefix="omega")

    @pytest.fixture
    def event_space(self, sample_space):
        return EventSpace(sample_space=sample_space)

    def test_get_event_subset_indices(self, event_space, sample_space):
        """Test get_event with subset of indices."""
        indices = ["omega_1", "omega_3"]
        name = "TestEvent"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(sig_alg=SigmaAlgebra.power_set(sample_space), name=name).from_list(
            indices,
        )

        assert event == expected_event

    def test_get_event_single_index(self, event_space, sample_space):
        """Test get_event with single index."""
        indices = ["omega_0"]
        name = "SingleEvent"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(sig_alg=SigmaAlgebra.power_set(sample_space), name=name).from_list(
            indices,
        )

        assert event == expected_event

    def test_get_event_empty_indices(self, event_space, sample_space):
        """Test get_event with empty list of indices."""
        indices = []
        name = "EmptyEvent"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(sig_alg=SigmaAlgebra.power_set(sample_space), name=name).from_list(
            indices,
        )

        assert event == expected_event

    def test_get_event_all_indices(self, event_space, sample_space):
        """Test get_event with all sample space indices."""
        indices = ["omega_0", "omega_1", "omega_2", "omega_3"]
        name = "FullEvent"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(sig_alg=SigmaAlgebra.power_set(sample_space), name=name).from_list(
            indices,
        )

        assert event == expected_event


class TestEquality:

    def test_non_equality_different_sample_spaces(self):
        """Test inequality when sample spaces are different."""
        given = EventSpace(
            sample_space=SampleSpace(name="Omega", data_name="sample").from_sequence(
                size=2,
                initial_index=0,
                prefix="omega"
            ),
        )
        other = EventSpace(
            sample_space=SampleSpace(name="Omega", data_name="sample").from_sequence(
                size=3,
                initial_index=0,
                prefix="omega"
            ),
        )
        assert given != other

    def test_non_equality_different_sigma_algebras(self):
        """Test inequality when sigma algebras are different."""
        sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="omega")
        given = EventSpace(
            sample_space=sample_space,
            sig_alg=SigmaAlgebra.power_set(sample_space),
        )
        other = EventSpace(
            sample_space=sample_space,
            sig_alg=SigmaAlgebra(sample_space=sample_space).from_dict(
                {"omega_0": 0, "omega_1": 0, "omega_2": 1}
            ),
        )
        assert given != other

    def test_non_equality_wrong_type(self):
        """Test inequality when comparing to wrong type."""
        given = EventSpace(
            sample_space=SampleSpace(name="Omega", data_name="sample").from_sequence(
                size=2,
                initial_index=0,
                prefix="omega"
            )
        )
        other = "not an event space"
        assert given != other

    def test_equality_same_parameters(self):
        """Test equality when parameters are the same."""
        sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="omega")
        given = EventSpace(
            sample_space=sample_space,
            sig_alg=SigmaAlgebra.power_set(sample_space),
        )
        other = EventSpace(
            sample_space=sample_space,
            sig_alg=SigmaAlgebra.power_set(sample_space),
        )
        assert given == other


def test_make_probability_space():
    """Test that make_probability_space creates a ProbabilitySpace correctly."""
    sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="s")
    sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
        {"s_0": 0, "s_1": 0, "s_2": 1}
    )
    event_space = EventSpace(sample_space=sample_space, sig_alg=sigma_algebra)
    custom_prob_measure = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(sample_space)).from_dict(
        {"s_0": 0.5, "s_1": 0.3, "s_2": 0.2},
    )
    uniform_prob_measure = ProbabilityMeasure.uniform(sig_alg=sigma_algebra)

    prob_space1 = event_space.make_probability_space(
        prob_measure=custom_prob_measure
    )
    prob_space2 = event_space.make_probability_space()

    assert isinstance(prob_space1, ProbabilitySpace)
    assert prob_space1.sample_space == sample_space
    assert prob_space1.sig_alg == sigma_algebra
    assert prob_space1.prob_measure == custom_prob_measure

    assert isinstance(prob_space2, ProbabilitySpace)
    assert prob_space2.sample_space == sample_space
    assert prob_space2.sig_alg == sigma_algebra
    assert prob_space2.prob_measure == uniform_prob_measure
