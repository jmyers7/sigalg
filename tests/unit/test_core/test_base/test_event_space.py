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
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)

    def test_constructor_with_custom_sigma_algebra(self, Omega):
        """Test constructor with custom sigma algebra."""
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 1, 2: 1})
        event_space = EventSpace(sample_space=Omega, sig_alg=F)

        assert event_space.sample_space == Omega
        assert event_space.sig_alg == F

    def test_constructor_with_default_sigma_algebra(self, Omega):
        """Test constructor with default sigma algebra."""
        event_space = EventSpace(sample_space=Omega)
        F_expected = SigmaAlgebra.power_set(Omega)

        assert event_space.sample_space == Omega
        assert event_space.sig_alg == F_expected

    def test_invalid_wrong_type_raises(self, Omega):
        """Test that invalid type for sigma_algebra raises TypeError."""
        with pytest.raises(TypeError):
            EventSpace(sample_space=Omega, sig_alg="not a sigma algebra")

    def test_invalid_mismatched_sample_space_raises(self, Omega):
        """Test that mismatched sample space raises ValueError."""
        Omega_mismatched = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=2
        )
        F_invalid = SigmaAlgebra(sample_space=Omega_mismatched).from_dict({0: 0, 1: 0})

        with pytest.raises(ValueError):
            EventSpace(sample_space=Omega, sig_alg=F_invalid)


def test_set_sigma_algebra():
    """Test that the sigma-algebra setter correctly updates the sigma-algebra."""
    Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
    event_space = EventSpace(sample_space=Omega)
    F_new = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 1, 2: 1})

    event_space.sig_alg = F_new

    assert event_space.sig_alg == F_new


class TestGetEventMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def event_space(self, Omega):
        return EventSpace(sample_space=Omega)

    def test_get_event_subset_indices(self, event_space, Omega):
        """Test get_event with subset of indices."""
        indices = [1, 3]
        name = "A"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(
            sig_alg=SigmaAlgebra.power_set(Omega), name=name
        ).from_list(indices)

        assert event == expected_event

    def test_get_event_single_index(self, event_space, Omega):
        """Test get_event with single index."""
        indices = [0]
        name = "B"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(
            sig_alg=SigmaAlgebra.power_set(Omega), name=name
        ).from_list(indices)

        assert event == expected_event

    def test_get_event_empty_indices(self, event_space, Omega):
        """Test get_event with empty list of indices."""
        indices = []
        name = "C"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(
            sig_alg=SigmaAlgebra.power_set(Omega), name=name
        ).from_list(indices)

        assert event == expected_event

    def test_get_event_all_indices(self, event_space, Omega):
        """Test get_event with all sample space indices."""
        indices = [0, 1, 2, 3]
        name = "D"
        event = event_space.get_event(indices, name=name)
        expected_event = Event(
            sig_alg=SigmaAlgebra.power_set(Omega), name=name
        ).from_list(indices)

        assert event == expected_event


class TestEquality:
    def test_non_equality_different_sample_spaces(self):
        """Test inequality when sample spaces are different."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        event_space1 = EventSpace(sample_space=Omega1)
        event_space2 = EventSpace(sample_space=Omega2)

        assert event_space1 != event_space2

    def test_non_equality_different_sigma_algebras(self):
        """Test inequality when sigma algebras are different."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        F1 = SigmaAlgebra.power_set(Omega)
        F2 = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        event_space1 = EventSpace(sample_space=Omega, sig_alg=F1)
        event_space2 = EventSpace(sample_space=Omega, sig_alg=F2)

        assert event_space1 != event_space2

    def test_non_equality_wrong_type(self):
        """Test inequality when comparing to wrong type."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        event_space = EventSpace(sample_space=Omega)
        other = "not an event space"

        assert event_space != other

    def test_equality_same_parameters(self):
        """Test equality when parameters are the same."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        F = SigmaAlgebra.power_set(Omega)
        event_space1 = EventSpace(sample_space=Omega, sig_alg=F)
        event_space2 = EventSpace(sample_space=Omega, sig_alg=F)

        assert event_space1 == event_space2


def test_make_probability_space():
    """Test that make_probability_space creates a ProbabilitySpace correctly."""
    Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
    F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
    event_space = EventSpace(sample_space=Omega, sig_alg=F)
    P_custom = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.5, 1: 0.3, 2: 0.2})
    P_uniform = ProbabilityMeasure.uniform(sig_alg=F)
    prob_space1 = event_space.make_probability_space(prob_measure=P_custom)
    prob_space2 = event_space.make_probability_space()

    assert isinstance(prob_space1, ProbabilitySpace)
    assert prob_space1.sample_space == Omega
    assert prob_space1.sig_alg == F
    assert prob_space1.prob_measure == P_custom
    assert isinstance(prob_space2, ProbabilitySpace)
    assert prob_space2.sample_space == Omega
    assert prob_space2.sig_alg == F
    assert prob_space2.prob_measure == P_uniform
