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

    def test_constructor_with_no_parameters(self):
        """Test constructor with no parameters."""
        event_space = EventSpace()

        assert event_space.sample_space is None
        assert event_space.sig_alg is None

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


class TestSetters:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)

    @pytest.fixture
    def event_space(self, Omega):
        return EventSpace(sample_space=Omega)

    def test_set_sample_space_creates_power_set(self):
        """Test that setting sample_space creates power-set sigma-algebra."""
        Omega1 = SampleSpace().from_sequence(size=2)
        event_space = EventSpace(sample_space=Omega1)
        Omega2 = SampleSpace().from_sequence(size=4)

        event_space.sample_space = Omega2

        assert event_space.sample_space == Omega2
        assert event_space.sig_alg == SigmaAlgebra.power_set(Omega2)

    def test_set_sample_space_invalid_type_raises(self, event_space):
        """Test that setting sample_space with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="must be a SampleSpace instance"):
            event_space.sample_space = "not a sample space"

    def test_set_sig_alg_with_sample_space(self, Omega, event_space):
        """Test that setting sig_alg with existing sample_space validates sample space match."""
        F_new = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 1, 2: 1})

        event_space.sig_alg = F_new

        assert event_space.sig_alg == F_new
        assert event_space.sample_space == Omega

    def test_set_sig_alg_without_sample_space_sets_sample_space(self):
        """Test that setting sig_alg without sample_space sets sample_space from sigma-algebra."""
        event_space = EventSpace()
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 1, 2: 1})

        event_space.sig_alg = F

        assert event_space.sample_space == Omega
        assert event_space.sig_alg == F

    def test_set_sig_alg_with_mismatched_sample_space_raises(self, event_space):
        """Test that setting sig_alg with mismatched sample space raises ValueError."""
        Omega_other = SampleSpace().from_sequence(size=2)
        F_other = SigmaAlgebra(sample_space=Omega_other).from_dict({0: 0, 1: 1})

        with pytest.raises(
            ValueError, match="sample_space must match the provided sample_space"
        ):
            event_space.sig_alg = F_other

    def test_set_sig_alg_invalid_type_raises(self, event_space):
        """Test that setting sig_alg with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="must be a SigmaAlgebra instance"):
            event_space.sig_alg = "not a sigma algebra"


class TestGetEventMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(sample_space=Omega)

    @pytest.fixture
    def event_space(self, Omega, F):
        return EventSpace(sample_space=Omega, sig_alg=F)

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
        F1 = SigmaAlgebra.power_set(Omega1)
        F2 = SigmaAlgebra.power_set(Omega2)
        event_space1 = EventSpace(sample_space=Omega1, sig_alg=F1)
        event_space2 = EventSpace(sample_space=Omega2, sig_alg=F2)

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


class TestFromDict:
    def test_from_dict_with_no_parameters_at_initialization(self):
        """Test from_dict when event space is initialized with no parameters."""
        event_space = EventSpace()
        sample_id_to_atom_id = {0: 0, 1: 0, 2: 1}
        event_space.from_dict(sample_id_to_atom_id)
        expected_sample_space = SampleSpace().from_sequence(size=3)
        expected_sig_alg = SigmaAlgebra().from_dict(sample_id_to_atom_id)

        assert event_space.sig_alg == expected_sig_alg
        assert event_space.sample_space == expected_sample_space

    def test_from_dict_with_existing_sample_space(self):
        """Test from_dict when event space is initialized with a sample space."""
        Omega = SampleSpace().from_sequence(size=3)
        event_space = EventSpace(sample_space=Omega)
        sample_id_to_atom_id = {0: 0, 1: 0, 2: 1}
        event_space.from_dict(sample_id_to_atom_id)
        expected_sig_alg = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id
        )

        assert event_space.sig_alg == expected_sig_alg
        assert event_space.sample_space is Omega


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
