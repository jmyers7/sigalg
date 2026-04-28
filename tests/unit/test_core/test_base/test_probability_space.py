import pytest

from sigalg.core import (
    Event,
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)

    def test_constructor_all_parameters(self, Omega):
        """Test constructing ProbabilitySpace with all parameters."""
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        P = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.5, 1: 0.3, 2: 0.2})
        prob_space = ProbabilitySpace(Omega, F, P)

        assert prob_space.sample_space == Omega
        assert prob_space.sig_alg == F
        assert prob_space.prob_measure == P

    def test_constructor_defaults_only(self, Omega):
        """Test constructing ProbabilitySpace with defaults only."""
        prob_space = ProbabilitySpace(Omega)
        F_expected = SigmaAlgebra.power_set(Omega)
        P_expected = ProbabilityMeasure.uniform(sig_alg=SigmaAlgebra.power_set(Omega))

        assert prob_space.sample_space == Omega
        assert prob_space.sig_alg == F_expected
        assert prob_space.prob_measure == P_expected

    def test_constructor_custom_probabilities_only(self, Omega):
        """Test constructing ProbabilitySpace with custom probabilities only."""
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 1.0 / 3, 1: 1.0 / 3, 2: 1.0 / 3}
        )
        prob_space = ProbabilitySpace(Omega, prob_measure=P)
        F_expected = SigmaAlgebra.power_set(Omega)

        assert prob_space.sample_space == Omega
        assert prob_space.sig_alg == F_expected
        assert prob_space.prob_measure == P

    def test_constructor_custom_sigma_algebra_only(self, Omega):
        """Test constructing ProbabilitySpace with custom sigma algebra only."""
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        prob_space = ProbabilitySpace(Omega, sig_alg=F)
        P_expected = ProbabilityMeasure.uniform(sig_alg=F)

        assert prob_space.sample_space == Omega
        assert prob_space.sig_alg == F
        assert prob_space.prob_measure == P_expected

    def test_invalid_both_invalid_types_raises(self, Omega):
        """Test that invalid types for both parameters raise error."""
        with pytest.raises(TypeError):
            ProbabilitySpace(
                Omega,
                sig_alg="not_a_sigma_algebra",
                prob_measure="not_a_prob_measure",
            )

    def test_invalid_prob_measure_type_raises(self, Omega):
        """Test that invalid probability measure type raises error."""
        Omega_other = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=2
        )
        F = SigmaAlgebra(sample_space=Omega_other).from_dict({0: 0, 1: 0})

        with pytest.raises(TypeError):
            ProbabilitySpace(
                Omega,
                sig_alg=F,
                prob_measure="not_a_prob_measure",
            )

    def test_invalid_sigma_algebra_type_raises(self, Omega):
        """Test that invalid sigma algebra type raises error."""
        Omega_other = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=2
        )
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega_other)).from_dict(
            {0: 0.5, 1: 0.5}
        )

        with pytest.raises(TypeError):
            ProbabilitySpace(
                Omega,
                sig_alg="not_a_sigma_algebra",
                prob_measure=P,
            )

    def test_invalid_mismatched_prob_measure_sample_space_raises(self, Omega):
        """Test that mismatched probability measure sample space raises error."""
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        Omega_other = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=2
        )
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega_other)).from_dict(
            {0: 0.5, 1: 0.5}
        )

        with pytest.raises(ValueError):
            ProbabilitySpace(
                Omega,
                sig_alg=F,
                prob_measure=P,
            )

    def test_invalid_mismatched_sigma_algebra_sample_space_raises(self, Omega):
        """Test that mismatched sigma algebra sample space raises error."""
        Omega_other = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=2
        )
        F = SigmaAlgebra(sample_space=Omega_other).from_dict({0: 0, 1: 0})
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.5, 1: 0.5, 2: 0.0}
        )

        with pytest.raises(ValueError):
            ProbabilitySpace(
                Omega,
                sig_alg=F,
                prob_measure=P,
            )


class TestSetters:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)

    @pytest.fixture
    def prob_space(self, Omega):
        return ProbabilitySpace(Omega)

    def test_set_sigma_algebra_updates_sigma_algebra(self, Omega, prob_space):
        """Test setting a new sigma_algebra updates the ProbabilitySpace's sigma_algebra."""
        F_new = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

        prob_space.sig_alg = F_new

        assert prob_space.sig_alg == F_new

    def test_set_probability_measure_updates_probability_measure(
        self, Omega, prob_space
    ):
        """Test setting a new probability_measure updates the ProbabilitySpace's probability_measure."""
        P_new = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.4, 1: 0.4, 2: 0.2}
        )

        prob_space.prob_measure = P_new

        assert prob_space.prob_measure == P_new


def test_get_event():
    """Test that get_event returns an Event instance with correct indices."""
    Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
    prob_space = ProbabilitySpace(Omega)
    event = prob_space.get_event([0, 2])

    assert isinstance(event, Event)
    assert list(event.data) == [0, 2]


class TestConditionalProbability:
    def test_conditional_probability_basic(self):
        """Test conditional probability with basic intersection."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        )
        prob_space = ProbabilitySpace(Omega, prob_measure=P)
        A = prob_space.get_event([0], name="A")
        B = prob_space.get_event([0, 1], name="B")
        cond_prob = prob_space.conditional_probability(A, B)
        expected_prob = prob_space.prob_measure(A & B) / prob_space.prob_measure(B)

        assert abs(cond_prob - expected_prob) < 1e-10

    def test_conditional_probability_disjoint(self):
        """Test conditional probability with disjoint events."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        )
        prob_space = ProbabilitySpace(Omega, prob_measure=P)
        A = prob_space.get_event([0, 1], name="A")
        B = prob_space.get_event([2, 3], name="B")
        cond_prob = prob_space.conditional_probability(A, B)
        expected_prob = prob_space.prob_measure(A & B) / prob_space.prob_measure(B)

        assert abs(cond_prob - expected_prob) < 1e-10

    def test_conditional_probability_A_subset_of_B(self):
        """Test conditional probability when A is subset of B."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        )
        prob_space = ProbabilitySpace(Omega, prob_measure=P)
        A = prob_space.get_event([0], name="A")
        B = prob_space.get_event([0, 1, 2], name="B")
        cond_prob = prob_space.conditional_probability(A, B)
        expected_prob = prob_space.prob_measure(A & B) / prob_space.prob_measure(B)

        assert abs(cond_prob - expected_prob) < 1e-10


class TestAreIndependent:
    @pytest.fixture
    def prob_space(self):
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.25**2, 1: 0.75 * 0.25, 2: 0.25 * 0.75, 3: 0.75**2}
        )
        return ProbabilitySpace(Omega, prob_measure=P)

    def test_independent_events(self, prob_space):
        """Test that two independent events are correctly identified."""
        A = prob_space.get_event([0, 2], name="A")
        B = prob_space.get_event([0, 1], name="B")

        assert prob_space.are_independent(A, B)

    def test_dependent_events(self, prob_space):
        """Test that two dependent events are correctly identified."""
        A = prob_space.get_event([0, 2], name="A")
        B = prob_space.get_event([1, 3], name="B")

        assert not prob_space.are_independent(A, B)


class TestEquality:
    def test_non_equality_different_probability_measures(self):
        """Test inequality when probability measures are different."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        prob_space1 = ProbabilitySpace(
            Omega,
            prob_measure=ProbabilityMeasure(
                sig_alg=SigmaAlgebra.power_set(Omega)
            ).from_dict({0: 0.5, 1: 0.5}),
        )
        prob_space2 = ProbabilitySpace(
            Omega,
            prob_measure=ProbabilityMeasure(
                sig_alg=SigmaAlgebra.power_set(Omega)
            ).from_dict({0: 0.7, 1: 0.3}),
        )

        assert prob_space1 != prob_space2

    def test_non_equality_different_sigma_algebras(self):
        """Test inequality when sigma algebras are different."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        prob_space1 = ProbabilitySpace(
            Omega,
            sig_alg=SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1}),
        )
        prob_space2 = ProbabilitySpace(
            Omega,
            sig_alg=SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 1, 2: 1}),
        )

        assert prob_space1 != prob_space2

    def test_non_equality_different_sample_spaces(self):
        """Test inequality when sample spaces are different."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_list(["a", "b"])
        prob_space1 = ProbabilitySpace(Omega1)
        prob_space2 = ProbabilitySpace(Omega2)

        assert prob_space1 != prob_space2

    def test_non_equality_wrong_type_string(self):
        """Test inequality when comparing to string."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        prob_space = ProbabilitySpace(Omega)
        other = "not a probability space"

        assert prob_space != other

    def test_non_equality_wrong_type_int(self):
        """Test inequality when comparing to integer."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        prob_space = ProbabilitySpace(Omega)
        other = 123

        assert prob_space != other

    def test_equality_same_components(self):
        """Test equality when all components are the same."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 1})
        P = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.5, 1: 0.5})
        prob_space1 = ProbabilitySpace(Omega, F, P)
        prob_space2 = ProbabilitySpace(Omega, F, P)

        assert prob_space1 == prob_space2


class TestProbabilityAxioms:
    @pytest.fixture
    def prob_space(self):
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.5, 1: 0.3, 2: 0.2}
        )
        return ProbabilitySpace(Omega, prob_measure=P)

    @pytest.fixture
    def F(self, prob_space):
        return SigmaAlgebra.power_set(prob_space.sample_space)

    def test_axiom_non_negativity(self, prob_space):
        """Test that probabilities are non-negative."""
        for idx in prob_space.sample_space.data:
            assert prob_space.prob_measure(idx) >= 0

    def test_axiom_normalization(self, prob_space, F):
        """Test that the probability of the entire sample space is 1."""
        full_event = Event(sig_alg=F).from_list(list(prob_space.sample_space.data))

        assert abs(prob_space.prob_measure(full_event) - 1.0) < 1e-10

    def test_axiom_additivity_disjoint_events(self, prob_space, F):
        """Test that the probability of the union of disjoint events equals the sum of their probabilities."""
        A = Event(sig_alg=F).from_list([0])
        B = Event(sig_alg=F).from_list([1])
        union = A | B
        prob_union = prob_space.prob_measure(union)
        prob_sum = prob_space.prob_measure(A) + prob_space.prob_measure(B)

        assert abs(prob_union - prob_sum) < 1e-10

    def test_complement_rule(self, prob_space, F):
        """Test that the probability of an event and its complement sum to 1."""
        A = Event(sig_alg=F).from_list([0, 1])
        A_complement = ~A

        assert (
            abs(
                prob_space.prob_measure(A) + prob_space.prob_measure(A_complement) - 1.0
            )
            < 1e-10
        )
