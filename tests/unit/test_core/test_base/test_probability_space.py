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

    def test_set_probability_measure_updates_probability_measure(
        self, Omega, prob_space
    ):
        """Test setting a new probability_measure updates the ProbabilitySpace's probability_measure."""
        P_new = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.4, 1: 0.4, 2: 0.2}
        )

        prob_space.prob_measure = P_new

        assert prob_space.prob_measure == P_new

    def test_set_sample_space_creates_new_power_set_and_uniform(
        self,
    ):
        """Test that setting sample_space creates new power-set sigma-algebra and uniform probability measure."""
        Omega1 = SampleSpace(name="Omega1", data_name="sample").from_sequence(size=2)
        prob_space = ProbabilitySpace(Omega1)
        Omega2 = SampleSpace(name="Omega2", data_name="sample").from_sequence(size=4)

        prob_space.sample_space = Omega2

        assert prob_space.sample_space == Omega2
        assert prob_space.sig_alg == SigmaAlgebra.power_set(Omega2)
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(Omega2)
        )

    def test_set_sample_space_invalid_type_raises(self, prob_space):
        """Test that setting sample_space with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="must be a SampleSpace instance"):
            prob_space.sample_space = "not a sample space"

    def test_set_prob_measure_without_sample_space_raises(self):
        """Test that setting prob_measure without sample_space raises ValueError."""
        prob_space = ProbabilitySpace()
        Omega = SampleSpace().from_sequence(size=2)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.5, 1: 0.5}
        )

        with pytest.raises(ValueError, match="Cannot set prob_measure without"):
            prob_space.prob_measure = P

    def test_set_prob_measure_with_mismatched_sample_space_raises(self, prob_space):
        """Test that setting prob_measure with mismatched sample space raises ValueError."""
        Omega_other = SampleSpace().from_sequence(size=2)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega_other)).from_dict(
            {0: 0.5, 1: 0.5}
        )

        with pytest.raises(
            ValueError, match="probability measure must be defined on the given"
        ):
            prob_space.prob_measure = P


class TestFromDict:
    def test_from_dict_creates_power_set_and_prob_measure(self):
        """Test that from_dict creates power-set sigma-algebra and probability measure."""
        probabilities = {0: 0.3, 1: 0.5, 2: 0.2}
        prob_space = ProbabilitySpace().from_dict(probabilities=probabilities)

        assert prob_space.sample_space == SampleSpace().from_list([0, 1, 2])
        assert prob_space.sig_alg.is_power_set
        assert prob_space.prob_measure.probabilities == probabilities

    def test_from_dict_with_existing_sample_space(self):
        """Test from_dict with existing sample space."""
        Omega = SampleSpace().from_sequence(size=3)
        probabilities = {0: 0.4, 1: 0.3, 2: 0.3}
        prob_space = ProbabilitySpace(Omega).from_dict(probabilities=probabilities)

        assert prob_space.sample_space == Omega
        assert prob_space.prob_measure.probabilities == probabilities

    def test_from_dict_mismatched_keys_raises(self):
        """Test that from_dict with mismatched keys raises ValueError."""
        Omega = SampleSpace().from_sequence(size=3)
        probabilities = {0: 0.5, 1: 0.5}
        prob_space = ProbabilitySpace(Omega)

        with pytest.raises(ValueError, match="elements must match the keys"):
            prob_space.from_dict(probabilities=probabilities)

    def test_from_dict_invalid_type_raises(self):
        """Test that from_dict with non-dict raises TypeError."""
        prob_space = ProbabilitySpace()

        with pytest.raises(TypeError, match="must be a dictionary"):
            prob_space.from_dict(probabilities=[0.3, 0.5, 0.2])


def test_get_event():
    """Test that get_event returns an Event instance with correct indices."""
    Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
    prob_space = ProbabilitySpace(Omega)
    event = prob_space.get_event([0, 2])

    assert isinstance(event, Event)
    assert list(event.data) == [0, 2]


class TestFromEvent:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        atom_ids = {0: 0, 1: 1, 2: 1, 3: 2}
        return SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    @pytest.fixture
    def P(self, F):
        probabilities = {0: 0.15, 1: 0.25, 2: 0.35, 3: 0.25}
        return ProbabilityMeasure(sig_alg=F).from_dict(probabilities=probabilities)

    def test_from_event_basic(self, F, P):
        """Test creating conditional probability space from basic event."""
        A = F.get_event([1, 2], name="A")
        prob_space = ProbabilitySpace.from_event(event=A, prob_measure=P)

        assert prob_space.sample_space.name == "A"
        assert set(prob_space.sample_space) == {1, 2}
        assert prob_space.sig_alg.name == "F_A"
        assert prob_space.prob_measure.name == "P_A"

    def test_from_event_probabilities_sum_to_one(self, F, P):
        """Test that conditional probabilities sum to 1."""
        A = F.get_event([1, 2], name="A")
        prob_space = ProbabilitySpace.from_event(event=A, prob_measure=P)
        total_prob = sum(prob_space.prob_measure.probabilities.values())

        assert abs(total_prob - 1.0) < 1e-10

    def test_from_event_conditional_probabilities_correct(self, F, P):
        """Test that conditional probabilities are correctly calculated."""
        A = F.get_event([1, 2], name="A")
        prob_space = ProbabilitySpace.from_event(event=A, prob_measure=P)
        prob_A = P(A)
        expected_prob_atom = P([1, 2]) / prob_A

        assert abs(prob_space.prob_measure([1, 2]) - expected_prob_atom) < 1e-10

    def test_from_event_sigma_algebra_structure_preserved(self, F, P):
        """Test that sigma-algebra structure is preserved in conditional space."""
        A = F.get_event([1, 2], name="A")
        prob_space = ProbabilitySpace.from_event(event=A, prob_measure=P)
        atom_ids_conditional = prob_space.sig_alg.sample_id_to_atom_id

        assert atom_ids_conditional[1] == 1
        assert atom_ids_conditional[2] == 1
        assert atom_ids_conditional[1] == atom_ids_conditional[2]

    def test_from_event_full_sample_space(self, Omega, F, P):
        """Test creating conditional space from full sample space."""
        full = F.get_event([0, 1, 2, 3], name="Omega")
        prob_space = ProbabilitySpace.from_event(event=full, prob_measure=P)

        assert set(prob_space.sample_space) == set(Omega)
        assert abs(prob_space.prob_measure(0) - 0.15) < 1e-10

    def test_from_event_invalid_event_type_raises(self, P):
        """Test that from_event with non-Event raises TypeError."""
        with pytest.raises(TypeError, match="event must be an Event instance"):
            ProbabilitySpace.from_event(event="not an event", prob_measure=P)

    def test_from_event_invalid_prob_measure_type_raises(self, F):
        """Test that from_event with non-ProbabilityMeasure raises TypeError."""
        A = F.get_event([1, 2])

        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure instance"
        ):
            ProbabilitySpace.from_event(event=A, prob_measure="not a prob measure")

    def test_from_event_event_not_in_domain_raises(self, Omega, P):
        """Test that from_event with event not in domain raises ValueError."""
        F_other = SigmaAlgebra.power_set(Omega)
        A = F_other.get_event([0, 1])

        with pytest.raises(
            ValueError, match="event must be in the domain.*of the given"
        ):
            ProbabilitySpace.from_event(event=A, prob_measure=P)

    def test_from_event_zero_probability_raises(self, Omega, F):
        """Test that from_event with zero probability event raises ValueError."""
        P_zero = ProbabilityMeasure(sig_alg=F).from_dict(
            {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
        )
        A = F.get_event([1, 2])

        with pytest.raises(
            ValueError, match="Cannot create a probability space from.*0 probability"
        ):
            ProbabilitySpace.from_event(event=A, prob_measure=P_zero)

    def test_from_event_power_set_sigma_algebra(self):
        """Test from_event with power-set sigma-algebra."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra.power_set(Omega)
        P = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        A = F.get_event([0, 1], name="A")
        prob_space = ProbabilitySpace.from_event(event=A, prob_measure=P)
        expected_prob_0 = 0.2 / 0.7
        expected_prob_1 = 0.5 / 0.7

        assert abs(prob_space.prob_measure(0) - expected_prob_0) < 1e-10
        assert abs(prob_space.prob_measure(1) - expected_prob_1) < 1e-10


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


class TestSample:
    def test_sample_returns_correct_size(self):
        """Test that sample method returns correct number of samples."""
        Omega = SampleSpace().from_sequence(size=3)
        prob_space = ProbabilitySpace(Omega)
        samples = prob_space.sample(size=10, random_state=42)

        assert len(samples) == 10

    def test_sample_returns_valid_outcomes(self):
        """Test that sampled outcomes are in the sample space."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra.power_set(Omega)
        P = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4})
        prob_space = ProbabilitySpace(Omega, prob_measure=P)
        samples = prob_space.sample(size=20, random_state=42)

        assert all(s in Omega for s in samples)

    def test_sample_respects_probabilities(self):
        """Test that sampling respects probability distribution."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        P = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.0, 1: 1.0})
        prob_space = ProbabilitySpace(Omega, prob_measure=P)
        samples = prob_space.sample(size=100, random_state=42)

        assert all(s == 1 for s in samples)

    def test_sample_invalid_size_raises(self):
        """Test that invalid size raises ValueError."""
        Omega = SampleSpace().from_sequence(size=3)
        prob_space = ProbabilitySpace(Omega)

        with pytest.raises(ValueError, match="size must be a positive integer"):
            prob_space.sample(size=0)

    def test_sample_non_power_set_raises(self):
        """Test that sampling on non-power-set sigma-algebra raises ValueError."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1, 3: 1})
        P = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.3, 1: 0.3, 2: 0.2, 3: 0.2})
        prob_space = ProbabilitySpace(Omega, sig_alg=F, prob_measure=P)

        with pytest.raises(
            ValueError, match="only supported for.*power set sigma-algebras"
        ):
            prob_space.sample(size=5)

    def test_sample_invalid_random_state_raises(self):
        """Test that invalid random_state raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        prob_space = ProbabilitySpace(Omega)

        with pytest.raises(
            TypeError, match="random_state must be an integer.*Generator.*None"
        ):
            prob_space.sample(size=5, random_state="not valid")


class TestIteration:
    def test_iteration_unpacks_correctly(self):
        """Test that iteration unpacks sample space, sigma-algebra, and probability measure."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        P = ProbabilityMeasure(sig_alg=F).from_dict({0: 0.3, 1: 0.5, 2: 0.2})
        prob_space = ProbabilitySpace(Omega, F, P)
        unpacked_Omega, unpacked_F, unpacked_P = prob_space

        assert unpacked_Omega == Omega
        assert unpacked_F == F
        assert unpacked_P == P

    def test_iteration_order(self):
        """Test that iteration yields components in correct order."""
        Omega = SampleSpace().from_sequence(size=2)
        prob_space = ProbabilitySpace(Omega)
        components = list(prob_space)

        assert len(components) == 3
        assert components[0] == prob_space.sample_space
        assert components[1] == prob_space.sig_alg
        assert components[2] == prob_space.prob_measure


class TestValidation:
    def test_constructor_mismatched_prob_measure_sig_alg_raises(self):
        """Test that constructor raises when prob_measure.sig_alg does not match sig_alg."""
        Omega = SampleSpace().from_sequence(size=3)
        F1 = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        F2 = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 1, 2: 1})
        P = ProbabilityMeasure(sig_alg=F2).from_dict({0: 0.3, 1: 0.5, 2: 0.2})

        with pytest.raises(
            ValueError,
            match="probability measure must be defined on the given sigma-algebra",
        ):
            ProbabilitySpace(Omega, sig_alg=F1, prob_measure=P)

    def test_constructor_none_sample_space_with_sig_alg_raises(self):
        """Test that constructor raises when sample_space is None but sig_alg is provided."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)

        with pytest.raises(
            ValueError, match="If sample_space is not given, sig_alg must also be None"
        ):
            ProbabilitySpace(sample_space=None, sig_alg=F)

    def test_constructor_none_sample_space_with_prob_measure_raises(self):
        """Test that constructor raises when sample_space is None but prob_measure is provided."""
        Omega = SampleSpace().from_sequence(size=2)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.5, 1: 0.5}
        )

        with pytest.raises(
            ValueError,
            match="If sample_space is not given, prob_measure must also be None",
        ):
            ProbabilitySpace(sample_space=None, prob_measure=P)
