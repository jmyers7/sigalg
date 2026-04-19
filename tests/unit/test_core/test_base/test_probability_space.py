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
    def sample_space(self):
        return SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )

    def test_constructor_all_parameters(self, sample_space):
        """Test constructing ProbabilitySpace with all parameters."""
        probabilities = {"omega_0": 0.5, "omega_1": 0.3, "omega_2": 0.2}
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        prob_space = ProbabilitySpace(sample_space, sigma_algebra, prob_measure)
        expected_sigma_algebra = sigma_algebra
        expected_prob_measure = prob_measure

        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == expected_sigma_algebra
        assert prob_space.probability_measure == expected_prob_measure

    def test_constructor_defaults_only(self, sample_space):
        """Test constructing ProbabilitySpace with defaults only."""
        prob_space = ProbabilitySpace(sample_space)
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)
        expected_prob_measure = ProbabilityMeasure.uniform(sample_space)

        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == expected_sigma_algebra
        assert prob_space.probability_measure == expected_prob_measure

    def test_constructor_custom_probabilities_only(self, sample_space):
        """Test constructing ProbabilitySpace with custom probabilities only."""
        probabilities = {"omega_0": 1.0 / 3, "omega_1": 1.0 / 3, "omega_2": 1.0 / 3}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)
        expected_prob_measure = prob_measure

        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == expected_sigma_algebra
        assert prob_space.probability_measure == expected_prob_measure

    def test_constructor_custom_sigma_algebra_only(self, sample_space):
        """Test constructing ProbabilitySpace with custom sigma algebra only."""
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        prob_space = ProbabilitySpace(sample_space, sigma_algebra=sigma_algebra)
        expected_sigma_algebra = sigma_algebra
        expected_prob_measure = ProbabilityMeasure.uniform(sample_space)

        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == expected_sigma_algebra
        assert prob_space.probability_measure == expected_prob_measure

    def test_invalid_both_invalid_types_raises(self, sample_space):
        """Test that invalid types for both parameters raise error."""
        with pytest.raises((TypeError, ValueError)):
            ProbabilitySpace(
                sample_space,
                sigma_algebra="not_a_sigma_algebra",
                probability_measure="not_a_prob_measure",
            )

    def test_invalid_prob_measure_type_raises(self, sample_space):
        """Test that invalid probability measure type raises error."""
        sigma_algebra = SigmaAlgebra(
            sample_space=SampleSpace.generate_sequence(
                size=2,
                initial_index=0,
                prefix="omega",
                name="Omega",
                data_name="sample",
            )
        ).from_dict({"omega_0": 0, "omega_1": 0})
        with pytest.raises((TypeError, ValueError)):
            ProbabilitySpace(
                sample_space,
                sigma_algebra=sigma_algebra,
                probability_measure="not_a_prob_measure",
            )

    def test_invalid_sigma_algebra_type_raises(self, sample_space):
        """Test that invalid sigma algebra type raises error."""
        prob_measure = ProbabilityMeasure(
            sample_space=SampleSpace.generate_sequence(
                size=2,
                initial_index=0,
                prefix="omega",
                name="Omega",
                data_name="sample",
            )
        ).from_dict(
            probabilities={"omega_0": 0.5, "omega_1": 0.5},
        )
        with pytest.raises((TypeError, ValueError)):
            ProbabilitySpace(
                sample_space,
                sigma_algebra="not_a_sigma_algebra",
                probability_measure=prob_measure,
            )

    def test_invalid_mismatched_prob_measure_sample_space_raises(self, sample_space):
        """Test that mismatched probability measure sample space raises error."""
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            {"omega_0": 0, "omega_1": 0, "omega_2": 1},
        )
        prob_measure = ProbabilityMeasure(
            sample_space=SampleSpace.generate_sequence(
                size=2,
                initial_index=0,
                prefix="omega",
                name="Omega",
                data_name="sample",
            )
        ).from_dict(probabilities={"omega_0": 0.5, "omega_1": 0.5})
        with pytest.raises((TypeError, ValueError)):
            ProbabilitySpace(
                sample_space,
                sigma_algebra=sigma_algebra,
                probability_measure=prob_measure,
            )

    def test_invalid_mismatched_sigma_algebra_sample_space_raises(self, sample_space):
        """Test that mismatched sigma algebra sample space raises error."""
        sigma_algebra = SigmaAlgebra(
            sample_space=SampleSpace.generate_sequence(
                size=2,
                initial_index=0,
                prefix="omega",
                name="Omega",
                data_name="sample",
            )
        ).from_dict(sample_id_to_atom_id={"omega_0": 0, "omega_1": 0})
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities={"omega_0": 0.5, "omega_1": 0.5, "omega_2": 0.0}
        )
        with pytest.raises((TypeError, ValueError)):
            ProbabilitySpace(
                sample_space,
                sigma_algebra=sigma_algebra,
                probability_measure=prob_measure,
            )


class TestSetters:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )

    @pytest.fixture
    def prob_space(self, sample_space):
        return ProbabilitySpace(sample_space)

    def test_set_sigma_algebra_updates_sigma_algebra(self, sample_space, prob_space):
        """Test setting a new sigma_algebra updates the ProbabilitySpace's sigma_algebra."""
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1}
        new_sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        prob_space.sigma_algebra = new_sigma_algebra

        assert prob_space.sigma_algebra == new_sigma_algebra

    def test_set_probability_measure_updates_probability_measure(
        self, sample_space, prob_space
    ):
        """Test setting a new probability_measure updates the ProbabilitySpace's probability_measure."""
        probabilities = {"omega_0": 0.4, "omega_1": 0.4, "omega_2": 0.2}
        new_prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space.probability_measure = new_prob_measure

        assert prob_space.probability_measure == new_prob_measure


def test_get_event():
    """Test that get_event returns an Event instance with correct indices."""
    sample_space = SampleSpace.generate_sequence(
        size=3, initial_index=0, prefix="omega", name="Omega", data_name="sample"
    )
    prob_space = ProbabilitySpace(sample_space)
    event = prob_space.get_event(["omega_0", "omega_2"])

    assert isinstance(event, Event)
    assert list(event.data) == ["omega_0", "omega_2"]


class TestPMethod:
    def test_P_method_single_omega0(self):
        """Test P method with single outcome omega0."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {
            "omega_0": 0.1,
            "omega_1": 0.2,
            "omega_2": 0.3,
            "omega_3": 0.4,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities,
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        result = prob_space.P("omega_0")
        expected_probability = 0.1

        assert abs(result - expected_probability) < 1e-10

    def test_P_method_single_omega1(self):
        """Test P method with single outcome omega1."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {
            "omega_0": 0.1,
            "omega_1": 0.2,
            "omega_2": 0.3,
            "omega_3": 0.4,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities,
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        result = prob_space.P("omega_1")
        expected_probability = 0.2

        assert abs(result - expected_probability) < 1e-10

    def test_P_method_single_omega3(self):
        """Test P method with single outcome omega3."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {
            "omega_0": 0.1,
            "omega_1": 0.2,
            "omega_2": 0.3,
            "omega_3": 0.4,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities,
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        result = prob_space.P("omega_3")
        expected_probability = 0.4

        assert abs(result - expected_probability) < 1e-10

    def test_P_method_event_two_outcomes(self):
        """Test P method with event containing two outcomes."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {
            "omega_0": 0.1,
            "omega_1": 0.2,
            "omega_2": 0.3,
            "omega_3": 0.4,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities,
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        event_input = Event(sample_space=sample_space).from_list(
            indices=["omega_0", "omega_1"]
        )
        result = prob_space.P(event_input)
        expected_probability = 0.3

        assert abs(result - expected_probability) < 1e-10

    def test_P_method_empty_event(self):
        """Test P method with empty event."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {
            "omega_0": 0.1,
            "omega_1": 0.2,
            "omega_2": 0.3,
            "omega_3": 0.4,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities,
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        event_input = Event(sample_space=sample_space).from_list(indices=[])
        result = prob_space.P(event_input)
        expected_probability = 0.0

        assert abs(result - expected_probability) < 1e-10

    def test_P_method_full_space_event(self):
        """Test P method with full sample space event."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {
            "omega_0": 0.1,
            "omega_1": 0.2,
            "omega_2": 0.3,
            "omega_3": 0.4,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities,
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        event_input = Event(sample_space=sample_space).from_list(
            indices=["omega_0", "omega_1", "omega_2", "omega_3"]
        )
        result = prob_space.P(event_input)
        expected_probability = 1.0

        assert abs(result - expected_probability) < 1e-10


class TestConditionalProbability:
    def test_conditional_probability_basic(self):
        """Test conditional probability with basic intersection."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {"omega_0": 0.1, "omega_1": 0.2, "omega_2": 0.3, "omega_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        A = prob_space.get_event(["omega_0"], name="A")
        B = prob_space.get_event(["omega_0", "omega_1"], name="B")
        cond_prob = prob_space.conditional_probability(A, B)
        expected_prob = prob_space.P(A & B) / prob_space.P(B)

        assert abs(cond_prob - expected_prob) < 1e-10

    def test_conditional_probability_disjoint(self):
        """Test conditional probability with disjoint events."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {"omega_0": 0.1, "omega_1": 0.2, "omega_2": 0.3, "omega_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        A = prob_space.get_event(["omega_0", "omega_1"], name="A")
        B = prob_space.get_event(["omega_2", "omega_3"], name="B")
        cond_prob = prob_space.conditional_probability(A, B)
        expected_prob = prob_space.P(A & B) / prob_space.P(B)

        assert abs(cond_prob - expected_prob) < 1e-10

    def test_conditional_probability_A_subset_of_B(self):
        """Test conditional probability when A is subset of B."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {"omega_0": 0.1, "omega_1": 0.2, "omega_2": 0.3, "omega_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        A = prob_space.get_event(["omega_0"], name="A")
        B = prob_space.get_event(["omega_0", "omega_1", "omega_2"], name="B")
        cond_prob = prob_space.conditional_probability(A, B)
        expected_prob = prob_space.P(A & B) / prob_space.P(B)

        assert abs(cond_prob - expected_prob) < 1e-10


class TestAreIndependent:
    @pytest.fixture
    def prob_space(self):
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probabilities = {
            "omega_0": 0.25**2,
            "omega_1": 0.75 * 0.25,
            "omega_2": 0.25 * 0.75,
            "omega_3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        return ProbabilitySpace(sample_space, probability_measure=prob_measure)

    def test_independent_events(self, prob_space):
        """Test that two independent events are correctly identified."""
        A = prob_space.get_event(["omega_0", "omega_2"], name="A")
        B = prob_space.get_event(["omega_0", "omega_1"], name="B")
        assert prob_space.are_independent(A, B)

    def test_dependent_events(self, prob_space):
        """Test that two dependent events are correctly identified."""
        A = prob_space.get_event(["omega_0", "omega_2"], name="A")
        B = prob_space.get_event(["omega_1", "omega_3"], name="B")
        assert not prob_space.are_independent(A, B)


class TestEquality:
    def test_non_equality_different_probability_measures(self):
        """Test inequality when probability measures are different."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        given = ProbabilitySpace(
            sample_space,
            probability_measure=ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities={"omega_0": 0.5, "omega_1": 0.5},
            ),
        )
        other = ProbabilitySpace(
            sample_space,
            probability_measure=ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities={"omega_0": 0.7, "omega_1": 0.3}
            ),
        )
        assert given != other

    def test_non_equality_different_sigma_algebras(self):
        """Test inequality when sigma algebras are different."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        given = ProbabilitySpace(
            sample_space,
            sigma_algebra=SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1},
            ),
        )
        other = ProbabilitySpace(
            sample_space,
            sigma_algebra=SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id={"omega_0": 0, "omega_1": 1, "omega_2": 1},
            ),
        )
        assert given != other

    def test_non_equality_different_sample_spaces(self):
        """Test inequality when sample spaces are different."""
        given = ProbabilitySpace(
            SampleSpace.generate_sequence(
                size=2,
                initial_index=0,
                prefix="omega",
                name="Omega",
                data_name="sample",
            )
        )
        other = ProbabilitySpace(
            SampleSpace(name="Omega", data_name="sample").from_list(["a", "b"])
        )
        assert given != other

    def test_non_equality_wrong_type_string(self):
        """Test inequality when comparing to string."""
        given = ProbabilitySpace(
            SampleSpace.generate_sequence(
                size=2,
                initial_index=0,
                prefix="omega",
                name="Omega",
                data_name="sample",
            )
        )
        other = "not a probability space"
        assert given != other

    def test_non_equality_wrong_type_int(self):
        """Test inequality when comparing to integer."""
        given = ProbabilitySpace(
            SampleSpace.generate_sequence(
                size=2,
                initial_index=0,
                prefix="omega",
                name="Omega",
                data_name="sample",
            )
        )
        other = 123
        assert given != other

    def test_equality_same_components(self):
        """Test equality when all components are the same."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        given = ProbabilitySpace(
            sample_space,
            SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id={"omega_0": 0, "omega_1": 1},
            ),
            ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities={"omega_0": 0.5, "omega_1": 0.5},
            ),
        )
        other = ProbabilitySpace(
            sample_space,
            SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id={"omega_0": 0, "omega_1": 1},
            ),
            ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities={"omega_0": 0.5, "omega_1": 0.5},
            ),
        )
        assert given == other


class TestProbabilityAxioms:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )
        probs = {"omega_0": 0.5, "omega_1": 0.3, "omega_2": 0.2}
        prob_measure = ProbabilityMeasure(sample_space=space).from_dict(
            probabilities=probs
        )
        return ProbabilitySpace(space, probability_measure=prob_measure)

    def test_axiom_non_negativity(self, prob_space):
        """Test that probabilities are non-negative."""
        for idx in prob_space.sample_space.data:
            assert prob_space.P(idx) >= 0

    def test_axiom_normalization(self, prob_space):
        """Test that the probability of the entire sample space is 1."""
        full_event = Event(sample_space=prob_space.sample_space).from_list(
            indices=list(prob_space.sample_space.data),
        )
        assert abs(prob_space.P(full_event) - 1.0) < 1e-10

    def test_axiom_additivity_disjoint_events(self, prob_space):
        """Test that the probability of the union of disjoint events equals the sum of their probabilities."""
        event_A = Event(sample_space=prob_space.sample_space).from_list(
            indices=["omega_0"]
        )
        event_B = Event(sample_space=prob_space.sample_space).from_list(
            indices=["omega_1"]
        )
        union = event_A | event_B
        prob_union = prob_space.P(union)
        prob_sum = prob_space.P(event_A) + prob_space.P(event_B)
        assert abs(prob_union - prob_sum) < 1e-10

    def test_complement_rule(self, prob_space):
        """Test that the probability of an event and its complement sum to 1."""
        event = Event(sample_space=prob_space.sample_space).from_list(
            indices=["omega_0", "omega_1"]
        )
        complement = ~event
        assert abs(prob_space.P(event) + prob_space.P(complement) - 1.0) < 1e-10
