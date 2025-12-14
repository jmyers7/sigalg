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
        return SampleSpace(["omega0", "omega1", "omega2"])

    def test_construction_with_all_parameters(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = ProbabilityMeasure(probs, sample_space)
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        prob_space = ProbabilitySpace(sample_space, sigma_algebra, prob_measure)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == sigma_algebra
        assert prob_space.probability_measure == prob_measure

    def test_construction_with_sample_space_only(self, sample_space):
        prob_space = ProbabilitySpace(sample_space)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra.num_atoms == 3
        assert abs(prob_space.P("omega0") - 1 / 3) < 1e-10

    def test_construction_with_sample_space_and_sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_alg = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        prob_space = ProbabilitySpace(sample_space, sigma_alg)
        assert prob_space.sigma_algebra == sigma_alg
        assert abs(prob_space.P("omega0") - 1 / 3) < 1e-10

    def test_construction_with_sample_space_and_probability_measure(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = ProbabilityMeasure(probs, sample_space)
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        assert prob_space.probability_measure == prob_measure
        assert prob_space.sigma_algebra.num_atoms == 3


class TestValidation:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2"])

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            ProbabilitySpace("not a space")

    def test_construction_with_invalid_sigma_algebra(self, sample_space):
        with pytest.raises(TypeError, match="must be a SigmaAlgebra"):
            ProbabilitySpace(sample_space, sigma_algebra="not a sigma algebra")

    def test_construction_with_mismatched_sigma_algebra(self, sample_space):
        other_space = SampleSpace(["a", "b"])
        atom_ids = {"a": 0, "b": 1}
        sigma_alg = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=other_space
        )
        with pytest.raises(
            ValueError, match="must be defined on the given sample_space"
        ):
            ProbabilitySpace(sample_space, sigma_algebra=sigma_alg)

    def test_construction_with_invalid_probability_measure(self, sample_space):
        with pytest.raises(TypeError, match="must be a ProbabilityMeasure"):
            ProbabilitySpace(sample_space, probability_measure="not a measure")

    def test_construction_with_mismatched_probability_measure(self, sample_space):
        other_space = SampleSpace(["a", "b"])
        probs = {"a": 0.5, "b": 0.5}
        prob_measure = ProbabilityMeasure(probs, other_space)
        with pytest.raises(
            ValueError, match="must be defined on the given sample_space"
        ):
            ProbabilitySpace(sample_space, probability_measure=prob_measure)


class TestSetters:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        return ProbabilitySpace(space)

    def test_set_sigma_algebra_valid(self, prob_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_alg = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=prob_space.sample_space
        )

        prob_space.sigma_algebra = sigma_alg
        assert prob_space.sigma_algebra == sigma_alg

    def test_set_sigma_algebra_invalid_type(self, prob_space):
        with pytest.raises(TypeError, match="must be a SigmaAlgebra"):
            prob_space.sigma_algebra = "not a sigma algebra"

    def test_set_sigma_algebra_wrong_sample_space(self, prob_space):
        other_space = SampleSpace(["a", "b"])
        atom_ids = {"a": 0, "b": 1}
        sigma_alg = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=other_space
        )
        with pytest.raises(ValueError, match="must be defined on"):
            prob_space.sigma_algebra = sigma_alg

    def test_set_probability_measure_valid(self, prob_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = ProbabilityMeasure(
            probabilities=probs, sample_space=prob_space.sample_space
        )

        prob_space.probability_measure = prob_measure
        assert prob_space.probability_measure == prob_measure

    def test_set_probability_measure_invalid_type(self, prob_space):
        with pytest.raises(TypeError, match="must be a ProbabilityMeasure"):
            prob_space.probability_measure = "not a measure"

    def test_set_probability_measure_wrong_sample_space(self, prob_space):
        other_space = SampleSpace(["a", "b"])
        probs = {"a": 0.5, "b": 0.5}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=other_space)

        with pytest.raises(ValueError, match="must be defined on"):
            prob_space.probability_measure = prob_measure


class TestProbabilityMeasureMethod:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        return ProbabilitySpace(space, probability_measure=prob_measure)

    def test_P_with_single_outcome(self, prob_space):
        assert prob_space.P("omega0") == 0.1
        assert prob_space.P("omega1") == 0.2

    def test_P_with_event(self, prob_space):
        event = Event(prob_space.sample_space, ["omega0", "omega1"])
        assert abs(prob_space.P(event) - 0.3) < 1e-10

    def test_P_with_empty_event(self, prob_space):
        event = Event(prob_space.sample_space, [])
        assert prob_space.P(event) == 0.0

    def test_P_with_full_space_event(self, prob_space):
        event = Event(prob_space.sample_space, list(prob_space.sample_space.values))
        assert abs(prob_space.P(event) - 1.0) < 1e-10


class TestEventAccessMethods:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        return ProbabilitySpace(space)

    def test_get_event_returns_event(self, prob_space):
        event = prob_space.get_event(["omega0", "omega1"])
        assert isinstance(event, Event)
        assert list(event.values) == ["omega0", "omega1"]

    def test_get_event_with_empty_list(self, prob_space):
        event = prob_space.get_event([])
        assert isinstance(event, Event)
        assert len(event) == 0

    def test_get_event_with_non_list(self, prob_space):
        with pytest.raises(TypeError, match="must be a list"):
            prob_space.get_event("omega0")


class TestConditionalProbabilitySpace:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_alg = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        return ProbabilitySpace(space, sigma_alg, prob_measure)

    def test_get_event_as_probability_space_creates_new_space(self, prob_space):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        assert isinstance(conditional_space, ProbabilitySpace)

    def test_get_event_as_probability_space_has_correct_sample_space(self, prob_space):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        assert len(conditional_space.sample_space) == 2
        assert set(conditional_space.sample_space.values) == {"omega0", "omega1"}

    def test_get_event_as_probability_space_conditional_probabilities(self, prob_space):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        assert abs(conditional_space.P("omega0") - 0.1 / 0.3) < 1e-10
        assert abs(conditional_space.P("omega1") - 0.2 / 0.3) < 1e-10

    def test_get_event_as_probability_space_probabilities_sum_to_one(self, prob_space):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        total = sum(
            conditional_space.P(idx) for idx in conditional_space.sample_space.values
        )
        assert abs(total - 1.0) < 1e-10

    def test_get_event_as_probability_space_preserves_sigma_algebra_structure(
        self, prob_space
    ):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        assert (
            conditional_space.sigma_algebra.sample_id_to_atom_id["omega0"]
            == conditional_space.sigma_algebra.sample_id_to_atom_id["omega1"]
        )

    def test_get_event_as_probability_space_with_zero_probability_event(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.0, "omega1": 0.5, "omega2": 0.5}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        prob_space_zero = ProbabilitySpace(space, probability_measure=prob_measure)
        with pytest.raises(ValueError, match="zero probability"):
            prob_space_zero.get_event_as_probability_space(["omega0"])


class TestConditionalProbability:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        return ProbabilitySpace(space, probability_measure=prob_measure)

    def test_conditional_probability_basic(self, prob_space):
        event_A = Event(prob_space.sample_space, ["omega0"])
        event_B = Event(prob_space.sample_space, ["omega0", "omega1"])
        cond_prob = prob_space.conditional_probability(event_A, event_B)
        assert abs(cond_prob - 0.1 / 0.3) < 1e-10

    def test_conditional_probability_disjoint_events(self, prob_space):
        event_A = Event(prob_space.sample_space, ["omega0", "omega1"])
        event_B = Event(prob_space.sample_space, ["omega2", "omega3"])
        cond_prob = prob_space.conditional_probability(event_A, event_B)
        assert cond_prob == 0.0

    def test_conditional_probability_A_subset_of_B(self, prob_space):
        event_A = Event(prob_space.sample_space, ["omega0"])
        event_B = Event(prob_space.sample_space, ["omega0", "omega1", "omega2"])
        cond_prob = prob_space.conditional_probability(event_A, event_B)
        assert abs(cond_prob - 0.1 / 0.6) < 1e-10

    def test_conditional_probability_with_zero_probability_conditioning(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.0, "omega1": 0.5, "omega2": 0.5}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        prob_space_zero = ProbabilitySpace(space, probability_measure=prob_measure)
        event_A = Event(space, ["omega1"])
        event_B = Event(space, ["omega0"])
        with pytest.raises(ValueError, match="P\\(B\\) = 0"):
            prob_space_zero.conditional_probability(event_A, event_B)

    def test_conditional_probability_wrong_sample_space_A(self, prob_space):
        other_space = SampleSpace(["a", "b"])
        event_A = Event(other_space, ["a"])
        event_B = Event(prob_space.sample_space, ["omega0"])
        with pytest.raises(
            ValueError, match="event_A must be from this probability space"
        ):
            prob_space.conditional_probability(event_A, event_B)

    def test_conditional_probability_wrong_sample_space_B(self, prob_space):
        other_space = SampleSpace(["a", "b"])
        event_A = Event(prob_space.sample_space, ["omega0"])
        event_B = Event(other_space, ["a"])
        with pytest.raises(
            ValueError, match="event_B must be from this probability space"
        ):
            prob_space.conditional_probability(event_A, event_B)


class TestIndependence:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.25, "omega1": 0.25, "omega2": 0.25, "omega3": 0.25}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        return ProbabilitySpace(space, probability_measure=prob_measure)

    def test_independent_events(self, prob_space):
        event_A = Event(prob_space.sample_space, ["omega0", "omega1"])
        event_B = Event(prob_space.sample_space, ["omega0", "omega2"])
        assert prob_space.are_independent(event_A, event_B)

    def test_dependent_events(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        prob_space = ProbabilitySpace(space, probability_measure=prob_measure)
        event_A = Event(space, ["omega0"])
        event_B = Event(space, ["omega0", "omega1"])
        assert not prob_space.are_independent(event_A, event_B)

    def test_disjoint_events_not_independent_unless_zero_prob(self, prob_space):
        event_A = Event(prob_space.sample_space, ["omega0", "omega1"])
        event_B = Event(prob_space.sample_space, ["omega2", "omega3"])
        assert not prob_space.are_independent(event_A, event_B)

    def test_independence_with_custom_tolerance(self):
        space = SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.5, "omega1": 0.5}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        prob_space = ProbabilitySpace(space, probability_measure=prob_measure)
        event_A = Event(space, ["omega0"])
        event_B = Event(space, ["omega0"])
        assert not prob_space.are_independent(event_A, event_B, tolerance=1e-10)

    def test_independence_wrong_sample_space_A(self, prob_space):
        other_space = SampleSpace(["a", "b"])
        event_A = Event(other_space, ["a"])
        event_B = Event(prob_space.sample_space, ["omega0"])
        with pytest.raises(
            ValueError, match="event_A must be from this probability space"
        ):
            prob_space.are_independent(event_A, event_B)

    def test_independence_wrong_sample_space_B(self, prob_space):
        other_space = SampleSpace(["a", "b"])
        event_A = Event(prob_space.sample_space, ["omega0"])
        event_B = Event(other_space, ["a"])
        with pytest.raises(
            ValueError, match="event_B must be from this probability space"
        ):
            prob_space.are_independent(event_A, event_B)


class TestSampling:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        return ProbabilitySpace(space, probability_measure=prob_measure)

    def test_sample_returns_list(self, prob_space):
        samples = prob_space.sample(size=10)
        assert isinstance(samples, list)

    def test_sample_returns_correct_size(self, prob_space):
        samples = prob_space.sample(size=10)
        assert len(samples) == 10

    def test_sample_single_outcome(self, prob_space):
        sample = prob_space.sample(size=1)
        assert len(sample) == 1
        assert sample[0] in prob_space.sample_space.values

    def test_sample_all_from_sample_space(self, prob_space):
        samples = prob_space.sample(size=100)
        for s in samples:
            assert s in prob_space.sample_space.values

    def test_sample_with_random_state_reproducible(self, prob_space):
        samples1 = prob_space.sample(size=10, random_state=42)
        samples2 = prob_space.sample(size=10, random_state=42)
        assert samples1 == samples2

    def test_sample_with_different_random_states(self, prob_space):
        samples1 = prob_space.sample(size=10, random_state=42)
        samples2 = prob_space.sample(size=10, random_state=43)
        assert samples1 != samples2

    def test_sample_with_invalid_size(self, prob_space):
        with pytest.raises(ValueError, match="must be a positive integer"):
            prob_space.sample(size=0)
        with pytest.raises(ValueError, match="must be a positive integer"):
            prob_space.sample(size=-1)
        with pytest.raises(ValueError, match="must be a positive integer"):
            prob_space.sample(size=1.5)

    def test_sample_distribution_approximates_probabilities(self, prob_space):
        samples = prob_space.sample(size=10000, random_state=42)
        counts = {idx: samples.count(idx) for idx in prob_space.sample_space.values}
        empirical_probs = {idx: count / 10000 for idx, count in counts.items()}
        assert abs(empirical_probs["omega0"] - 0.5) < 0.05
        assert abs(empirical_probs["omega1"] - 0.3) < 0.05
        assert abs(empirical_probs["omega2"] - 0.2) < 0.05


class TestEquality:
    def test_equality_same_components(self):
        space = SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.5, "omega1": 0.5}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        atom_ids = {"omega0": 0, "omega1": 1}
        sigma_alg = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        prob_space1 = ProbabilitySpace(space, sigma_alg, prob_measure)
        prob_space2 = ProbabilitySpace(space, sigma_alg, prob_measure)
        assert prob_space1 == prob_space2

    def test_equality_different_probability_measures(self):
        space = SampleSpace(["omega0", "omega1"])
        probs1 = {"omega0": 0.5, "omega1": 0.5}
        probs2 = {"omega0": 0.7, "omega1": 0.3}
        prob_measure1 = ProbabilityMeasure(probabilities=probs1, sample_space=space)
        prob_measure2 = ProbabilityMeasure(probabilities=probs2, sample_space=space)
        prob_space1 = ProbabilitySpace(space, probability_measure=prob_measure1)
        prob_space2 = ProbabilitySpace(space, probability_measure=prob_measure2)
        assert prob_space1 != prob_space2

    def test_equality_different_sigma_algebras(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids1 = {"omega0": 0, "omega1": 0, "omega2": 1}
        atom_ids2 = {"omega0": 0, "omega1": 1, "omega2": 1}
        sigma_alg1 = SigmaAlgebra(sample_id_to_atom_id=atom_ids1, sample_space=space)
        sigma_alg2 = SigmaAlgebra(sample_id_to_atom_id=atom_ids2, sample_space=space)
        prob_space1 = ProbabilitySpace(space, sigma_algebra=sigma_alg1)
        prob_space2 = ProbabilitySpace(space, sigma_algebra=sigma_alg2)
        assert prob_space1 != prob_space2

    def test_equality_different_sample_spaces(self):
        space1 = SampleSpace(["omega0", "omega1"])
        space2 = SampleSpace(["a", "b"])
        prob_space1 = ProbabilitySpace(space1)
        prob_space2 = ProbabilitySpace(space2)
        assert prob_space1 != prob_space2

    def test_equality_with_non_probability_space(self):
        space = SampleSpace(["omega0", "omega1"])
        prob_space = ProbabilitySpace(space)
        assert prob_space != "not a probability space"
        assert prob_space != 123
        assert prob_space != space


class TestProbabilityAxioms:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        return ProbabilitySpace(space, probability_measure=prob_measure)

    def test_axiom_non_negativity(self, prob_space):
        for idx in prob_space.sample_space.values:
            assert prob_space.P(idx) >= 0

    def test_axiom_normalization(self, prob_space):
        full_event = Event(
            prob_space.sample_space, list(prob_space.sample_space.values)
        )
        assert abs(prob_space.P(full_event) - 1.0) < 1e-10

    def test_axiom_additivity_disjoint_events(self, prob_space):
        event_A = Event(prob_space.sample_space, ["omega0"])
        event_B = Event(prob_space.sample_space, ["omega1"])
        union = event_A | event_B
        prob_union = prob_space.P(union)
        prob_sum = prob_space.P(event_A) + prob_space.P(event_B)
        assert abs(prob_union - prob_sum) < 1e-10

    def test_complement_rule(self, prob_space):
        event = Event(prob_space.sample_space, ["omega0", "omega1"])
        complement = ~event
        assert abs(prob_space.P(event) + prob_space.P(complement) - 1.0) < 1e-10


class TestSigmaAlgebraMethods:
    @pytest.fixture
    def prob_space(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        return ProbabilitySpace(space, probability_measure=prob_measure)

    @pytest.fixture
    def prob_space_with_custom_sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B", "omega3": "B"}
        sigma_algebra = SigmaAlgebra(sample_space=space, sample_id_to_atom_id=atom_ids)
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(probabilities=probs, sample_space=space)
        return ProbabilitySpace(space, sigma_algebra, prob_measure)

    def test_is_measurable_with_measurable_event(self, prob_space):
        event = Event(prob_space.sample_space, ["omega0", "omega1"])
        assert prob_space.is_measurable(event) is True

    def test_is_measurable_with_single_sample_event(self, prob_space):
        event = Event(prob_space.sample_space, ["omega2"])
        assert prob_space.is_measurable(event) is True

    def test_is_measurable_with_full_event(self, prob_space):
        event = Event(prob_space.sample_space, list(prob_space.sample_space.values))
        assert prob_space.is_measurable(event) is True

    def test_is_measurable_custom_sigma_algebra_measurable(
        self, prob_space_with_custom_sigma_algebra
    ):
        event = Event(
            prob_space_with_custom_sigma_algebra.sample_space, ["omega0", "omega1"]
        )
        assert prob_space_with_custom_sigma_algebra.is_measurable(event) is True

    def test_is_measurable_custom_sigma_algebra_not_measurable(
        self, prob_space_with_custom_sigma_algebra
    ):
        event = Event(prob_space_with_custom_sigma_algebra.sample_space, ["omega0"])
        assert prob_space_with_custom_sigma_algebra.is_measurable(event) is False

    def test_is_measurable_custom_sigma_algebra_union_of_atoms(
        self, prob_space_with_custom_sigma_algebra
    ):
        event = Event(
            prob_space_with_custom_sigma_algebra.sample_space,
            list(prob_space_with_custom_sigma_algebra.sample_space.values),
        )
        assert prob_space_with_custom_sigma_algebra.is_measurable(event) is True

    def test_is_measurable_custom_sigma_algebra_partial_atoms(
        self, prob_space_with_custom_sigma_algebra
    ):
        event = Event(
            prob_space_with_custom_sigma_algebra.sample_space, ["omega0", "omega2"]
        )
        assert prob_space_with_custom_sigma_algebra.is_measurable(event) is False

    def test_is_measurable_invalid_event_type(self, prob_space):
        with pytest.raises(TypeError, match="event must be an Event instance"):
            prob_space.is_measurable(["omega0", "omega1"])

    def test_is_measurable_wrong_sample_space(self, prob_space):
        other_space = SampleSpace(["a", "b", "c"])
        event = Event(other_space, ["a", "b"])
        with pytest.raises(ValueError, match="same sample_space"):
            prob_space.is_measurable(event)

    def test_get_atom_containing(self, prob_space):
        atom = prob_space.get_atom_containing("omega0")
        assert isinstance(atom, Event)
        assert list(atom.values) == ["omega0"]

    def test_get_atom_containing_all_samples(self, prob_space):
        for sample_id in ["omega0", "omega1", "omega2", "omega3"]:
            atom = prob_space.get_atom_containing(sample_id)
            assert isinstance(atom, Event)
            assert list(atom.values) == [sample_id]

    def test_get_atom_containing_custom_sigma_algebra(
        self, prob_space_with_custom_sigma_algebra
    ):
        atom = prob_space_with_custom_sigma_algebra.get_atom_containing("omega0")
        assert isinstance(atom, Event)
        assert set(atom.values) == {"omega0", "omega1"}

        atom = prob_space_with_custom_sigma_algebra.get_atom_containing("omega2")
        assert isinstance(atom, Event)
        assert set(atom.values) == {"omega2", "omega3"}

    def test_get_atom_containing_invalid_sample_id(self, prob_space):
        with pytest.raises(ValueError, match="not in sample space"):
            prob_space.get_atom_containing("invalid")

    def test_get_atom_containing_numeric_sample_id(self):
        space = SampleSpace([0, 1, 2, 3])
        prob_space = ProbabilitySpace(space)
        atom = prob_space.get_atom_containing(1)
        assert isinstance(atom, Event)
        assert list(atom.values) == [1]
