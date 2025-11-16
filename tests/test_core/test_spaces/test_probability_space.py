import pandas as pd
import pytest

import sigalg as sa


class TestConstruction:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_construction_with_all_parameters(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(sample_space, probs)
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_alg = sa.SigmaAlgebra(sample_space, atom_ids)
        prob_space = sa.ProbabilitySpace(sample_space, sigma_alg, prob_measure)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == sigma_alg
        assert prob_space.probability_measure == prob_measure

    def test_construction_with_sample_space_only(self, sample_space):
        prob_space = sa.ProbabilitySpace(sample_space)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra.num_atoms == 3
        assert abs(prob_space.P("omega0") - 1 / 3) < 1e-10

    def test_construction_with_sample_space_and_sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_alg = sa.SigmaAlgebra(sample_space, atom_ids)
        prob_space = sa.ProbabilitySpace(sample_space, sigma_alg)
        assert prob_space.sigma_algebra == sigma_alg
        assert abs(prob_space.P("omega0") - 1 / 3) < 1e-10

    def test_construction_with_sample_space_and_probability_measure(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(sample_space, probs)
        prob_space = sa.ProbabilitySpace(sample_space, probability_measure=prob_measure)
        assert prob_space.probability_measure == prob_measure
        assert prob_space.sigma_algebra.num_atoms == 3

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.ProbabilitySpace("not a space")

    def test_construction_with_invalid_sigma_algebra(self, sample_space):
        with pytest.raises(TypeError, match="must be a SigmaAlgebra"):
            sa.ProbabilitySpace(sample_space, sigma_algebra="not a sigma algebra")

    def test_construction_with_mismatched_sigma_algebra(self, sample_space):
        other_space = sa.SampleSpace(["a", "b"])
        atom_ids = {"a": 0, "b": 1}
        sigma_alg = sa.SigmaAlgebra(other_space, atom_ids)
        with pytest.raises(
            ValueError, match="must be defined on the given sample_space"
        ):
            sa.ProbabilitySpace(sample_space, sigma_algebra=sigma_alg)

    def test_construction_with_invalid_probability_measure(self, sample_space):
        with pytest.raises(TypeError, match="must be a ProbabilityMeasure"):
            sa.ProbabilitySpace(sample_space, probability_measure="not a measure")

    def test_construction_with_mismatched_probability_measure(self, sample_space):
        other_space = sa.SampleSpace(["a", "b"])
        probs = {"a": 0.5, "b": 0.5}
        prob_measure = sa.ProbabilityMeasure(other_space, probs)
        with pytest.raises(
            ValueError, match="must be defined on the given sample_space"
        ):
            sa.ProbabilitySpace(sample_space, probability_measure=prob_measure)


class TestProperties:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    @pytest.fixture
    def sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        return sa.SigmaAlgebra(sample_space, atom_ids)

    @pytest.fixture
    def prob_measure(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        return sa.ProbabilityMeasure(sample_space, probs)

    @pytest.fixture
    def prob_space(self, sample_space, sigma_algebra, prob_measure):
        return sa.ProbabilitySpace(sample_space, sigma_algebra, prob_measure)

    def test_sample_space_property(self, prob_space, sample_space):
        assert prob_space.sample_space == sample_space

    def test_probability_measure_property(self, prob_space, prob_measure):
        assert prob_space.probability_measure == prob_measure

    def test_sigma_algebra_property(self, prob_space, sigma_algebra):
        assert prob_space.sigma_algebra == sigma_algebra

    def test_index_property(self, prob_space):
        assert isinstance(prob_space.index, pd.Index)
        assert list(prob_space.index) == ["omega0", "omega1", "omega2"]


class TestSetters:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        return sa.ProbabilitySpace(space)

    def test_set_sigma_algebra_valid(self, prob_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_alg = sa.SigmaAlgebra(prob_space.sample_space, atom_ids)

        prob_space.set_sigma_algebra(sigma_alg)
        assert prob_space.sigma_algebra == sigma_alg

    def test_set_sigma_algebra_invalid_type(self, prob_space):
        with pytest.raises(TypeError, match="must be a SigmaAlgebra"):
            prob_space.set_sigma_algebra("not a sigma algebra")

    def test_set_sigma_algebra_wrong_sample_space(self, prob_space):
        other_space = sa.SampleSpace(["a", "b"])
        atom_ids = {"a": 0, "b": 1}
        sigma_alg = sa.SigmaAlgebra(other_space, atom_ids)

        with pytest.raises(ValueError, match="must be defined on this sample space"):
            prob_space.set_sigma_algebra(sigma_alg)

    def test_set_probability_measure_valid(self, prob_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(prob_space.sample_space, probs)

        prob_space.set_probability_measure(prob_measure)
        assert prob_space.probability_measure == prob_measure

    def test_set_probability_measure_invalid_type(self, prob_space):
        with pytest.raises(TypeError, match="must be a ProbabilityMeasure"):
            prob_space.set_probability_measure("not a measure")

    def test_set_probability_measure_wrong_sample_space(self, prob_space):
        other_space = sa.SampleSpace(["a", "b"])
        probs = {"a": 0.5, "b": 0.5}
        prob_measure = sa.ProbabilityMeasure(other_space, probs)

        with pytest.raises(ValueError, match="must be defined on this sample space"):
            prob_space.set_probability_measure(prob_measure)


class TestPMethod:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        return sa.ProbabilitySpace(space, probability_measure=prob_measure)

    def test_P_with_single_outcome(self, prob_space):
        assert prob_space.P("omega0") == 0.1
        assert prob_space.P("omega1") == 0.2

    def test_P_with_event(self, prob_space):
        event = sa.Event(prob_space.sample_space, ["omega0", "omega1"])
        assert abs(prob_space.P(event) - 0.3) <= 1e-10

    def test_P_with_empty_event(self, prob_space):
        event = sa.Event(prob_space.sample_space, [])
        assert prob_space.P(event) == 0.0

    def test_P_with_full_space_event(self, prob_space):
        event = sa.Event(prob_space.sample_space, list(prob_space.sample_space.index))
        assert abs(prob_space.P(event) - 1.0) < 1e-10


class TestGetEvent:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        return sa.ProbabilitySpace(space)

    def test_get_event_returns_event(self, prob_space):
        event = prob_space.get_event(["omega0", "omega1"])
        assert isinstance(event, sa.Event)
        assert list(event.index) == ["omega0", "omega1"]

    def test_get_event_with_empty_list(self, prob_space):
        event = prob_space.get_event([])
        assert isinstance(event, sa.Event)
        assert len(event) == 0

    def test_get_event_with_non_list(self, prob_space):
        with pytest.raises(TypeError, match="must be a list"):
            prob_space.get_event("omega0")


class TestGetEventAsProbabilitySpace:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_alg = sa.SigmaAlgebra(space, atom_ids)
        return sa.ProbabilitySpace(space, sigma_alg, prob_measure)

    def test_get_event_as_probability_space_creates_new_space(self, prob_space):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        assert isinstance(conditional_space, sa.ProbabilitySpace)

    def test_get_event_as_probability_space_has_correct_sample_space(self, prob_space):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        assert len(conditional_space.sample_space) == 2
        assert set(conditional_space.sample_space.index) == {"omega0", "omega1"}

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
            conditional_space.P(idx) for idx in conditional_space.sample_space.index
        )
        assert abs(total - 1.0) < 1e-10

    def test_get_event_as_probability_space_preserves_sigma_algebra_structure(
        self, prob_space
    ):
        conditional_space = prob_space.get_event_as_probability_space(
            ["omega0", "omega1"]
        )
        assert (
            conditional_space.sigma_algebra.atom_ids["omega0"]
            == conditional_space.sigma_algebra.atom_ids["omega1"]
        )

    def test_get_event_as_probability_space_with_zero_probability_event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.0, "omega1": 0.5, "omega2": 0.5}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        prob_space_zero = sa.ProbabilitySpace(space, probability_measure=prob_measure)
        with pytest.raises(ValueError, match="zero probability"):
            prob_space_zero.get_event_as_probability_space(["omega0"])


class TestConditionalProbability:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        return sa.ProbabilitySpace(space, probability_measure=prob_measure)

    def test_conditional_probability_basic(self, prob_space):
        event_A = sa.Event(prob_space.sample_space, ["omega0"])
        event_B = sa.Event(prob_space.sample_space, ["omega0", "omega1"])
        cond_prob = prob_space.conditional_probability(event_A, event_B)
        assert abs(cond_prob - 0.1 / 0.3) < 1e-10

    def test_conditional_probability_disjoint_events(self, prob_space):
        event_A = sa.Event(prob_space.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(prob_space.sample_space, ["omega2", "omega3"])
        cond_prob = prob_space.conditional_probability(event_A, event_B)
        assert cond_prob == 0.0

    def test_conditional_probability_A_subset_of_B(self, prob_space):
        event_A = sa.Event(prob_space.sample_space, ["omega0"])
        event_B = sa.Event(prob_space.sample_space, ["omega0", "omega1", "omega2"])
        cond_prob = prob_space.conditional_probability(event_A, event_B)
        assert abs(cond_prob - 0.1 / 0.6) < 1e-10

    def test_conditional_probability_with_zero_probability_conditioning(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.0, "omega1": 0.5, "omega2": 0.5}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        prob_space_zero = sa.ProbabilitySpace(space, probability_measure=prob_measure)
        event_A = sa.Event(space, ["omega1"])
        event_B = sa.Event(space, ["omega0"])
        with pytest.raises(ValueError, match="P\\(B\\) = 0"):
            prob_space_zero.conditional_probability(event_A, event_B)

    def test_conditional_probability_wrong_sample_space_A(self, prob_space):
        other_space = sa.SampleSpace(["a", "b"])
        event_A = sa.Event(other_space, ["a"])
        event_B = sa.Event(prob_space.sample_space, ["omega0"])
        with pytest.raises(
            ValueError, match="event_A must be from this probability space"
        ):
            prob_space.conditional_probability(event_A, event_B)

    def test_conditional_probability_wrong_sample_space_B(self, prob_space):
        other_space = sa.SampleSpace(["a", "b"])
        event_A = sa.Event(prob_space.sample_space, ["omega0"])
        event_B = sa.Event(other_space, ["a"])
        with pytest.raises(
            ValueError, match="event_B must be from this probability space"
        ):
            prob_space.conditional_probability(event_A, event_B)


class TestAreIndependent:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.25, "omega1": 0.25, "omega2": 0.25, "omega3": 0.25}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        return sa.ProbabilitySpace(space, probability_measure=prob_measure)

    def test_independent_events(self, prob_space):
        event_A = sa.Event(prob_space.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(prob_space.sample_space, ["omega0", "omega2"])
        assert prob_space.are_independent(event_A, event_B)

    def test_dependent_events(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        prob_space_dep = sa.ProbabilitySpace(space, probability_measure=prob_measure)
        event_A = sa.Event(space, ["omega0"])
        event_B = sa.Event(space, ["omega0", "omega1"])
        assert not prob_space_dep.are_independent(event_A, event_B)

    def test_disjoint_events_not_independent_unless_zero_prob(self, prob_space):
        event_A = sa.Event(prob_space.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(prob_space.sample_space, ["omega2", "omega3"])
        assert not prob_space.are_independent(event_A, event_B)

    def test_independence_with_custom_tolerance(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.5, "omega1": 0.5}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        prob_space_simple = sa.ProbabilitySpace(space, probability_measure=prob_measure)
        event_A = sa.Event(space, ["omega0"])
        event_B = sa.Event(space, ["omega0"])
        assert not prob_space_simple.are_independent(event_A, event_B, tolerance=1e-10)

    def test_independence_wrong_sample_space_A(self, prob_space):
        other_space = sa.SampleSpace(["a", "b"])
        event_A = sa.Event(other_space, ["a"])
        event_B = sa.Event(prob_space.sample_space, ["omega0"])
        with pytest.raises(
            ValueError, match="event_A must be from this probability space"
        ):
            prob_space.are_independent(event_A, event_B)

    def test_independence_wrong_sample_space_B(self, prob_space):
        other_space = sa.SampleSpace(["a", "b"])
        event_A = sa.Event(prob_space.sample_space, ["omega0"])
        event_B = sa.Event(other_space, ["a"])
        with pytest.raises(
            ValueError, match="event_B must be from this probability space"
        ):
            prob_space.are_independent(event_A, event_B)


class TestSample:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        return sa.ProbabilitySpace(space, probability_measure=prob_measure)

    def test_sample_returns_list(self, prob_space):
        samples = prob_space.sample(size=10)
        assert isinstance(samples, list)

    def test_sample_returns_correct_size(self, prob_space):
        samples = prob_space.sample(size=10)
        assert len(samples) == 10

    def test_sample_single_outcome(self, prob_space):
        sample = prob_space.sample(size=1)
        assert len(sample) == 1
        assert sample[0] in prob_space.sample_space.index

    def test_sample_all_from_sample_space(self, prob_space):
        samples = prob_space.sample(size=100)
        for s in samples:
            assert s in prob_space.sample_space.index

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
        counts = {idx: samples.count(idx) for idx in prob_space.sample_space.index}
        empirical_probs = {idx: count / 10000 for idx, count in counts.items()}
        assert abs(empirical_probs["omega0"] - 0.5) < 0.05
        assert abs(empirical_probs["omega1"] - 0.3) < 0.05
        assert abs(empirical_probs["omega2"] - 0.2) < 0.05


class TestSequenceMethods:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        return sa.ProbabilitySpace(space)

    def test_len(self, prob_space):
        assert len(prob_space) == 3

    def test_getitem_with_single_index(self, prob_space):
        assert prob_space[0] == "omega0"

    def test_getitem_with_list(self, prob_space):
        event = prob_space[["omega0", "omega1"]]
        assert isinstance(event, sa.Event)
        assert list(event.index) == ["omega0", "omega1"]

    def test_iteration(self, prob_space):
        indices = list(prob_space)
        assert indices == ["omega0", "omega1", "omega2"]

    def test_iteration_multiple_times(self, prob_space):
        list1 = list(prob_space)
        list2 = list(prob_space)
        assert list1 == list2


class TestEquality:
    def test_equality_same_components(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.5, "omega1": 0.5}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        atom_ids = {"omega0": 0, "omega1": 1}
        sigma_alg = sa.SigmaAlgebra(space, atom_ids)
        prob_space1 = sa.ProbabilitySpace(space, sigma_alg, prob_measure)
        prob_space2 = sa.ProbabilitySpace(space, sigma_alg, prob_measure)
        assert prob_space1 == prob_space2

    def test_equality_different_probability_measures(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs1 = {"omega0": 0.5, "omega1": 0.5}
        probs2 = {"omega0": 0.7, "omega1": 0.3}
        prob_measure1 = sa.ProbabilityMeasure(space, probs1)
        prob_measure2 = sa.ProbabilityMeasure(space, probs2)
        prob_space1 = sa.ProbabilitySpace(space, probability_measure=prob_measure1)
        prob_space2 = sa.ProbabilitySpace(space, probability_measure=prob_measure2)
        assert prob_space1 != prob_space2

    def test_equality_different_sigma_algebras(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids1 = {"omega0": 0, "omega1": 0, "omega2": 1}
        atom_ids2 = {"omega0": 0, "omega1": 1, "omega2": 1}
        sigma_alg1 = sa.SigmaAlgebra(space, atom_ids1)
        sigma_alg2 = sa.SigmaAlgebra(space, atom_ids2)
        prob_space1 = sa.ProbabilitySpace(space, sigma_algebra=sigma_alg1)
        prob_space2 = sa.ProbabilitySpace(space, sigma_algebra=sigma_alg2)
        assert prob_space1 != prob_space2

    def test_equality_different_sample_spaces(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        prob_space1 = sa.ProbabilitySpace(space1)
        prob_space2 = sa.ProbabilitySpace(space2)
        assert prob_space1 != prob_space2

    def test_equality_with_non_probability_space(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        prob_space = sa.ProbabilitySpace(space)
        assert prob_space != "not a probability space"
        assert prob_space != 123
        assert prob_space != space


class TestEdgeCases:
    def test_single_outcome_space(self):
        space = sa.SampleSpace(["omega0"])
        prob_space = sa.ProbabilitySpace(space)
        assert len(prob_space) == 1
        assert prob_space.P("omega0") == 1.0

    def test_large_sample_space(self):
        indices = [f"omega{i}" for i in range(100)]
        space = sa.SampleSpace(indices)
        prob_space = sa.ProbabilitySpace(space)
        assert len(prob_space) == 100
        for idx in indices:
            assert abs(prob_space.P(idx) - 0.01) < 1e-10


class TestProbabilityAxioms:
    @pytest.fixture
    def prob_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(space, probs)
        return sa.ProbabilitySpace(space, probability_measure=prob_measure)

    def test_axiom_non_negativity(self, prob_space):
        for idx in prob_space.sample_space.index:
            assert prob_space.P(idx) >= 0

    def test_axiom_normalization(self, prob_space):
        full_event = sa.Event(
            prob_space.sample_space, list(prob_space.sample_space.index)
        )
        assert abs(prob_space.P(full_event) - 1.0) < 1e-10

    def test_axiom_additivity_disjoint_events(self, prob_space):
        event_A = sa.Event(prob_space.sample_space, ["omega0"])
        event_B = sa.Event(prob_space.sample_space, ["omega1"])
        union = event_A | event_B
        prob_union = prob_space.P(union)
        prob_sum = prob_space.P(event_A) + prob_space.P(event_B)
        assert abs(prob_union - prob_sum) < 1e-10

    def test_complement_rule(self, prob_space):
        event = sa.Event(prob_space.sample_space, ["omega0", "omega1"])
        complement = ~event
        assert abs(prob_space.P(event) + prob_space.P(complement) - 1.0) < 1e-10
