import pytest

import sigalg as sa


class TestConstruction:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_construction_with_valid_probabilities(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        measure = sa.ProbabilityMeasure(sample_space, probs)
        assert measure.sample_space == sample_space
        assert measure.probabilities == probs

    def test_construction_with_integer_probabilities(self, sample_space):
        probs = {"omega0": 0, "omega1": 0, "omega2": 1}
        measure = sa.ProbabilityMeasure(sample_space, probs)
        assert measure.probabilities == probs

    def test_construction_with_extreme_probabilities(self, sample_space):
        probs = {"omega0": 0.0, "omega1": 0.0, "omega2": 1.0}
        measure = sa.ProbabilityMeasure(sample_space, probs)
        assert measure("omega2") == 1.0

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.ProbabilityMeasure("not a space", {"omega0": 1.0})

    def test_construction_with_non_dict_probabilities(self, sample_space):
        with pytest.raises(TypeError, match="must be a dictionary"):
            sa.ProbabilityMeasure(sample_space, [0.5, 0.3, 0.2])

    def test_construction_with_missing_indices(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.5}  # Missing omega2
        with pytest.raises(ValueError, match="must match sample space indices"):
            sa.ProbabilityMeasure(sample_space, probs)

    def test_construction_with_extra_indices(self, sample_space):
        probs = {"omega0": 0.3, "omega1": 0.3, "omega2": 0.3, "extra": 0.1}
        with pytest.raises(ValueError, match="must match sample space indices"):
            sa.ProbabilityMeasure(sample_space, probs)

    def test_construction_with_negative_probability(self, sample_space):
        probs = {"omega0": -0.1, "omega1": 0.6, "omega2": 0.5}
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            sa.ProbabilityMeasure(sample_space, probs)

    def test_construction_with_probability_greater_than_one(self, sample_space):
        probs = {"omega0": 1.5, "omega1": 0.0, "omega2": -0.5}
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            sa.ProbabilityMeasure(sample_space, probs)

    def test_construction_with_probabilities_not_summing_to_one(self, sample_space):
        probs = {"omega0": 0.3, "omega1": 0.3, "omega2": 0.3}
        with pytest.raises(ValueError, match="must sum to 1"):
            sa.ProbabilityMeasure(sample_space, probs)

    def test_construction_with_non_real_probability(self, sample_space):
        probs = {"omega0": "0.5", "omega1": 0.3, "omega2": 0.2}
        with pytest.raises(TypeError, match="must be a Real number"):
            sa.ProbabilityMeasure(sample_space, probs)

    def test_construction_with_close_to_one_sum(self, sample_space):
        # Sum is very close to 1.0 due to floating point precision
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.7}
        measure = sa.ProbabilityMeasure(sample_space, probs)
        assert abs(sum(measure.probabilities.values()) - 1.0) < 1e-10


class TestProperties:
    @pytest.fixture
    def measure(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        return sa.ProbabilityMeasure(space, probs)

    def test_sample_space_property(self, measure):
        sample_space = measure.sample_space
        assert isinstance(sample_space, sa.SampleSpace)
        assert len(sample_space) == 3

    def test_probabilities_property_returns_copy(self, measure):
        probs = measure.probabilities
        probs["omega0"] = 999  # Try to modify
        # Original should be unchanged
        assert measure.probabilities["omega0"] == 0.5

    def test_probabilities_property_has_correct_values(self, measure):
        probs = measure.probabilities
        assert probs == {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}


class TestUniformClassMethod:
    def test_uniform_creates_equal_probabilities(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        measure = sa.ProbabilityMeasure.uniform(space)

        for idx in space.values:
            assert abs(measure(idx) - 1 / 3) < 1e-10

    def test_uniform_probabilities_sum_to_one(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        measure = sa.ProbabilityMeasure.uniform(space)

        total = sum(measure(idx) for idx in space.values)
        assert abs(total - 1.0) < 1e-10

    def test_uniform_with_single_element(self):
        space = sa.SampleSpace(["omega0"])
        measure = sa.ProbabilityMeasure.uniform(space)
        assert measure("omega0") == 1.0

    def test_uniform_with_large_space(self):
        indices = [f"omega{i}" for i in range(100)]
        space = sa.SampleSpace(indices)
        measure = sa.ProbabilityMeasure.uniform(space)

        for idx in space.values:
            assert abs(measure(idx) - 0.01) < 1e-10


class TestCallMethod:
    @pytest.fixture
    def measure(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        return sa.ProbabilityMeasure(space, probs)

    def test_call_with_single_index(self, measure):
        assert measure("omega0") == 0.1
        assert measure("omega1") == 0.2
        assert measure("omega2") == 0.3
        assert measure("omega3") == 0.4

    def test_call_with_invalid_index(self, measure):
        with pytest.raises(KeyError, match="not found in sample space"):
            measure("invalid")

    def test_call_with_list_of_indices(self, measure):
        prob = measure(["omega0", "omega1"])
        assert prob - 0.3 < 1e-10

    def test_call_with_empty_list(self, measure):
        prob = measure([])
        assert prob == 0.0

    def test_call_with_all_indices(self, measure):
        all_indices = ["omega0", "omega1", "omega2", "omega3"]
        prob = measure(all_indices)
        assert abs(prob - 1.0) < 1e-10

    def test_call_with_list_containing_invalid_index(self, measure):
        with pytest.raises(KeyError, match="not found in sample space"):
            measure(["omega0", "invalid"])

    def test_call_with_event(self, measure):
        event = sa.Event(measure.sample_space, ["omega0", "omega1"])
        prob = measure(event)
        assert abs(prob - 0.3) < 1e-10

    def test_call_with_event_from_different_space(self, measure):
        other_space = sa.SampleSpace(["a", "b"])
        event = sa.Event(other_space, ["a"])
        with pytest.raises(ValueError, match="same sample space"):
            measure(event)

    def test_call_with_empty_event(self, measure):
        event = sa.Event(measure.sample_space, [])
        prob = measure(event)
        assert prob == 0.0

    def test_call_with_full_space_event(self, measure):
        event = sa.Event(measure.sample_space, list(measure.sample_space.values))
        prob = measure(event)
        assert abs(prob - 1.0) < 1e-10


class TestGetItemMethod:
    @pytest.fixture
    def measure(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        return sa.ProbabilityMeasure(space, probs)

    def test_getitem_with_single_index(self, measure):
        assert measure["omega0"] == 0.5
        assert measure["omega1"] == 0.3

    def test_getitem_with_list(self, measure):
        prob = measure[["omega0", "omega1"]]
        assert prob == 0.8

    def test_getitem_with_event(self, measure):
        event = sa.Event(measure.sample_space, ["omega0", "omega2"])
        prob = measure[event]
        assert prob == 0.7

    def test_getitem_matches_call(self, measure):
        # __getitem__ should behave identically to __call__
        assert measure["omega0"] == measure("omega0")
        assert measure[["omega0", "omega1"]] == measure(["omega0", "omega1"])


class TestToPandas:
    def test_to_pandas_returns_series(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        measure = sa.ProbabilityMeasure(space, probs)

        series = measure.to_pandas()
        import pandas as pd

        assert isinstance(series, pd.Series)

    def test_to_pandas_has_correct_values(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        measure = sa.ProbabilityMeasure(space, probs)

        series = measure.to_pandas()
        assert series["omega0"] == 0.5
        assert series["omega1"] == 0.3
        assert series["omega2"] == 0.2

    def test_to_pandas_has_correct_name(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.6, "omega1": 0.4}
        measure = sa.ProbabilityMeasure(space, probs)

        series = measure.to_pandas()
        assert series.name == "probability"

    def test_to_pandas_index_matches_sample_space(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        measure = sa.ProbabilityMeasure(space, probs)

        series = measure.to_pandas()
        assert list(series.index) == list(space.values)


class TestEquality:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_equality_same_probabilities(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        measure1 = sa.ProbabilityMeasure(sample_space, probs)
        measure2 = sa.ProbabilityMeasure(sample_space, probs)
        assert measure1 == measure2

    def test_equality_different_probabilities(self, sample_space):
        probs1 = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        probs2 = {"omega0": 0.6, "omega1": 0.2, "omega2": 0.2}
        measure1 = sa.ProbabilityMeasure(sample_space, probs1)
        measure2 = sa.ProbabilityMeasure(sample_space, probs2)
        assert measure1 != measure2

    def test_equality_different_sample_spaces(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        probs1 = {"omega0": 0.5, "omega1": 0.5}
        probs2 = {"a": 0.5, "b": 0.5}
        measure1 = sa.ProbabilityMeasure(space1, probs1)
        measure2 = sa.ProbabilityMeasure(space2, probs2)
        assert measure1 != measure2

    def test_equality_with_non_probability_measure(self, sample_space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        measure = sa.ProbabilityMeasure(sample_space, probs)

        assert measure != "not a measure"
        assert measure != 123
        assert measure != sample_space
        assert measure != probs


class TestHashability:
    def test_probability_measure_is_not_hashable(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.5, "omega1": 0.5}
        measure = sa.ProbabilityMeasure(space, probs)

        # Should not be hashable
        with pytest.raises(TypeError):
            hash(measure)


class TestEdgeCases:
    def test_single_outcome_certain_event(self):
        space = sa.SampleSpace(["omega0"])
        probs = {"omega0": 1.0}
        measure = sa.ProbabilityMeasure(space, probs)
        assert measure("omega0") == 1.0

    def test_all_zero_except_one(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.0, "omega1": 0.0, "omega2": 1.0}
        measure = sa.ProbabilityMeasure(space, probs)
        assert measure("omega2") == 1.0
        assert measure(["omega0", "omega1"]) == 0.0

    def test_many_outcomes(self):
        indices = [f"omega{i}" for i in range(100)]
        space = sa.SampleSpace(indices)
        probs = dict.fromkeys(indices, 0.01)
        measure = sa.ProbabilityMeasure(space, probs)

        total = sum(measure(idx) for idx in indices)
        assert abs(total - 1.0) < 1e-10

    def test_floating_point_precision(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        # These might not sum exactly to 1.0 due to floating point
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.7}
        measure = sa.ProbabilityMeasure(space, probs)

        # But our validation allows small tolerance
        assert abs(sum(measure.probabilities.values()) - 1.0) < 1e-10


class TestIntegrationWithEvents:
    @pytest.fixture
    def measure(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        return sa.ProbabilityMeasure(space, probs)

    def test_probability_of_complement(self, measure):
        event = sa.Event(measure.sample_space, ["omega0", "omega1"])
        complement = ~event

        prob_event = measure(event)
        prob_complement = measure(complement)

        assert abs(prob_event + prob_complement - 1.0) < 1e-10

    def test_probability_of_union(self, measure):
        event_A = sa.Event(measure.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(measure.sample_space, ["omega1", "omega2"])
        union = event_A | event_B

        prob_union = measure(union)
        assert prob_union - 0.6 < 1e-10  # 0.1 + 0.2 + 0.3

    def test_probability_of_intersection(self, measure):
        event_A = sa.Event(measure.sample_space, ["omega0", "omega1", "omega2"])
        event_B = sa.Event(measure.sample_space, ["omega1", "omega2", "omega3"])
        intersection = event_A & event_B

        prob_intersection = measure(intersection)
        assert prob_intersection == 0.5  # 0.2 + 0.3

    def test_probability_of_disjoint_events(self, measure):
        event_A = sa.Event(measure.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(measure.sample_space, ["omega2", "omega3"])
        intersection = event_A & event_B

        prob_intersection = measure(intersection)
        assert prob_intersection == 0.0

    def test_addition_rule(self, measure):
        # P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
        event_A = sa.Event(measure.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(measure.sample_space, ["omega1", "omega2"])

        prob_A = measure(event_A)
        prob_B = measure(event_B)
        prob_union = measure(event_A | event_B)
        prob_intersection = measure(event_A & event_B)

        assert abs(prob_union - (prob_A + prob_B - prob_intersection)) < 1e-10


class TestValidation:
    def test_validation_catches_sum_greater_than_one(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.6, "omega1": 0.6}  # Sum = 1.2
        with pytest.raises(ValueError, match="must sum to 1"):
            sa.ProbabilityMeasure(space, probs)

    def test_validation_catches_sum_less_than_one(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.3, "omega1": 0.3}  # Sum = 0.6
        with pytest.raises(ValueError, match="must sum to 1"):
            sa.ProbabilityMeasure(space, probs)

    def test_validation_allows_very_small_probabilities(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        probs = {"omega0": 1e-15, "omega1": 1e-15, "omega2": 1.0 - 2e-15}
        measure = sa.ProbabilityMeasure(space, probs)
        assert measure("omega0") == 1e-15

    def test_validation_exact_zero_allowed(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.0, "omega1": 1.0}
        measure = sa.ProbabilityMeasure(space, probs)
        assert measure("omega0") == 0.0

    def test_validation_exact_one_allowed(self):
        space = sa.SampleSpace(["omega0"])
        probs = {"omega0": 1.0}
        measure = sa.ProbabilityMeasure(space, probs)
        assert measure("omega0") == 1.0
