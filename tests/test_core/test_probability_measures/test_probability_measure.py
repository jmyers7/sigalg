import numpy as np
import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_construction_with_valid_probabilities(self, sample_space):
        probabilities = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space, name="Q"
        )
        expected_series = pd.Series(data=probabilities, name="Q")
        expected_series.index.name = "Omega"
        pd.testing.assert_series_equal(prob_measure.values, expected_series)

    def test_uniform_creates_equal_probabilities(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        prob_measure = sa.ProbabilityMeasure.uniform(space)
        assert np.allclose(prob_measure.values.to_numpy(), 1 / 3)

    def test_construction_without_sample_space(self):
        probabilities = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(probabilities=probabilities)
        assert isinstance(prob_measure.sample_space, sa.SampleSpace)


class TestValidation:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.ProbabilityMeasure("not a space", {"omega0": 1.0})

    def test_construction_with_non_dict_probabilities(self, sample_space):
        with pytest.raises(TypeError, match="must be a dictionary"):
            sa.ProbabilityMeasure(
                probabilities=[0.5, 0.3, 0.2], sample_space=sample_space
            )

    def test_construction_with_missing_indices(self, sample_space):
        probabilities = {"omega0": 0.5, "omega1": 0.5}
        with pytest.raises(ValueError, match="must match sample space indices"):
            sa.ProbabilityMeasure(
                probabilities=probabilities, sample_space=sample_space
            )

    def test_construction_with_extra_indices(self, sample_space):
        probabilities = {"omega0": 0.3, "omega1": 0.3, "omega2": 0.3, "extra": 0.1}
        with pytest.raises(ValueError, match="must match sample space indices"):
            sa.ProbabilityMeasure(
                probabilities=probabilities, sample_space=sample_space
            )

    def test_construction_with_negative_probability(self, sample_space):
        probabilities = {"omega0": -0.1, "omega1": 0.6, "omega2": 0.5}
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            sa.ProbabilityMeasure(
                probabilities=probabilities, sample_space=sample_space
            )

    def test_construction_with_probability_greater_than_one(self, sample_space):
        probabilities = {"omega0": 1.5, "omega1": 0.0, "omega2": -0.5}
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            sa.ProbabilityMeasure(
                probabilities=probabilities, sample_space=sample_space
            )

    def test_construction_with_probabilities_not_summing_to_one(self, sample_space):
        probabilities = {"omega0": 0.3, "omega1": 0.3, "omega2": 0.3}
        with pytest.raises(ValueError, match="must sum to 1"):
            sa.ProbabilityMeasure(
                probabilities=probabilities, sample_space=sample_space
            )


class TestCallMethod:
    @pytest.fixture
    def prob_measure(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        return sa.ProbabilityMeasure(probabilities=probs, sample_space=space)

    def test_call_with_single_index(self, prob_measure):
        assert prob_measure("omega0") == 0.1
        assert prob_measure("omega1") == 0.2
        assert prob_measure("omega2") == 0.3
        assert prob_measure("omega3") == 0.4

    def test_call_with_invalid_index(self, prob_measure):
        with pytest.raises(KeyError, match="not found in sample space"):
            prob_measure("invalid")

    def test_call_with_list_of_indices(self, prob_measure):
        prob = prob_measure(["omega0", "omega1"])
        assert prob - 0.3 < 1e-10

    def test_call_with_empty_list(self, prob_measure):
        prob = prob_measure([])
        assert prob == 0.0

    def test_call_with_all_indices(self, prob_measure):
        all_indices = ["omega0", "omega1", "omega2", "omega3"]
        prob = prob_measure(all_indices)
        assert abs(prob - 1.0) < 1e-10

    def test_call_with_list_containing_invalid_index(self, prob_measure):
        with pytest.raises(KeyError, match="not found in sample space"):
            prob_measure(["omega0", "invalid"])

    def test_call_with_event(self, prob_measure):
        event = sa.Event(prob_measure.sample_space, ["omega0", "omega1"])
        prob = prob_measure(event)
        assert abs(prob - 0.3) < 1e-10

    def test_call_with_event_from_different_space(self, prob_measure):
        other_space = sa.SampleSpace(["a", "b"])
        event = sa.Event(other_space, ["a"])
        with pytest.raises(ValueError, match="same sample space"):
            prob_measure(event)

    def test_call_with_empty_event(self, prob_measure):
        event = sa.Event(prob_measure.sample_space, [])
        prob = prob_measure(event)
        assert prob == 0.0

    def test_call_with_full_space_event(self, prob_measure):
        event = sa.Event(
            prob_measure.sample_space, list(prob_measure.sample_space.values)
        )
        prob = prob_measure(event)
        assert abs(prob - 1.0) < 1e-10


class TestEquality:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_equality_same_probabilities(self, sample_space):
        probabilities = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure1 = sa.ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        prob_measure2 = sa.ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        assert prob_measure1 == prob_measure2

    def test_equality_different_probabilities(self, sample_space):
        probabilities1 = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        probabilities2 = {"omega0": 0.6, "omega1": 0.2, "omega2": 0.2}
        measure1 = sa.ProbabilityMeasure(
            probabilities=probabilities1, sample_space=sample_space
        )
        measure2 = sa.ProbabilityMeasure(
            probabilities=probabilities2, sample_space=sample_space
        )
        assert measure1 != measure2

    def test_equality_different_sample_spaces(self):
        sample_space1 = sa.SampleSpace(["omega0", "omega1"])
        sample_space2 = sa.SampleSpace(["a", "b"])
        probabilities1 = {"omega0": 0.5, "omega1": 0.5}
        probabilities2 = {"a": 0.5, "b": 0.5}
        prob_measure1 = sa.ProbabilityMeasure(
            probabilities=probabilities1, sample_space=sample_space1
        )
        prob_measure2 = sa.ProbabilityMeasure(
            probabilities=probabilities2, sample_space=sample_space2
        )
        assert prob_measure1 != prob_measure2

    def test_equality_with_non_probability_measure(self, sample_space):
        probabilities = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = sa.ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )

        assert prob_measure != "not a measure"
        assert prob_measure != 123
        assert prob_measure != sample_space
        assert prob_measure != probabilities


class TestIntegrationWithEvents:
    @pytest.fixture
    def prob_measure(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        return sa.ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )

    def test_probability_of_complement(self, prob_measure):
        event = sa.Event(prob_measure.sample_space, ["omega0", "omega1"])
        complement = ~event
        prob_event = prob_measure(event)
        prob_complement = prob_measure(complement)
        assert abs(prob_event + prob_complement - 1.0) < 1e-10

    def test_probability_of_union(self, prob_measure):
        event_A = sa.Event(prob_measure.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(prob_measure.sample_space, ["omega1", "omega2"])
        union = event_A | event_B
        prob_union = prob_measure(union)
        assert prob_union - 0.6 < 1e-10

    def test_probability_of_intersection(self, prob_measure):
        event_A = sa.Event(prob_measure.sample_space, ["omega0", "omega1", "omega2"])
        event_B = sa.Event(prob_measure.sample_space, ["omega1", "omega2", "omega3"])
        intersection = event_A & event_B
        prob_intersection = prob_measure(intersection)
        assert prob_intersection == 0.5  # 0.2 + 0.3

    def test_probability_of_disjoint_events(self, prob_measure):
        event_A = sa.Event(prob_measure.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(prob_measure.sample_space, ["omega2", "omega3"])
        intersection = event_A & event_B
        prob_intersection = prob_measure(intersection)
        assert prob_intersection == 0.0

    def test_addition_rule(self, prob_measure):
        event_A = sa.Event(prob_measure.sample_space, ["omega0", "omega1"])
        event_B = sa.Event(prob_measure.sample_space, ["omega1", "omega2"])
        prob_A = prob_measure(event_A)
        prob_B = prob_measure(event_B)
        prob_union = prob_measure(event_A | event_B)
        prob_intersection = prob_measure(event_A & event_B)

        assert abs(prob_union - (prob_A + prob_B - prob_intersection)) < 1e-10
