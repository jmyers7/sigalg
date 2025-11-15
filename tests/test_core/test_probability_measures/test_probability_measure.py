import pytest

import sigalg as sa


class TestConstruction:
    @pytest.fixture
    def space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_construction_valid(self, space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        P = sa.ProbabilityMeasure(space, probs)
        assert P.sample_space == space
        assert len(P.probabilities) == 3

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.ProbabilityMeasure("not a space", {"omega0": 1.0})

    def test_construction_with_non_dict_probabilities(self):
        space = sa.SampleSpace(["omega0"])
        with pytest.raises(TypeError, match="must be a dictionary"):
            sa.ProbabilityMeasure(space, [0.5, 0.5])

    def test_construction_probabilities_not_sum_to_one(self, space):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.1}
        with pytest.raises(ValueError, match="must sum to 1"):
            sa.ProbabilityMeasure(space, probs)

    def test_construction_probability_negative(self, space):
        probs = {"omega0": -0.1, "omega1": 0.6, "omega2": 0.5}
        with pytest.raises(ValueError, match="must be in"):
            sa.ProbabilityMeasure(space, probs)

    def test_construction_probability_greater_than_one(self, space):
        probs = {"omega0": 1.5, "omega1": -0.3, "omega2": -0.2}
        with pytest.raises(ValueError, match="must be in"):
            sa.ProbabilityMeasure(space, probs)

    def test_construction_missing_indices(self, space):
        probs = {"omega0": 0.6, "omega1": 0.4}  # Missing omega2
        with pytest.raises(ValueError, match="must match"):
            sa.ProbabilityMeasure(space, probs)

    def test_construction_extra_indices(self, space):
        probs = {
            "omega0": 0.3,
            "omega1": 0.3,
            "omega2": 0.2,
            "omega3": 0.2,
        }
        with pytest.raises(ValueError, match="must match"):
            sa.ProbabilityMeasure(space, probs)


class TestUniformDistribution:
    def test_uniform_two_points(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        P = sa.ProbabilityMeasure.uniform(space)
        assert P("omega0") == 0.5
        assert P("omega1") == 0.5

    def test_uniform_three_points(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        P = sa.ProbabilityMeasure.uniform(space)
        assert abs(P("omega0") - 1 / 3) < 1e-10
        assert abs(P("omega1") - 1 / 3) < 1e-10
        assert abs(P("omega2") - 1 / 3) < 1e-10

    def test_uniform_single_point(self):
        space = sa.SampleSpace(["omega0"])
        P = sa.ProbabilityMeasure.uniform(space)
        assert P("omega0") == 1.0


class TestProbabilityEvaluation:
    @pytest.fixture
    def space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def P(self, space):
        probs = {"omega0": 0.4, "omega1": 0.3, "omega2": 0.2, "omega3": 0.1}
        return sa.ProbabilityMeasure(space, probs)

    def test_call_with_string_index(self, P):
        assert P("omega0") == 0.4
        assert P("omega1") == 0.3
        assert P("omega2") == 0.2
        assert P("omega3") == 0.1

    def test_getitem_with_string_index(self, P):
        assert P["omega0"] == 0.4
        assert P["omega2"] == 0.2

    def test_call_with_invalid_index(self, P):
        with pytest.raises(KeyError):
            P("invalid_index")

    def test_call_with_event(self, space, P):
        event = space[["omega0", "omega2"]]
        prob = P(event)
        assert prob - 0.6 <= 1e-10

    def test_getitem_with_event(self, space, P):
        event = space[["omega1", "omega3"]]
        prob = P[event]
        assert prob == 0.4

    def test_call_with_empty_event(self, space, P):
        event = space[[]]
        prob = P(event)
        assert prob == 0.0

    def test_call_with_full_space_event(self, space, P):
        event = space[["omega0", "omega1", "omega2", "omega3"]]
        prob = P(event)
        assert abs(prob - 1.0) < 1e-10

    def test_call_with_event_from_different_space(self, P):
        other_space = sa.SampleSpace(["a", "b"])
        event = other_space[["a"]]
        with pytest.raises(ValueError, match="same sample space"):
            P(event)


class TestProperties:
    @pytest.fixture
    def space(self):
        return sa.SampleSpace(["omega0", "omega1"])

    @pytest.fixture
    def P(self, space):
        probs = {"omega0": 0.7, "omega1": 0.3}
        return sa.ProbabilityMeasure(space, probs)

    def test_sample_space_property(self, space, P):
        assert P.sample_space == space

    def test_probabilities_property(self, P):
        probs = P.probabilities
        assert probs["omega0"] == 0.7
        assert probs["omega1"] == 0.3

    def test_probabilities_is_series(self, P):
        import pandas as pd

        assert isinstance(P.probabilities, pd.Series)


class TestEquality:
    def test_equality_same_probabilities(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        probs = {"omega0": 0.6, "omega1": 0.4}
        P1 = sa.ProbabilityMeasure(space, probs)
        P2 = sa.ProbabilityMeasure(space, probs)
        assert P1 == P2

    def test_equality_different_probabilities(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        P1 = sa.ProbabilityMeasure(space, {"omega0": 0.6, "omega1": 0.4})
        P2 = sa.ProbabilityMeasure(space, {"omega0": 0.5, "omega1": 0.5})
        assert P1 != P2

    def test_equality_different_sample_spaces(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        P1 = sa.ProbabilityMeasure(space1, {"omega0": 0.5, "omega1": 0.5})
        P2 = sa.ProbabilityMeasure(space2, {"a": 0.5, "b": 0.5})
        assert P1 != P2

    def test_equality_with_non_probability_measure(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        P = sa.ProbabilityMeasure(space, {"omega0": 0.5, "omega1": 0.5})
        assert P != "not a probability measure"
        assert P != {"omega0": 0.5, "omega1": 0.5}


class TestEdgeCases:
    def test_single_outcome_probability_one(self):
        space = sa.SampleSpace(["omega0"])
        P = sa.ProbabilityMeasure(space, {"omega0": 1.0})
        assert P("omega0") == 1.0

    def test_zero_probability_allowed(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        P = sa.ProbabilityMeasure(space, {"omega0": 0.0, "omega1": 1.0})
        assert P("omega0") == 0.0
        assert P("omega1") == 1.0
