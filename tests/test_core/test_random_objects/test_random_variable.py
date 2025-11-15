import pandas as pd
import pytest

import sigalg as sa


class TestConstructionAndBasicProperties:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def probability_space(self, sample_space):
        probabilities = {"s0": 0.2, "s1": 0.5, "s2": 0.3}
        prob_measure = sa.ProbabilityMeasure(sample_space, probabilities)
        return sample_space.add_probability_measure(prob_measure)

    @pytest.fixture
    def domain_features(self, sample_space):
        features = [[1, 2], [3, 4], [5, 6]]
        return sa.SampleSpaceFeatures(features=features, sample_space=sample_space)

    def test_construction_from_sample_space(self, sample_space):
        values = dict(zip(sample_space, [10, 20, 30]))
        Y = sa.RandomVariable(domain=sample_space, values=values, name="Y")
        assert Y.domain == sample_space
        assert Y.name == "Y"
        expected_values = pd.Series(
            data=[10, 20, 30], index=sample_space.index, name="Y"
        )
        pd.testing.assert_series_equal(Y.values, expected_values)

    def test_construction_from_probability_space(self, probability_space):
        values = dict(zip(probability_space.sample_space, [5, 15, 25]))
        Z = sa.RandomVariable(domain=probability_space, values=values, name="Z")
        assert Z.domain == probability_space
        assert Z.name == "Z"
        expected_values = pd.Series(
            data=[5, 15, 25], index=probability_space.sample_space.index, name="Z"
        )
        pd.testing.assert_series_equal(Z.values, expected_values)

    def test_construction_from_features(self, domain_features):
        def function(sample_features):
            return sample_features.feature_at[0] + sample_features.feature_at[1]

        X = sa.RandomVariable.from_features(
            domain_features=domain_features, function=function, name="X"
        )
        assert X.domain == domain_features.sample_space
        assert X.name == "X"
        assert X.function == function
        expected_values = pd.Series(
            data=[3, 7, 11], index=domain_features.sample_space.index, name="X"
        )
        pd.testing.assert_series_equal(X.values, expected_values)


class TestMethods:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def domain_features(self, sample_space):
        features = [[1, 2], [3, 4], [5, 6]]
        return sa.SampleSpaceFeatures(features=features, sample_space=sample_space)

    def test_random_variable_set_name(self, sample_space):
        values = dict(zip(sample_space, [1, 2, 3]))
        rv = sa.RandomVariable(domain=sample_space, values=values, name="A")
        assert rv.name == "A"
        rv.set_name("B")
        assert rv.name == "B"

    def test_call_rv_from_features(self, domain_features):
        def function(sample_features):
            return sample_features.feature_at[0] * 2

        X = sa.RandomVariable.from_features(
            domain_features=domain_features, function=function, name="X"
        )
        sample_features = domain_features["s2"]
        result = X(sample_features)
        assert result == 10
        result = X("s1")
        assert result == 6

    def test_call_rv_from_values(self, sample_space):
        values = dict(zip(sample_space, [7, 8, 9]))
        Y = sa.RandomVariable(domain=sample_space, values=values, name="Y")
        result = Y("s0")
        assert result == 7
        result = Y("s2")
        assert result == 9

    def test_sigma_algebra(self, sample_space):
        values = dict(zip(sample_space, [0, 1, 0]))
        U = sa.RandomVariable(domain=sample_space, values=values, name="U")
        sigma_algebra = U.sigma_algebra
        expected_atom_ids = {"s0": 0, "s1": 1, "s2": 0}
        assert sigma_algebra._sample_space == sample_space
        assert sigma_algebra._atom_ids == expected_atom_ids
        expected_events = {
            0: sa.Event(sample_space=sample_space, event_indices=["s0", "s2"]),
            1: sa.Event(sample_space=sample_space, event_indices=["s1"]),
        }
        actual_events = sigma_algebra.to_events()
        assert actual_events == expected_events


class TestRangeProperty:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def prob_space(self):
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        return sa.ProbabilitySpace(["omega0", "omega1", "omega2", "omega3"], probs)

    def test_range_from_regular_sample_space(self, sample_space):
        X = sa.RandomVariable(
            sample_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 3}
        )
        range_space = X.range
        assert isinstance(range_space, sa.SampleSpace)
        assert not isinstance(range_space, sa.ProbabilitySpace)
        expected_range = sa.SampleSpace([1, 2, 3])
        assert range_space == expected_range

    def test_range_from_probability_space(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 2}
        )
        range_space = X.range
        assert isinstance(range_space, sa.ProbabilitySpace)

    def test_range_probability_space_has_correct_probabilities(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 2}
        )
        range_space = X.range
        assert range_space.P(1) - 0.4 < 1e-10
        assert range_space.P(2) - 0.6 < 1e-10

    def test_range_probabilities_sum_to_one(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 3}
        )
        range_space = X.range
        total_prob = sum(range_space.P(val) for val in range_space.index)
        assert abs(total_prob - 1.0) < 1e-10

    def test_range_with_single_value(self, sample_space):
        X = sa.RandomVariable(
            sample_space, {"omega0": 5, "omega1": 5, "omega2": 5, "omega3": 5}
        )
        range_space = X.range
        assert len(range_space) == 1
        assert 5 in range_space.index

    def test_range_with_all_unique_values(self, sample_space):
        X = sa.RandomVariable(
            sample_space, {"omega0": 1, "omega1": 2, "omega2": 3, "omega3": 4}
        )
        range_space = X.range
        assert len(range_space) == 4
        assert set(range_space.index) == {1, 2, 3, 4}

    def test_range_preserves_value_types(self, sample_space):
        X = sa.RandomVariable(
            sample_space, {"omega0": "a", "omega1": "b", "omega2": "a", "omega3": "c"}
        )
        range_space = X.range
        assert set(range_space.index) == {"a", "b", "c"}

    def test_range_with_float_values(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1.5, "omega1": 2.5, "omega2": 1.5, "omega3": 2.5}
        )
        range_space = X.range
        assert abs(range_space.P(1.5) - 0.4) < 1e-10
        assert abs(range_space.P(2.5) - 0.6) < 1e-10


class TestProbabilityMeasureProperty:
    @pytest.fixture
    def prob_space(self):
        probs = {"omega0": 0.25, "omega1": 0.25, "omega2": 0.25, "omega3": 0.25}
        return sa.ProbabilitySpace(["omega0", "omega1", "omega2", "omega3"], probs)

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_probability_measure_from_probability_space(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 2}
        )
        P = X.probability_measure
        assert isinstance(P, sa.ProbabilityMeasure)

    def test_probability_measure_values_correct(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 2}
        )
        P = X.probability_measure
        assert P(1) == 0.5
        assert P(2) == 0.5

    def test_probability_measure_raises_for_regular_space(self, sample_space):
        X = sa.RandomVariable(sample_space, {"omega0": 1, "omega1": 2, "omega2": 3})
        with pytest.raises(ValueError, match="probability measure is only defined"):
            _ = X.probability_measure

    def test_probability_measure_is_same_as_range_measure(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 2}
        )
        assert X.probability_measure == X.range.probability_measure

    def test_probability_measure_with_zero_probability_outcome(self):
        probs = {"omega0": 0.0, "omega1": 0.5, "omega2": 0.0, "omega3": 0.5}
        prob_space = sa.ProbabilitySpace(
            ["omega0", "omega1", "omega2", "omega3"], probs
        )
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 2}
        )
        P = X.probability_measure
        assert P(1) == 0.0
        assert P(2) == 1.0

    def test_probability_measure_with_non_uniform_distribution(self):
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_space = sa.ProbabilitySpace(
            ["omega0", "omega1", "omega2", "omega3"], probs
        )
        X = sa.RandomVariable(
            prob_space, {"omega0": "a", "omega1": "b", "omega2": "a", "omega3": "b"}
        )
        P = X.probability_measure
        assert P("a") - 0.4 < 1e-10
        assert P("b") - 0.6 < 1e-10


class TestRangeAndProbabilityMeasureIntegration:
    @pytest.fixture
    def prob_space(self):
        probs = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        return sa.ProbabilitySpace(["omega0", "omega1", "omega2", "omega3"], probs)

    def test_range_can_create_events(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 3}
        )
        range_space = X.range
        event = range_space[[1, 3]]
        assert isinstance(event, sa.Event)
        assert abs(event.probability - 0.8) < 1e-10

    def test_range_probability_measure_matches_direct_computation(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 3}
        )

        p1 = X.probability_measure(1)
        p2 = X.range.P(1)
        assert p1 == p2 == 0.4

    def test_pushforward_is_valid_probability_measure(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 3}
        )
        range_space = X.range

        for val in range_space.index:
            assert range_space.P(val) >= 0

        total = sum(range_space.P(val) for val in range_space.index)
        assert abs(total - 1.0) < 1e-10

    def test_constant_random_variable(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 42, "omega1": 42, "omega2": 42, "omega3": 42}
        )
        range_space = X.range
        assert len(range_space) == 1
        assert range_space.P(42) == 1.0

    def test_range_with_indicator_function(self, prob_space):
        X = sa.RandomVariable(
            prob_space, {"omega0": 1, "omega1": 1, "omega2": 0, "omega3": 0}
        )
        range_space = X.range
        assert abs(range_space.P(1) - 0.3) < 1e-10
        assert abs(range_space.P(0) - 0.7) < 1e-10


class TestAlgebra:

    @pytest.fixture
    def domain(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    @pytest.fixture
    def X(self, domain):
        values_X = {"omega0": 1, "omega1": 2, "omega2": 3}
        return sa.RandomVariable(domain=domain, values=values_X, name="X")

    @pytest.fixture
    def Y(self, domain):
        values_Y = {"omega0": 10, "omega1": 20, "omega2": 30}
        return sa.RandomVariable(domain=domain, values=values_Y, name="Y")

    def test_random_variable_add_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 11, "omega1": 22, "omega2": 33}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X+Y"
        )
        Z = X + Y
        assert Z == expected_rv

    def test_random_variable_add_with_scalar(self, domain, X):
        scalar = 5
        expected_values = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X+5"
        )
        Z = X + scalar
        assert Z == expected_rv

    def test_random_variable_radd_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 11, "omega1": 22, "omega2": 33}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="Y+X"
        )
        Z = Y + X
        assert Z == expected_rv

    def test_random_variable_radd_with_scalar(self, domain, X):
        scalar = 5
        expected_values = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="5+X"
        )
        Z = scalar + X
        assert Z == expected_rv

    def test_random_variable_mul_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 10, "omega1": 40, "omega2": 90}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X*Y"
        )
        Z = X * Y
        assert Z == expected_rv

    def test_random_variable_mul_with_scalar(self, domain, X):
        scalar = 3
        expected_values = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X*3"
        )
        Z = X * scalar
        assert Z == expected_rv

    def test_random_variable_rmul_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 10, "omega1": 40, "omega2": 90}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="Y*X"
        )
        Z = Y * X
        assert Z == expected_rv

    def test_random_variable_rmul_with_scalar(self, domain, X):
        scalar = 3
        expected_values = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="3*X"
        )
        Z = scalar * X
        assert Z == expected_rv

    def test_random_variable_sub_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": -9, "omega1": -18, "omega2": -27}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X-Y"
        )
        Z = X - Y
        assert Z == expected_rv

    def test_random_variable_sub_with_scalar(self, domain, X):
        scalar = 2
        expected_values = {"omega0": -1, "omega1": 0, "omega2": 1}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X-2"
        )
        Z = X - scalar
        assert Z == expected_rv

    def test_random_variable_rsub_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 9, "omega1": 18, "omega2": 27}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="Y-X"
        )
        Z = Y - X
        assert Z == expected_rv

    def test_random_variable_rsub_with_scalar(self, domain, X):
        scalar = 2
        expected_values = {"omega0": 1, "omega1": 0, "omega2": -1}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="2-X"
        )
        Z = scalar - X
        assert Z == expected_rv

    def test_random_variable_truediv_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 0.1, "omega1": 0.1, "omega2": 0.1}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X/Y"
        )
        Z = X / Y
        assert Z == expected_rv

    def test_random_variable_truediv_with_scalar(self, domain, X):
        scalar = 2
        expected_values = {"omega0": 0.5, "omega1": 1.0, "omega2": 1.5}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X/2"
        )
        Z = X / scalar
        assert Z == expected_rv

    def test_random_variable_rtruediv_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 10.0, "omega1": 10.0, "omega2": 10.0}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="Y/X"
        )
        Z = Y / X
        assert Z == expected_rv

    def test_random_variable_rtruediv_with_scalar(self, domain, X):
        scalar = 60
        expected_values = {"omega0": 60.0, "omega1": 30.0, "omega2": 20.0}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="60/X"
        )
        Z = scalar / X
        assert Z == expected_rv

    def test_random_variable_pow_with_random_variable(self, domain, X, Y):
        expected_values = {"omega0": 1**10, "omega1": 2**20, "omega2": 3**30}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X**Y"
        )
        Z = X**Y
        assert Z == expected_rv

    def test_random_variable_pow_with_scalar(self, domain, X):
        scalar = 2
        expected_values = {"omega0": 1**2, "omega1": 2**2, "omega2": 3**2}
        expected_rv = sa.RandomVariable(
            domain=domain, values=expected_values, name="X**2"
        )
        Z = X**scalar
        assert Z == expected_rv
