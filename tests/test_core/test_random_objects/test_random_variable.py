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
