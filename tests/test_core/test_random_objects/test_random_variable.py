import pytest
import sigalg as sa
import pandas as pd


class TestConstructionAndBasicProperties:
    @pytest.fixture
    def space_features(self):
        data = [[1, 2], [3, 4], [5, 6]]
        return sa.SampleSpaceFeatures(data)

    @pytest.fixture
    def space_features_with_custom_labels(self):
        data = [[1, 2], [3, 4], [5, 6]]
        return sa.SampleSpaceFeatures(data=data, sample_prefix="s", feature_prefix="Y")

    def test_construction_from_function(self, space_features):
        def function(sample_features):
            return sample_features[0] ** 2 + sample_features[1] ** 2

        X = sa.RandomVariable(domain_features=space_features, function=function)
        omega0_features = space_features.get_sample_features("omega0")
        omega1_features = space_features.get_sample_features("omega1")
        omega2_features = space_features.get_sample_features("omega2")
        assert X(omega0_features) == 5
        assert X(omega1_features) == 25
        assert X(omega2_features) == 61
        assert X("omega0") == 5
        assert X("omega1") == 25
        assert X("omega2") == 61
        X.function(omega0_features) == 5
        X.function(omega1_features) == 25
        X.function(omega2_features) == 61
        X.idx_function("omega0") == 5
        X.idx_function("omega1") == 25
        X.idx_function("omega2") == 61
        assert X.name == "X"
        assert X.domain == space_features
        pd.testing.assert_series_equal(
            X.values,
            pd.Series(data=[5, 25, 61], index=space_features.sample_index, name="X"),
        )

    def test_from_values(self, space_features):
        values = {"omega0": 3, "omega1": 7, "omega2": 11}
        Z = sa.RandomVariable(domain_features=space_features, values=values, name="Z")
        omega0_features = space_features.get_sample_features("omega0")
        omega1_features = space_features.get_sample_features("omega1")
        omega2_features = space_features.get_sample_features("omega2")
        assert Z(omega0_features) == 3
        assert Z(omega1_features) == 7
        assert Z(omega2_features) == 11
        assert Z("omega0") == 3
        assert Z("omega1") == 7
        assert Z("omega2") == 11
        assert Z.function(omega0_features) == 3
        assert Z.function(omega1_features) == 7
        assert Z.function(omega2_features) == 11
        assert Z.idx_function("omega0") == 3
        assert Z.idx_function("omega1") == 7
        assert Z.idx_function("omega2") == 11
        assert Z.name == "Z"
        assert Z.domain == space_features
        pd.testing.assert_series_equal(
            Z.values,
            pd.Series(data=[3, 7, 11], index=space_features.sample_index, name="Z"),
        )

    def test_random_variable_set_name(self, space_features):
        values = {"omega0": 3, "omega1": 7, "omega2": 11}
        rv = sa.RandomVariable(domain_features=space_features, values=values, name="Z")
        assert rv.name == "Z"
        rv.set_name("W")
        assert rv.name == "W"


class TestAlgebra:

    @pytest.fixture
    def space_features(self):
        data = [[1, 2], [3, 4], [5, 6]]
        return sa.SampleSpaceFeatures(data)

    @pytest.fixture
    def X(self, space_features):
        values_X = {"omega0": 1, "omega1": 2, "omega2": 3}
        return sa.RandomVariable(domain_features=space_features, values=values_X, name="X")

    @pytest.fixture
    def Y(self, space_features):
        values_Y = {"omega0": 10, "omega1": 20, "omega2": 30}
        return sa.RandomVariable(domain_features=space_features, values=values_Y, name="Y")

    def test_random_variable_add_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 11, "omega1": 22, "omega2": 33}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X+Y"
        )
        Z = X + Y
        assert Z == expected_rv

    def test_random_variable_add_with_scalar(self, space_features, X):
        scalar = 5
        expected_values = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X+5"
        )
        Z = X + scalar
        assert Z == expected_rv

    def test_random_variable_radd_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 11, "omega1": 22, "omega2": 33}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="Y+X"
        )
        Z = Y + X
        assert Z == expected_rv

    def test_random_variable_radd_with_scalar(self, space_features, X):
        scalar = 5
        expected_values = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="5+X"
        )
        Z = scalar + X
        assert Z == expected_rv

    def test_random_variable_mul_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 10, "omega1": 40, "omega2": 90}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X*Y"
        )
        Z = X * Y
        assert Z == expected_rv

    def test_random_variable_mul_with_scalar(self, space_features, X):
        scalar = 3
        expected_values = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X*3"
        )
        Z = X * scalar
        assert Z == expected_rv

    def test_random_variable_rmul_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 10, "omega1": 40, "omega2": 90}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="Y*X"
        )
        Z = Y * X
        assert Z == expected_rv

    def test_random_variable_rmul_with_scalar(self, space_features, X):
        scalar = 3
        expected_values = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="3*X"
        )
        Z = scalar * X
        assert Z == expected_rv

    def test_random_variable_sub_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": -9, "omega1": -18, "omega2": -27}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X-Y"
        )
        Z = X - Y
        assert Z == expected_rv

    def test_random_variable_sub_with_scalar(self, space_features, X):
        scalar = 2
        expected_values = {"omega0": -1, "omega1": 0, "omega2": 1}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X-2"
        )
        Z = X - scalar
        assert Z == expected_rv

    def test_random_variable_rsub_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 9, "omega1": 18, "omega2": 27}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="Y-X"
        )
        Z = Y - X
        assert Z == expected_rv

    def test_random_variable_rsub_with_scalar(self, space_features, X):
        scalar = 2
        expected_values = {"omega0": 1, "omega1": 0, "omega2": -1}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="2-X"
        )
        Z = scalar - X
        assert Z == expected_rv

    def test_random_variable_truediv_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 0.1, "omega1": 0.1, "omega2": 0.1}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X/Y"
        )
        Z = X / Y
        assert Z == expected_rv

    def test_random_variable_truediv_with_scalar(self, space_features, X):
        scalar = 2
        expected_values = {"omega0": 0.5, "omega1": 1.0, "omega2": 1.5}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X/2"
        )
        Z = X / scalar
        assert Z == expected_rv

    def test_random_variable_rtruediv_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 10.0, "omega1": 10.0, "omega2": 10.0}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="Y/X"
        )
        Z = Y / X
        assert Z == expected_rv

    def test_random_variable_rtruediv_with_scalar(self, space_features, X):
        scalar = 60
        expected_values = {"omega0": 60.0, "omega1": 30.0, "omega2": 20.0}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="60/X"
        )
        Z = scalar / X
        assert Z == expected_rv

    def test_random_variable_pow_with_random_variable(self, space_features, X, Y):
        expected_values = {"omega0": 1**10, "omega1": 2**20, "omega2": 3**30}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X**Y"
        )
        Z = X**Y
        assert Z == expected_rv

    def test_random_variable_pow_with_scalar(self, space_features, X):
        scalar = 2
        expected_values = {"omega0": 1**2, "omega1": 2**2, "omega2": 3**2}
        expected_rv = sa.RandomVariable(
            domain_features=space_features, values=expected_values, name="X**2"
        )
        Z = X**scalar
        assert Z == expected_rv
