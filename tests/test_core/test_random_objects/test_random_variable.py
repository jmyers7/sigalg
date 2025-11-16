import pandas as pd
import pytest

import sigalg as sa


class TestConstructionAndBasicProperties:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

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
        pd.testing.assert_series_equal(Y.to_pandas(), expected_values)

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
        pd.testing.assert_series_equal(X.to_pandas(), expected_values)


class TestMethods:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def domain_features(self, sample_space):
        features = [[1, 2], [3, 4], [5, 6]]
        return sa.SampleSpaceFeatures(features=features, sample_space=sample_space)

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

    def test_range_from_regular_sample_space(self, sample_space):
        X = sa.RandomVariable(
            sample_space, {"omega0": 1, "omega1": 2, "omega2": 1, "omega3": 3}
        )
        range_space = X.range
        assert isinstance(range_space, sa.SampleSpace)
        assert not isinstance(range_space, sa.ProbabilitySpace)
        expected_range = sa.SampleSpace([1, 2, 3])
        assert range_space == expected_range

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


class TestHashMethod:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def values(self):
        return {"s0": 1, "s1": 2, "s2": 3}

    def test_hash_is_consistent(self, sample_space, values):
        rv = sa.RandomVariable(domain=sample_space, values=values, name="X")
        hash1 = hash(rv)
        hash2 = hash(rv)
        assert hash1 == hash2

    def test_hash_is_cached(self, sample_space, values):
        rv = sa.RandomVariable(domain=sample_space, values=values, name="X")
        # First call computes hash
        hash1 = hash(rv)
        assert rv._hash is not None
        # Second call should return cached value
        hash2 = hash(rv)
        assert hash1 == hash2
        assert hash1 == rv._hash

    def test_equal_random_variables_have_equal_hashes(self, sample_space, values):
        rv1 = sa.RandomVariable(domain=sample_space, values=values, name="X")
        rv2 = sa.RandomVariable(domain=sample_space, values=values, name="X")
        assert rv1 == rv2
        assert hash(rv1) == hash(rv2)

    def test_different_values_have_different_hashes(self, sample_space):
        values1 = {"s0": 1, "s1": 2, "s2": 3}
        values2 = {"s0": 1, "s1": 2, "s2": 4}
        rv1 = sa.RandomVariable(domain=sample_space, values=values1, name="X")
        rv2 = sa.RandomVariable(domain=sample_space, values=values2, name="Y")
        assert rv1 != rv2
        assert hash(rv1) != hash(rv2)

    def test_different_names_have_different_hashes(self, sample_space, values):
        rv1 = sa.RandomVariable(domain=sample_space, values=values, name="X")
        rv2 = sa.RandomVariable(domain=sample_space, values=values, name="Y")
        # They are equal based on domain and values, but hash includes name
        assert rv1 == rv2  # equality ignores name
        assert hash(rv1) != hash(rv2)  # hash includes name

    def test_different_domains_have_different_hashes(self, values):
        domain1 = sa.SampleSpace(["s0", "s1", "s2"])
        domain2 = sa.SampleSpace(["a", "b", "c"])
        values2 = {"a": 1, "b": 2, "c": 3}
        rv1 = sa.RandomVariable(domain=domain1, values=values, name="X")
        rv2 = sa.RandomVariable(domain=domain2, values=values2, name="X")
        assert rv1 != rv2
        assert hash(rv1) != hash(rv2)

    def test_random_variable_can_be_in_set(self, sample_space):
        values1 = {"s0": 1, "s1": 2, "s2": 3}
        values2 = {"s0": 4, "s1": 5, "s2": 6}
        rv1 = sa.RandomVariable(domain=sample_space, values=values1, name="X")
        rv2 = sa.RandomVariable(domain=sample_space, values=values2, name="Y")
        rv3 = sa.RandomVariable(domain=sample_space, values=values1, name="X")

        rv_set = {rv1, rv2, rv3}
        # rv1 and rv3 should be considered the same in a set
        assert len(rv_set) == 2
        assert rv1 in rv_set
        assert rv2 in rv_set
        assert rv3 in rv_set

    def test_random_variable_can_be_dict_key(self, sample_space):
        values1 = {"s0": 1, "s1": 2, "s2": 3}
        values2 = {"s0": 4, "s1": 5, "s2": 6}
        rv1 = sa.RandomVariable(domain=sample_space, values=values1, name="X")
        rv2 = sa.RandomVariable(domain=sample_space, values=values2, name="Y")

        rv_dict = {rv1: "first", rv2: "second"}
        assert rv_dict[rv1] == "first"
        assert rv_dict[rv2] == "second"

    def test_hash_with_different_value_order(self, sample_space):
        # Values are stored as sorted tuples, so order shouldn't matter
        values1 = {"s0": 1, "s1": 2, "s2": 3}
        values2 = {"s2": 3, "s0": 1, "s1": 2}
        rv1 = sa.RandomVariable(domain=sample_space, values=values1, name="X")
        rv2 = sa.RandomVariable(domain=sample_space, values=values2, name="X")
        assert rv1 == rv2
        assert hash(rv1) == hash(rv2)

    def test_hash_with_string_values(self):
        sample_space = sa.SampleSpace(["a", "b", "c"])
        values = {"a": "x", "b": "y", "c": "z"}
        rv = sa.RandomVariable(domain=sample_space, values=values, name="X")
        hash_value = hash(rv)
        assert isinstance(hash_value, int)

    def test_hash_with_tuple_values(self):
        sample_space = sa.SampleSpace(["a", "b"])
        values = {"a": (1, 2), "b": (3, 4)}
        rv = sa.RandomVariable(domain=sample_space, values=values, name="X")
        hash_value = hash(rv)
        assert isinstance(hash_value, int)

    def test_hash_with_none_values(self, sample_space):
        values = {"s0": None, "s1": None, "s2": None}
        rv = sa.RandomVariable(domain=sample_space, values=values, name="X")
        hash_value = hash(rv)
        assert isinstance(hash_value, int)

    def test_hash_with_mixed_type_values(self, sample_space):
        values = {"s0": 1, "s1": "two", "s2": 3.0}
        rv = sa.RandomVariable(domain=sample_space, values=values, name="X")
        hash_value = hash(rv)
        assert isinstance(hash_value, int)

    def test_hash_stability_across_instances(self, sample_space, values):
        # Create multiple instances with same parameters
        rv1 = sa.RandomVariable(domain=sample_space, values=values, name="X")
        rv2 = sa.RandomVariable(domain=sample_space, values=values, name="X")
        rv3 = sa.RandomVariable(domain=sample_space, values=values, name="X")

        hashes = [hash(rv1), hash(rv2), hash(rv3)]
        assert len(set(hashes)) == 1  # All hashes should be identical


class TestHashWithArithmeticOperations:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def rv1(self, sample_space):
        values = {"s0": 1, "s1": 2, "s2": 3}
        return sa.RandomVariable(domain=sample_space, values=values, name="X")

    @pytest.fixture
    def rv2(self, sample_space):
        values = {"s0": 4, "s1": 5, "s2": 6}
        return sa.RandomVariable(domain=sample_space, values=values, name="Y")

    def test_hash_of_sum(self, rv1, rv2):
        rv_sum = rv1 + rv2
        hash_value = hash(rv_sum)
        assert isinstance(hash_value, int)

    def test_hash_of_product(self, rv1, rv2):
        rv_product = rv1 * rv2
        hash_value = hash(rv_product)
        assert isinstance(hash_value, int)

    def test_hash_of_difference(self, rv1, rv2):
        rv_diff = rv1 - rv2
        hash_value = hash(rv_diff)
        assert isinstance(hash_value, int)

    def test_hash_of_quotient(self, rv1, rv2):
        rv_quot = rv1 / rv2
        hash_value = hash(rv_quot)
        assert isinstance(hash_value, int)

    def test_hash_of_power(self, rv1):
        rv_pow = rv1**2
        hash_value = hash(rv_pow)
        assert isinstance(hash_value, int)

    def test_different_arithmetic_results_have_different_hashes(self, rv1, rv2):
        rv_sum = rv1 + rv2
        rv_product = rv1 * rv2
        assert hash(rv_sum) != hash(rv_product)

    def test_same_arithmetic_operations_have_same_hashes(self, rv1, rv2):
        rv_sum1 = rv1 + rv2
        rv_sum2 = rv1 + rv2
        assert hash(rv_sum1) == hash(rv_sum2)
