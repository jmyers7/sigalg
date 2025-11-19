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
        return sa.FeaturizedSampleSpace(features=features, sample_space=sample_space)

    def test_construction_from_sample_space(self, sample_space):
        outputs = dict(zip(sample_space, [10, 20, 30]))
        Y = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        assert Y.domain == sample_space
        assert Y.name == "Y"
        expected_outputs = pd.Series(data=[10, 20, 30], index=sample_space, name="Y")
        pd.testing.assert_series_equal(Y.to_pandas(), expected_outputs)

    def test_construction_from_features(self, domain_features):
        def function(sample_features):
            return sample_features.feature_at[0] + sample_features.feature_at[1]

        X = sa.RandomVariable.from_features(
            domain_features=domain_features, function=function, name="X"
        )
        assert X.domain == domain_features.sample_space
        assert X.name == "X"
        assert X.function == function
        expected_outputs = pd.Series(
            data=[3, 7, 11], index=domain_features.sample_space, name="X"
        )
        pd.testing.assert_series_equal(X.to_pandas(), expected_outputs)


class TestMethods:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def domain_features(self, sample_space):
        features = [[1, 2], [3, 4], [5, 6]]
        return sa.FeaturizedSampleSpace(features=features, sample_space=sample_space)

    def test_call_rv_from_features(self, domain_features):
        def function(sample_features):
            return sample_features.feature_at[0] * 2

        X = sa.RandomVariable.from_features(
            domain_features=domain_features, function=function, name="X"
        )
        sample_features = domain_features.get_sample_features("s2")
        result = X(sample_features)
        assert result == 10
        result = X("s1")
        assert result == 6

    def test_call_rv_from_outputs(self, sample_space):
        outputs = dict(zip(sample_space, [7, 8, 9]))
        Y = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        result = Y("s0")
        assert result == 7
        result = Y("s2")
        assert result == 9

    def test_sigma_algebra(self, sample_space):
        outputs = dict(zip(sample_space, [0, 1, 0]))
        U = sa.RandomVariable(domain=sample_space, outputs=outputs, name="U")
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
    def test_range_property(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = dict(zip(sample_space, [15, 10, 15]))
        Z = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Z")
        range_space = Z.range
        expected_df = pd.DataFrame(data=[[15], [10]], index=["z0", "z1"], columns=["Z"])
        pd.testing.assert_frame_equal(range_space.features, expected_df)

    def test_range_property_with_function(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])

        def function(sample_features):
            return sample_features.feature_at[0] * 3

        features = [[1], [2], [3]]
        domain_features = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        W = sa.RandomVariable.from_features(
            domain_features=domain_features, function=function, name="W"
        )
        range_space = W.range
        expected_df = pd.DataFrame(
            data=[[3], [6], [9]], index=["w0", "w1", "w2"], columns=["W"]
        )
        pd.testing.assert_frame_equal(range_space.features, expected_df)


class TestAlgebra:

    @pytest.fixture
    def domain(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    @pytest.fixture
    def X(self, domain):
        outputs_X = {"omega0": 1, "omega1": 2, "omega2": 3}
        return sa.RandomVariable(domain=domain, outputs=outputs_X, name="X")

    @pytest.fixture
    def Y(self, domain):
        outputs_Y = {"omega0": 10, "omega1": 20, "omega2": 30}
        return sa.RandomVariable(domain=domain, outputs=outputs_Y, name="Y")

    def test_random_variable_add_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 11, "omega1": 22, "omega2": 33}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X+Y"
        )
        Z = X + Y
        assert Z == expected_rv

    def test_random_variable_add_with_scalar(self, domain, X):
        scalar = 5
        expected_outputs = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X+5"
        )
        Z = X + scalar
        assert Z == expected_rv

    def test_random_variable_radd_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 11, "omega1": 22, "omega2": 33}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="Y+X"
        )
        Z = Y + X
        assert Z == expected_rv

    def test_random_variable_radd_with_scalar(self, domain, X):
        scalar = 5
        expected_outputs = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="5+X"
        )
        Z = scalar + X
        assert Z == expected_rv

    def test_random_variable_mul_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 10, "omega1": 40, "omega2": 90}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X*Y"
        )
        Z = X * Y
        assert Z == expected_rv

    def test_random_variable_mul_with_scalar(self, domain, X):
        scalar = 3
        expected_outputs = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X*3"
        )
        Z = X * scalar
        assert Z == expected_rv

    def test_random_variable_rmul_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 10, "omega1": 40, "omega2": 90}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="Y*X"
        )
        Z = Y * X
        assert Z == expected_rv

    def test_random_variable_rmul_with_scalar(self, domain, X):
        scalar = 3
        expected_outputs = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="3*X"
        )
        Z = scalar * X
        assert Z == expected_rv

    def test_random_variable_sub_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": -9, "omega1": -18, "omega2": -27}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X-Y"
        )
        Z = X - Y
        assert Z == expected_rv

    def test_random_variable_sub_with_scalar(self, domain, X):
        scalar = 2
        expected_outputs = {"omega0": -1, "omega1": 0, "omega2": 1}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X-2"
        )
        Z = X - scalar
        assert Z == expected_rv

    def test_random_variable_rsub_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 9, "omega1": 18, "omega2": 27}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="Y-X"
        )
        Z = Y - X
        assert Z == expected_rv

    def test_random_variable_rsub_with_scalar(self, domain, X):
        scalar = 2
        expected_outputs = {"omega0": 1, "omega1": 0, "omega2": -1}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="2-X"
        )
        Z = scalar - X
        assert Z == expected_rv

    def test_random_variable_truediv_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 0.1, "omega1": 0.1, "omega2": 0.1}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X/Y"
        )
        Z = X / Y
        assert Z == expected_rv

    def test_random_variable_truediv_with_scalar(self, domain, X):
        scalar = 2
        expected_outputs = {"omega0": 0.5, "omega1": 1.0, "omega2": 1.5}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X/2"
        )
        Z = X / scalar
        assert Z == expected_rv

    def test_random_variable_rtruediv_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 10.0, "omega1": 10.0, "omega2": 10.0}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="Y/X"
        )
        Z = Y / X
        assert Z == expected_rv

    def test_random_variable_rtruediv_with_scalar(self, domain, X):
        scalar = 60
        expected_outputs = {"omega0": 60.0, "omega1": 30.0, "omega2": 20.0}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="60/X"
        )
        Z = scalar / X
        assert Z == expected_rv

    def test_random_variable_pow_with_random_variable(self, domain, X, Y):
        expected_outputs = {"omega0": 1**10, "omega1": 2**20, "omega2": 3**30}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X**Y"
        )
        Z = X**Y
        assert Z == expected_rv

    def test_random_variable_pow_with_scalar(self, domain, X):
        scalar = 2
        expected_outputs = {"omega0": 1**2, "omega1": 2**2, "omega2": 3**2}
        expected_rv = sa.RandomVariable(
            domain=domain, outputs=expected_outputs, name="X**2"
        )
        Z = X**scalar
        assert Z == expected_rv


class TestProbabilityMethods:
    def test_construction_on_prob_space(self):
        state_space = [0, 1]
        fss = sa.FeaturizedSampleSpace.from_sequences(
            state_space=state_space, sequence_length=3
        )

        def pmf(sample_features: sa.SamplePointFeatures) -> float:
            num_ones = sample_features.sum()
            return 0.25**num_ones * 0.75 ** (3 - num_ones)

        def X_function(sample_features: sa.SamplePointFeatures) -> int:
            return sample_features.sum()

        fps = fss.add_probability_measure_from_features(pmf=pmf)
        X = sa.RandomVariable.from_features(
            domain_features=fps, function=X_function, name="X"
        )
        range = X.range
        assert isinstance(range, sa.FeaturizedProbabilitySpace)
        expected_probabilities = {
            "x0": 0.75**3,  # P(X=0)
            "x1": 3 * 0.25 * 0.75**2,  # P(X=1)
            "x2": 3 * 0.25**2 * 0.75,  # P(X=2)
            "x3": 0.25**3,  # P(X=3)
        }
        actual_probabilities = {idx: range.P(idx) for idx in range.sample_space}
        for idx in expected_probabilities:
            assert abs(actual_probabilities[idx] - expected_probabilities[idx]) < 1e-10
