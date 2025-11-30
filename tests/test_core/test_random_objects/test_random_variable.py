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
        expected_outputs.index.name = "Omega"
        pd.testing.assert_series_equal(Y.values, expected_outputs)

    def test_construction_from_features(self, domain_features):
        def function(sample_features):
            return sample_features.feature_at[0] + sample_features.feature_at[1]

        X = sa.RandomVariable.from_features(
            fss=domain_features, function=function, name="X"
        )
        assert X.domain == domain_features.sample_space
        assert X.name == "X"
        assert X.function == function
        expected_outputs = pd.Series(
            data=[3, 7, 11], index=domain_features.sample_space, name="X"
        )
        expected_outputs.index.name = "Omega"
        pd.testing.assert_series_equal(X.values, expected_outputs)

    def test_construction_from_probability_space(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        X = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        assert X.domain == sample_space
        assert X.probability_space == prob_space
        assert X.name == "X"

    def test_construction_requires_domain_or_probability_space(self):
        outputs = {"s0": 10, "s1": 20}
        with pytest.raises(ValueError, match="Either domain or probability_space"):
            sa.RandomVariable(outputs=outputs, name="X")

    def test_construction_with_empty_outputs(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        with pytest.raises(ValueError, match="outputs dictionary cannot be empty"):
            sa.RandomVariable(domain=sample_space, outputs={}, name="X")

    def test_construction_with_mismatched_domain_outputs(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20}
        with pytest.raises(ValueError, match="outputs keys must match domain"):
            sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")

    def test_construction_with_non_callable_function(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        outputs = {"s0": 10, "s1": 20}
        with pytest.raises(TypeError, match="function must be callable"):
            sa.RandomVariable(
                domain=sample_space, outputs=outputs, function="not callable", name="X"
            )

    def test_construction_with_empty_name(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        outputs = {"s0": 10, "s1": 20}
        with pytest.raises(ValueError, match="name cannot be an empty string"):
            sa.RandomVariable(domain=sample_space, outputs=outputs, name="")

    def test_construction_with_inconsistent_domain_and_probability_space(self):
        sample_space1 = sa.SampleSpace(["s0", "s1"])
        sample_space2 = sa.SampleSpace(["a", "b"])
        prob_space = sa.ProbabilitySpace(sample_space=sample_space2)
        outputs = {"s0": 10, "s1": 20}
        with pytest.raises(
            ValueError,
            match="domain and probability_space.sample_space must be the same",
        ):
            sa.RandomVariable(
                domain=sample_space1, probability_space=prob_space, outputs=outputs
            )


class TestProperties:
    @pytest.fixture
    def rv(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        return sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")

    def test_values_property_returns_copy(self, rv):
        values1 = rv.values
        values2 = rv.values
        assert values1 is not values2
        pd.testing.assert_series_equal(values1, values2)

    def test_outputs_property_returns_copy(self, rv):
        outputs1 = rv.outputs
        outputs2 = rv.outputs
        assert outputs1 is not outputs2
        assert outputs1 == outputs2

    def test_probability_space_none_when_not_provided(self, rv):
        assert rv.probability_space is None

    def test_probability_measure_none_when_not_provided(self, rv):
        assert rv.probability_measure is None


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
            fss=domain_features, function=function, name="X"
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

    def test_call_rv_with_event(self, sample_space):
        outputs = dict(zip(sample_space, [4, 5, 6]))
        Z = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Z")
        event = sa.Event(sample_space=sample_space, event_indices=["s0", "s2"])
        result = Z(event)
        assert isinstance(result, sa.RandomVariable)
        expected_outputs = {"s0": 4, "s2": 6}
        assert result.domain == event.to_sample_space()
        assert result.name == "Z"
        assert result.outputs == expected_outputs

    def test_call_with_invalid_key_raises_error(self, sample_space):
        outputs = dict(zip(sample_space, [7, 8, 9]))
        Y = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        with pytest.raises(KeyError, match="not found in domain"):
            Y("invalid_key")

    def test_call_without_function_raises_error(self, sample_space, domain_features):
        outputs = dict(zip(sample_space, [7, 8, 9]))
        Y = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        sample_features = domain_features.get_sample_features("s0")
        with pytest.raises(ValueError, match="not defined with a function"):
            Y(sample_features)

    def test_sigma_algebra(self, sample_space):
        outputs = dict(zip(sample_space, [0, 1, 0]))
        U = sa.RandomVariable(domain=sample_space, outputs=outputs, name="U")
        sigma_algebra = U.sigma_algebra
        expected_atom_ids = {"s0": 0, "s1": 1, "s2": 0}
        assert sigma_algebra._sample_space == sample_space
        assert sigma_algebra._sample_id_to_atom_id == expected_atom_ids
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
        expected_df.index.name = "outputs"
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
            fss=domain_features, function=function, name="W"
        )
        range_space = W.range
        expected_df = pd.DataFrame(
            data=[[3], [6], [9]], index=["w0", "w1", "w2"], columns=["W"]
        )
        expected_df.index.name = "outputs"
        pd.testing.assert_frame_equal(range_space.features, expected_df)

    def test_range_with_all_unique_values(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        range_space = X.range
        assert len(range_space.sample_space) == 3

    def test_range_with_all_same_values(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 10, "s2": 10}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        range_space = X.range
        assert len(range_space.sample_space) == 1


class TestEquality:
    def test_equality_same_domain_same_values(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        X1 = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        X2 = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        assert X1 == X2

    def test_equality_different_domains(self):
        sample_space1 = sa.SampleSpace(["s0", "s1"])
        sample_space2 = sa.SampleSpace(["a", "b"])
        outputs1 = {"s0": 10, "s1": 20}
        outputs2 = {"a": 10, "b": 20}
        X1 = sa.RandomVariable(domain=sample_space1, outputs=outputs1, name="X")
        X2 = sa.RandomVariable(domain=sample_space2, outputs=outputs2, name="X")
        assert X1 != X2

    def test_equality_different_values(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs1 = {"s0": 10, "s1": 20, "s2": 30}
        outputs2 = {"s0": 10, "s1": 20, "s2": 40}
        X1 = sa.RandomVariable(domain=sample_space, outputs=outputs1, name="X")
        X2 = sa.RandomVariable(domain=sample_space, outputs=outputs2, name="X")
        assert X1 != X2

    def test_equality_with_non_random_variable(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        outputs = {"s0": 10, "s1": 20}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert X != "not a random variable"
        assert X != 42
        assert X is not None


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

    def test_operations_preserve_probability_space(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probabilities={"s0": 0.4, "s1": 0.6}
        )
        X = sa.RandomVariable(
            probability_space=prob_space, outputs={"s0": 1, "s1": 2}, name="X"
        )
        Y = X + 5
        assert Y.probability_space == prob_space
        Z = X * 2
        assert Z.probability_space == prob_space
        W = X**2
        assert W.probability_space == prob_space

    def test_operations_with_mismatched_domains_raise_error(self):
        domain1 = sa.SampleSpace(["s0", "s1"])
        domain2 = sa.SampleSpace(["a", "b"])
        X = sa.RandomVariable(domain=domain1, outputs={"s0": 1, "s1": 2}, name="X")
        Y = sa.RandomVariable(domain=domain2, outputs={"a": 3, "b": 4}, name="Y")
        with pytest.raises(ValueError, match="different domains"):
            X + Y
        with pytest.raises(ValueError, match="different domains"):
            X * Y
        with pytest.raises(ValueError, match="different domains"):
            X - Y
        with pytest.raises(ValueError, match="different domains"):
            X / Y
        with pytest.raises(ValueError, match="different domains"):
            X**Y


class TestProbabilityMethods:

    @pytest.fixture
    def fss(self):
        state_space = [0, 1]
        return sa.FeaturizedSampleSpace.from_sequences(
            state_space=state_space, sequence_length=3
        )

    @pytest.fixture
    def fps(self, fss):
        def pmf(sample_features: sa.SamplePointFeatures) -> float:
            num_ones = sample_features.sum()
            return 0.25**num_ones * 0.75 ** (3 - num_ones)

        return fss.add_probability_measure_from_features(pmf=pmf)

    @pytest.fixture
    def X_function(self):
        def function(sample_features: sa.SamplePointFeatures) -> int:
            return sample_features.sum()

        return function

    def test_construction_on_prob_space(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
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

    def test_P_method(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        # P(X=0)
        assert abs(X.P(0) - 0.75**3) < 1e-10
        # P(X=1)
        assert abs(X.P(1) - 3 * 0.25 * 0.75**2) < 1e-10
        # P(X=3)
        assert abs(X.P(3) - 0.25**3) < 1e-10

    def test_P_method_without_probability_space_raises_error(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        X = sa.RandomVariable(domain=sample_space, outputs={"s0": 1, "s1": 2}, name="X")
        with pytest.raises(
            ValueError, match="does not have an associated ProbabilityMeasure"
        ):
            X.P(1)

    def test_P_method_with_invalid_value_raises_error(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        with pytest.raises(ValueError, match="not in the range"):
            X.P(99)

    def test_call_on_event(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        probability_space = fps.probability_space
        # omega0 = 000, omega1 = 001, omega2 = 010
        event = probability_space.get_event_as_probability_space(
            ["omega0", "omega1", "omega2"]
        )
        event_prob = probability_space.P(["omega0", "omega1", "omega2"])
        X_restricted = X(event)
        assert isinstance(X_restricted, sa.RandomVariable)
        assert X_restricted.domain == event.sample_space
        assert abs(X_restricted.P(0) - 0.75**3 / event_prob) < 1e-10
        assert abs(X_restricted.P(1) - 2 * 0.25 * 0.75**2 / event_prob) < 1e-10

    def test_induced_probability_measure_sums_to_one(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        total_prob = sum(X.P(value) for value in [0, 1, 2, 3])
        assert abs(total_prob - 1.0) < 1e-10

    def test_unconditional_expectation(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        expected_expectation = (
            0 * 0.75**3
            + 1 * (3 * 0.25 * 0.75**2)
            + 2 * (3 * 0.25**2 * 0.75)
            + 3 * 0.25**3
        )
        actual_expectation = sa.unconditional_expectation(X)
        assert abs(actual_expectation - expected_expectation) < 1e-10

    def test_unconditional_expectation_without_probability_space_raises_error(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        X = sa.RandomVariable(domain=sample_space, outputs={"s0": 1, "s1": 2}, name="X")
        with pytest.raises(ValueError, match="must have a probability_space"):
            sa.unconditional_expectation(X)

    def test_expectation(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        atom_ids = dict(zip(X.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = sa.SigmaAlgebra(
            probability_space=X.probability_space, sample_id_to_atom_id=atom_ids
        )
        expectation = sa.expectation(rv=X, sigma_algebra=sigma_algebra)
        assert isinstance(expectation, sa.RandomVariable)
        assert expectation.name == "E(X|F)"
        expected_outputs = {
            "omega0": (0 * 0.75**3 + 1 * 0.25 * 0.75**2) / (0.75**3 + 0.25 * 0.75**2),
            "omega1": (0 * 0.75**3 + 1 * 0.25 * 0.75**2) / (0.75**3 + 0.25 * 0.75**2),
            "omega2": (1 * 0.25 * 0.75**2 + 2 * 0.25**2 * 0.75 + 1 * 0.25 * 0.75**2)
            / (0.25 * 0.75**2 + 0.25**2 * 0.75 + 0.25 * 0.75**2),
            "omega3": (1 * 0.25 * 0.75**2 + 2 * 0.25**2 * 0.75 + 1 * 0.25 * 0.75**2)
            / (0.25 * 0.75**2 + 0.25**2 * 0.75 + 0.25 * 0.75**2),
            "omega4": (1 * 0.25 * 0.75**2 + 2 * 0.25**2 * 0.75 + 1 * 0.25 * 0.75**2)
            / (0.25 * 0.75**2 + 0.25**2 * 0.75 + 0.25 * 0.75**2),
            "omega5": 2 * 0.25**2 * 0.75 / (0.25**2 * 0.75),
            "omega6": (2 * 0.25**2 * 0.75 + 3 * 0.25**3) / (0.25**2 * 0.75 + 0.25**3),
            "omega7": (2 * 0.25**2 * 0.75 + 3 * 0.25**3) / (0.25**2 * 0.75 + 0.25**3),
        }
        for sample_id in expectation.domain:
            assert (
                abs(expectation.outputs[sample_id] - expected_outputs[sample_id])
                < 1e-10
            )

    def test_expectation_without_sigma_algebra_returns_float(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        result = sa.expectation(X)
        assert isinstance(result, float)
        expected = (
            0 * 0.75**3
            + 1 * (3 * 0.25 * 0.75**2)
            + 2 * (3 * 0.25**2 * 0.75)
            + 3 * 0.25**3
        )
        assert abs(result - expected) < 1e-10

    def test_conditional_expectation_preserves_probability_space(self, fps, X_function):
        X = sa.RandomVariable.from_features(fps=fps, function=X_function, name="X")
        atom_ids = dict(zip(X.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = sa.SigmaAlgebra(
            probability_space=X.probability_space, sample_id_to_atom_id=atom_ids
        )
        expectation = sa.expectation(rv=X, sigma_algebra=sigma_algebra)
        assert expectation.probability_space == X.probability_space


class TestEdgeCases:
    def test_constant_random_variable(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 5, "s1": 5, "s2": 5}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert len(X.range.sample_space) == 1
        sigma_alg = X.sigma_algebra
        events = sigma_alg.to_events()
        assert len(events) == 1
        assert len(list(events.values())[0].values) == 3

    def test_indicator_random_variable(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        outputs = {"s0": 1, "s1": 0, "s2": 1, "s3": 0}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert len(X.range.sample_space) == 2
        assert X("s0") == 1
        assert X("s1") == 0

    def test_with_numeric_sample_indices(self):
        sample_space = sa.SampleSpace([0, 1, 2])
        outputs = {0: "a", 1: "b", 2: "c"}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert X(0) == "a"
        assert X(1) == "b"
        assert X(2) == "c"

    def test_with_string_outputs(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": "red", "s1": "green", "s2": "blue"}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="Color")
        assert X("s0") == "red"
        assert X("s1") == "green"

    def test_with_tuple_outputs(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        outputs = {"s0": (1, 2), "s1": (3, 4)}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert X("s0") == (1, 2)
        assert X("s1") == (3, 4)

    def test_large_sample_space(self):
        n = 1000
        sample_space = sa.SampleSpace([f"s{i}" for i in range(n)])
        outputs = {f"s{i}": i for i in range(n)}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert len(X.domain) == n
        assert X(f"s{500}") == 500


class TestIsMeasurable:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def prob_space(self, sample_space):
        probabilities = {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25}
        return sa.ProbabilitySpace(
            sample_space=sample_space, probabilities=probabilities
        )

    def test_rv_is_measurable_wrt_its_own_sigma_algebra(self, prob_space):
        outputs = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        assert X.is_measurable(X.sigma_algebra)

    def test_rv_is_measurable_wrt_power_set(self, prob_space):
        outputs = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        power_set = sa.SigmaAlgebra.power_set(probability_space=prob_space)
        assert X.is_measurable(power_set)

    def test_rv_not_measurable_wrt_trivial(self, prob_space):
        outputs = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        trivial = sa.SigmaAlgebra.trivial(probability_space=prob_space)
        assert not X.is_measurable(trivial)

    def test_constant_rv_measurable_wrt_trivial(self, prob_space):
        outputs = {"s0": 5, "s1": 5, "s2": 5, "s3": 5}
        X = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        trivial = sa.SigmaAlgebra.trivial(probability_space=prob_space)
        assert X.is_measurable(trivial)

    def test_rv_measurable_wrt_coarser_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            probability_space=prob_space, sample_id_to_atom_id=coarse_atom_ids
        )
        assert not X.is_measurable(coarse)

    def test_rv_not_measurable_wrt_finer_incompatible_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        incompatible_atom_ids = {"s0": 0, "s1": 1, "s2": 0, "s3": 2}
        incompatible = sa.SigmaAlgebra(
            probability_space=prob_space, sample_id_to_atom_id=incompatible_atom_ids
        )
        assert not X.is_measurable(incompatible)

    def test_rv_measurable_wrt_finer_compatible_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        power_set = sa.SigmaAlgebra.power_set(probability_space=prob_space)
        assert X.is_measurable(power_set)

    def test_is_measurable_uses_probability_space_sigma_algebra_by_default(
        self, prob_space
    ):
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        custom_sigma = sa.SigmaAlgebra(
            probability_space=prob_space, sample_id_to_atom_id=coarse_atom_ids
        )
        prob_space._sigma_algebra = custom_sigma
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        assert X.is_measurable()

    def test_finer_rv_measurable_wrt_coarser_rv_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        Y_outputs = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        Y = sa.RandomVariable(probability_space=prob_space, outputs=Y_outputs, name="Y")
        assert not Y.is_measurable(X.sigma_algebra)

    def test_coarser_rv_measurable_wrt_finer_rv_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        Y_outputs = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        Y = sa.RandomVariable(probability_space=prob_space, outputs=Y_outputs, name="Y")
        assert X.is_measurable(Y.sigma_algebra)

    def test_function_of_measurable_rv_is_measurable(self, prob_space):
        X_outputs = {"s0": 1, "s1": 1, "s2": 2, "s3": 2}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        Y_outputs = {"s0": 2, "s1": 2, "s2": 4, "s3": 4}
        Y = sa.RandomVariable(probability_space=prob_space, outputs=Y_outputs, name="Y")
        assert Y.sigma_algebra == X.sigma_algebra
        assert Y.is_measurable(X.sigma_algebra)

    def test_sum_of_measurable_rvs_is_measurable(self, prob_space):
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        coarse = sa.SigmaAlgebra(
            probability_space=prob_space, sample_id_to_atom_id=coarse_atom_ids
        )
        X_outputs = {"s0": 1, "s1": 1, "s2": 2, "s3": 2}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        Y_outputs = {"s0": 10, "s1": 10, "s2": 20, "s3": 20}
        Y = sa.RandomVariable(probability_space=prob_space, outputs=Y_outputs, name="Y")
        Z = X + Y
        assert Z.is_measurable(coarse)

    def test_measurability_with_three_rvs(self, prob_space):
        C_outputs = {"s0": 5, "s1": 5, "s2": 5, "s3": 5}
        C = sa.RandomVariable(probability_space=prob_space, outputs=C_outputs, name="C")
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        Y_outputs = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        Y = sa.RandomVariable(probability_space=prob_space, outputs=Y_outputs, name="Y")
        assert C.is_measurable(C.sigma_algebra)
        assert C.is_measurable(X.sigma_algebra)
        assert C.is_measurable(Y.sigma_algebra)
        assert not X.is_measurable(C.sigma_algebra)
        assert X.is_measurable(X.sigma_algebra)
        assert X.is_measurable(Y.sigma_algebra)
        assert not Y.is_measurable(C.sigma_algebra)
        assert not Y.is_measurable(X.sigma_algebra)
        assert Y.is_measurable(Y.sigma_algebra)

    def test_measurability_without_probability_space_raises_error(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        outputs = {"s0": 0, "s1": 1}
        X = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        with pytest.raises(ValueError):
            X.is_measurable()

    def test_measurability_respects_sigma_algebra_order(self, prob_space):
        trivial = sa.SigmaAlgebra.trivial(probability_space=prob_space)
        middle_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = sa.SigmaAlgebra(
            probability_space=prob_space, sample_id_to_atom_id=middle_atom_ids
        )
        power_set = sa.SigmaAlgebra.power_set(probability_space=prob_space)
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(probability_space=prob_space, outputs=X_outputs, name="X")
        assert not X.is_measurable(trivial)
        assert X.is_measurable(middle)
        assert X.is_measurable(power_set)
