import pandas as pd
import pytest

from sigalg.core import (
    Event,
    FeatureEmbedding,
    FeaturizedProbabilitySpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    SamplePointFeatures,
    SampleSpace,
    SigmaAlgebra,
)
from sigalg.l2 import expectation


class TestConstructor:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2"])

    def test_construction_from_sample_space(self, sample_space):
        outputs = dict(zip(sample_space, [10, 20, 30]))
        Y = RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        assert Y.domain == sample_space
        assert Y.name == "Y"
        expected_outputs = pd.Series(data=[10, 20, 30], index=sample_space, name="Y")
        expected_outputs.index.name = "sample"
        pd.testing.assert_series_equal(Y.values, expected_outputs)

    def test_construction_from_values_with_name(self):
        values = pd.Series(data=[10, 20, 30], index=["s0", "s1", "s2"], name="Y")
        Y = RandomVariable(values=values, name="X")
        assert Y.domain == SampleSpace(["s0", "s1", "s2"])
        assert Y.name == "Y"
        pd.testing.assert_series_equal(Y.values, values)

    def test_construction_from_values_no_name(self):
        values = pd.Series(data=[10, 20, 30], index=["s0", "s1", "s2"])
        Z = RandomVariable(values=values, name="Z")
        assert Z.domain == SampleSpace(["s0", "s1", "s2"])
        assert Z.name == "Z"
        pd.testing.assert_series_equal(Z.values, values)


class TestClassMethods:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def feature_embedding(self, sample_space):
        features = pd.DataFrame(
            data=[[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
        )
        return FeatureEmbedding(values=features)

    def test_from_features(self, feature_embedding):
        def function(sample_features):
            return sample_features.feature_at[0] + sample_features.feature_at[1]

        X = RandomVariable.from_features(
            feature_embedding=feature_embedding, function=function, name="X"
        )
        assert X.domain == feature_embedding.domain
        assert X.name == "X"
        assert X.function == function
        expected_outputs = pd.Series(
            data=[3, 7, 11], index=feature_embedding.domain, name="X"
        )
        expected_outputs.index.name = "sample"
        pd.testing.assert_series_equal(X.values, expected_outputs)

    def test_on_probability_space(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        X = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        assert X.domain == sample_space
        assert X.probability_space == prob_space
        assert X.name == "X"


class TestValidation:

    def test_cannot_provide_both_outputs_and_values(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 1, "s1": 2, "s2": 3}
        values = pd.Series([1, 2, 3], index=["s0", "s1", "s2"])
        with pytest.raises(
            ValueError, match="Cannot provide both outputs/domain and values"
        ):
            RandomVariable(outputs=outputs, domain=sample_space, values=values)

    def test_cannot_provide_domain_and_values(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        values = pd.Series([1, 2, 3], index=["s0", "s1", "s2"])
        with pytest.raises(
            ValueError, match="Cannot provide both outputs/domain and values"
        ):
            RandomVariable(domain=sample_space, values=values)

    def test_must_provide_either_outputs_or_values(self):
        with pytest.raises(
            ValueError, match="Must provide either outputs/domain or values"
        ):
            RandomVariable(name="X")

    def test_outputs_must_be_dict(self):
        sample_space = SampleSpace(["s0", "s1"])
        with pytest.raises(TypeError, match="outputs must be a dict"):
            RandomVariable(outputs=[1, 2, 3], domain=sample_space)

    def test_outputs_must_be_dict_not_list(self):
        sample_space = SampleSpace(["s0", "s1"])
        with pytest.raises(TypeError, match="outputs must be a dict"):
            RandomVariable(outputs=[(1, 2), (3, 4)], domain=sample_space)

    def test_domain_must_be_sample_space(self):
        outputs = {"s0": 1, "s1": 2}
        with pytest.raises(TypeError, match="domain must be a SampleSpace instance"):
            RandomVariable(outputs=outputs, domain="not_a_sample_space")

    def test_domain_must_be_sample_space_not_list(self):
        outputs = {"s0": 1, "s1": 2}
        with pytest.raises(TypeError, match="domain must be a SampleSpace instance"):
            RandomVariable(outputs=outputs, domain=["s0", "s1"])

    def test_outputs_keys_must_be_in_domain(self):
        sample_space = SampleSpace(["s0", "s1"])
        outputs = {"s0": 1, "s1": 2, "s2": 3}
        with pytest.raises(
            ValueError, match="All keys in outputs must be in the domain"
        ):
            RandomVariable(outputs=outputs, domain=sample_space)

    def test_outputs_keys_partially_in_domain_invalid(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 1, "invalid": 2}
        with pytest.raises(
            ValueError, match="All keys in outputs must be in the domain"
        ):
            RandomVariable(outputs=outputs, domain=sample_space)

    def test_values_must_be_series(self):
        with pytest.raises(TypeError, match="values must be a pandas Series instance"):
            RandomVariable(values=[1, 2, 3])

    def test_values_must_be_series_not_dataframe(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        with pytest.raises(TypeError, match="values must be a pandas Series instance"):
            RandomVariable(values=df)

    def test_values_must_be_series_not_dict(self):
        with pytest.raises(TypeError, match="values must be a pandas Series instance"):
            RandomVariable(values={"s0": 1, "s1": 2})

    def test_name_must_be_string(self):
        values = pd.Series([1, 2, 3], index=["s0", "s1", "s2"])
        with pytest.raises(TypeError, match="name must be a string"):
            RandomVariable(values=values, name=123)

    def test_name_must_be_string_not_none(self):
        values = pd.Series([1, 2, 3], index=["s0", "s1", "s2"])
        with pytest.raises(TypeError, match="name must be a string"):
            RandomVariable(values=values, name=None)

    def test_name_setter_must_be_string(self):
        values = pd.Series([1, 2, 3], index=["s0", "s1", "s2"])
        X = RandomVariable(values=values, name="X")
        with pytest.raises(TypeError, match="name must be a string"):
            X.name = 456


class TestProperties:
    @pytest.fixture
    def rv(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        return RandomVariable(domain=sample_space, outputs=outputs, name="X")

    def test_probability_space_none_when_not_provided(self, rv):
        assert rv.probability_space is None

    def test_probability_measure_none_when_not_provided(self, rv):
        assert rv.probability_measure is None

    def test_domain_property(self, rv):
        assert isinstance(rv.domain, SampleSpace)

    def test_name_property(self, rv):
        assert rv.name == "X"

    def test_sigma_algebra_property(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = dict(zip(sample_space, [0, 1, 0]))
        U = RandomVariable(domain=sample_space, outputs=outputs, name="U")
        sigma_algebra = U.sigma_algebra
        expected_atom_ids = {"s0": 0, "s1": 1, "s2": 0}
        assert sigma_algebra.sample_space == sample_space
        assert sigma_algebra.sample_id_to_atom_id == expected_atom_ids
        expected_events = {
            0: Event(sample_space=sample_space, event_indices=["s0", "s2"]),
            1: Event(sample_space=sample_space, event_indices=["s1"]),
        }
        actual_events = sigma_algebra.atom_id_to_event
        assert actual_events == expected_events

    def test_sigma_algebra_property_with_values(self):
        values = pd.Series([0, 1, 0], index=["s0", "s1", "s2"], name="U")
        U = RandomVariable(values=values)
        sigma_algebra = U.sigma_algebra
        expected_atom_ids = {"s0": 0, "s1": 1, "s2": 0}
        assert sigma_algebra.sample_id_to_atom_id == expected_atom_ids
        expected_events = {
            0: Event(sample_space=U.domain, event_indices=["s0", "s2"]),
            1: Event(sample_space=U.domain, event_indices=["s1"]),
        }
        actual_events = sigma_algebra.atom_id_to_event
        assert actual_events == expected_events

    def test_range_property(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = dict(zip(sample_space, [15, 10, 15]))
        Z = RandomVariable(domain=sample_space, outputs=outputs, name="Z")
        range_space = Z.range
        expected_df = pd.DataFrame(data=[[15], [10]], index=["z0", "z1"], columns=["Z"])
        expected_df.index.name = "outputs"
        pd.testing.assert_frame_equal(range_space.values, expected_df)

    def test_range_property_with_function(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])

        def function(sample_features):
            return sample_features.feature_at[0] * 3

        features = pd.DataFrame(data=[[1], [2], [3]], index=sample_space)
        feature_embedding = FeatureEmbedding(values=features)
        W = RandomVariable.from_features(
            feature_embedding=feature_embedding, function=function, name="W"
        )
        range_space = W.range
        expected_df = pd.DataFrame(
            data=[[3], [6], [9]], index=["w0", "w1", "w2"], columns=["W"]
        )
        expected_df.index.name = "outputs"
        pd.testing.assert_frame_equal(range_space.values, expected_df)

    def test_range_with_all_unique_values(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        range_space = X.range
        assert len(range_space.domain) == 3

    def test_range_with_all_same_values(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 10, "s2": 10}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        range_space = X.range
        assert len(range_space.domain) == 1

    def test_range_property_with_values(self):
        values = pd.Series([15, 10, 15], index=["s0", "s1", "s2"], name="Z")
        Z = RandomVariable(values=values)
        range_space = Z.range
        expected_df = pd.DataFrame(data=[[15], [10]], index=["z0", "z1"], columns=["Z"])
        expected_df.index.name = "outputs"
        pd.testing.assert_frame_equal(range_space.values, expected_df)


class TestCallMethod:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2"])

    @pytest.fixture
    def feature_embedding(self, sample_space):
        features = pd.DataFrame(data=[[1, 2], [3, 4], [5, 6]], index=sample_space)
        return FeatureEmbedding(values=features)

    def test_call_rv_from_features(self, feature_embedding):
        def function(sample_features):
            return sample_features.feature_at[0] * 2

        X = RandomVariable.from_features(
            feature_embedding=feature_embedding, function=function, name="X"
        )
        sample_features = feature_embedding.get_sample_features("s2")
        result = X(sample_features)
        assert result == 10
        result = X("s1")
        assert result == 6

    def test_call_rv_from_outputs(self, sample_space):
        outputs = dict(zip(sample_space, [7, 8, 9]))
        Y = RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        result = Y("s0")
        assert result == 7
        result = Y("s2")
        assert result == 9

    def test_call_rv_with_event(self, sample_space):
        outputs = dict(zip(sample_space, [4, 5, 6]))
        Z = RandomVariable(domain=sample_space, outputs=outputs, name="Z")
        event = Event(sample_space=sample_space, event_indices=["s0", "s2"])
        result = Z(event)
        assert isinstance(result, RandomVariable)
        expected_outputs = {"s0": 4, "s2": 6}
        assert result.domain == event.to_sample_space()
        assert result.name == "Z"
        assert result.outputs == expected_outputs

    def test_call_with_invalid_key_raises_error(self, sample_space):
        outputs = dict(zip(sample_space, [7, 8, 9]))
        Y = RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        with pytest.raises(KeyError, match="not found in domain"):
            Y("invalid_key")

    def test_call_without_function_raises_error(self, sample_space, feature_embedding):
        outputs = dict(zip(sample_space, [7, 8, 9]))
        Y = RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        sample_features = feature_embedding.get_sample_features("s0")
        with pytest.raises(ValueError, match="not defined with a function"):
            Y(sample_features)

    def test_call_rv_from_values(self, sample_space):
        values = pd.Series([7, 8, 9], index=sample_space.values, name="Y")
        Y = RandomVariable(values=values)
        result = Y("s0")
        assert result == 7
        result = Y("s2")
        assert result == 9

    def test_call_rv_with_event_from_values(self, sample_space):
        values = pd.Series([4, 5, 6], index=sample_space.values, name="Z")
        Z = RandomVariable(values=values)
        event = Event(sample_space=sample_space, event_indices=["s0", "s2"])
        result = Z(event)
        assert isinstance(result, RandomVariable)
        expected_outputs = {"s0": 4, "s2": 6}
        assert result.domain == event.to_sample_space()
        assert result.name == "Z"
        assert result.outputs == expected_outputs


class TestEquality:
    def test_equality_same_domain_same_values(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        X1 = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        X2 = RandomVariable(domain=sample_space, outputs=outputs, name="Y")
        assert X1 == X2

    def test_equality_different_domains(self):
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        outputs1 = {"s0": 10, "s1": 20}
        outputs2 = {"a": 10, "b": 20}
        X1 = RandomVariable(domain=sample_space1, outputs=outputs1, name="X")
        X2 = RandomVariable(domain=sample_space2, outputs=outputs2, name="X")
        assert X1 != X2

    def test_equality_different_values(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs1 = {"s0": 10, "s1": 20, "s2": 30}
        outputs2 = {"s0": 10, "s1": 20, "s2": 40}
        X1 = RandomVariable(domain=sample_space, outputs=outputs1, name="X")
        X2 = RandomVariable(domain=sample_space, outputs=outputs2, name="X")
        assert X1 != X2

    def test_equality_with_non_random_variable(self):
        sample_space = SampleSpace(["s0", "s1"])
        outputs = {"s0": 10, "s1": 20}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert X != "not a random variable"
        assert X != 42
        assert X is not None

    def test_equality_with_values_construction(self):
        values1 = pd.Series([10, 20, 30], index=["s0", "s1", "s2"], name="X")
        values2 = pd.Series([10, 20, 30], index=["s0", "s1", "s2"], name="Y")
        X1 = RandomVariable(values=values1)
        X2 = RandomVariable(values=values2)
        assert X1 == X2

    def test_equality_mixed_construction_methods(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        values = pd.Series([10, 20, 30], index=["s0", "s1", "s2"], name="X")
        X1 = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        X2 = RandomVariable(values=values)
        assert X1 == X2


class TestArithmeticOperations:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2"])

    @pytest.fixture
    def X(self, sample_space):
        outputs_X = {"omega0": 1, "omega1": 2, "omega2": 3}
        return RandomVariable(domain=sample_space, outputs=outputs_X, name="X")

    @pytest.fixture
    def Y(self, sample_space):
        outputs_Y = {"omega0": 10, "omega1": 20, "omega2": 30}
        return RandomVariable(domain=sample_space, outputs=outputs_Y, name="Y")

    @pytest.fixture
    def probability_measure(self, sample_space):
        probabilities = {"omega0": 0.2, "omega1": 0.5, "omega2": 0.3}
        return ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities, name="P"
        )

    def test_random_variable_add_with_random_variable(
        self, sample_space, X, Y, probability_measure
    ):
        expected_outputs = {"omega0": 11, "omega1": 22, "omega2": 33}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X+Y"
        )
        Z = X + Y
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Y.add_probability_measure_to_domain(probability_measure)
        Z = X + Y
        assert Z.probability_measure is not None

    def test_random_variable_add_with_scalar(
        self, sample_space, X, probability_measure
    ):
        scalar = 5
        expected_outputs = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X+5"
        )
        Z = X + scalar
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Z = X + scalar
        assert Z.probability_measure is not None

    def test_random_variable_radd_with_scalar(self, sample_space, X):
        scalar = 5
        expected_outputs = {"omega0": 6, "omega1": 7, "omega2": 8}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="5+X"
        )
        Z = scalar + X
        assert Z == expected_rv

    def test_random_variable_mul_with_random_variable(
        self, sample_space, X, Y, probability_measure
    ):
        expected_outputs = {"omega0": 10, "omega1": 40, "omega2": 90}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X*Y"
        )
        Z = X * Y
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Y.add_probability_measure_to_domain(probability_measure)
        Z = X * Y
        assert Z.probability_measure is not None

    def test_random_variable_mul_with_scalar(
        self, sample_space, X, probability_measure
    ):
        scalar = 3
        expected_outputs = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X*3"
        )
        Z = X * scalar
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Z = X * scalar
        assert Z.probability_measure is not None

    def test_random_variable_rmul_with_scalar(self, sample_space, X):
        scalar = 3
        expected_outputs = {"omega0": 3, "omega1": 6, "omega2": 9}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="3*X"
        )
        Z = scalar * X
        assert Z == expected_rv

    def test_random_variable_sub_with_random_variable(
        self, sample_space, X, Y, probability_measure
    ):
        expected_outputs = {"omega0": -9, "omega1": -18, "omega2": -27}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X-Y"
        )
        Z = X - Y
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Y.add_probability_measure_to_domain(probability_measure)
        Z = X - Y
        assert Z.probability_measure is not None

    def test_random_variable_sub_with_scalar(
        self, sample_space, X, probability_measure
    ):
        scalar = 2
        expected_outputs = {"omega0": -1, "omega1": 0, "omega2": 1}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X-2"
        )
        Z = X - scalar
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Z = X - scalar
        assert Z.probability_measure is not None

    def test_random_variable_rsub_with_scalar(
        self, sample_space, X, probability_measure
    ):
        scalar = 2
        expected_outputs = {"omega0": 1, "omega1": 0, "omega2": -1}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="2-X"
        )
        Z = scalar - X
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Z = scalar - X
        assert Z.probability_measure is not None

    def test_random_variable_truediv_with_random_variable(
        self, sample_space, X, Y, probability_measure
    ):
        expected_outputs = {"omega0": 0.1, "omega1": 0.1, "omega2": 0.1}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X/Y"
        )
        Z = X / Y
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Y.add_probability_measure_to_domain(probability_measure)
        Z = X / Y
        assert Z.probability_measure is not None

    def test_random_variable_truediv_with_scalar(
        self, sample_space, X, probability_measure
    ):
        scalar = 2
        expected_outputs = {"omega0": 0.5, "omega1": 1.0, "omega2": 1.5}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X/2"
        )
        Z = X / scalar
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Z = X / scalar
        assert Z.probability_measure is not None

    def test_random_variable_rtruediv_with_scalar(self, sample_space, X):
        scalar = 60
        expected_outputs = {"omega0": 60.0, "omega1": 30.0, "omega2": 20.0}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="60/X"
        )
        Z = scalar / X
        assert Z == expected_rv

    def test_random_variable_pow_with_random_variable(
        self, sample_space, X, Y, probability_measure
    ):
        expected_outputs = {"omega0": 1**10, "omega1": 2**20, "omega2": 3**30}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X**Y"
        )
        Z = X**Y
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Y.add_probability_measure_to_domain(probability_measure)
        Z = X**Y
        assert Z.probability_measure is not None

    def test_random_variable_pow_with_scalar(
        self, sample_space, X, probability_measure
    ):
        scalar = 2
        expected_outputs = {"omega0": 1**2, "omega1": 2**2, "omega2": 3**2}
        expected_rv = RandomVariable(
            domain=sample_space, outputs=expected_outputs, name="X**2"
        )
        Z = X**scalar
        assert Z == expected_rv
        X.add_probability_measure_to_domain(probability_measure)
        Z = X**scalar
        assert Z.probability_measure is not None

    def test_operations_preserve_probability_space(self):
        sample_space = SampleSpace(["s0", "s1"])
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities={"s0": 0.4, "s1": 0.6}
        )
        X = RandomVariable.on_probability_space(
            outputs={"s0": 1, "s1": 2}, probability_space=prob_space, name="X"
        )
        Y = X + 5
        assert Y.probability_space == prob_space
        Z = X * 2
        assert Z.probability_space == prob_space
        W = X**2
        assert W.probability_space == prob_space

    def test_operations_with_mismatched_domains_raise_error(self):
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        X = RandomVariable(domain=sample_space1, outputs={"s0": 1, "s1": 2}, name="X")
        Y = RandomVariable(domain=sample_space2, outputs={"a": 3, "b": 4}, name="Y")
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

    def test_arithmetic_add_with_values_construction(self, sample_space):
        values_X = pd.Series([1, 2, 3], index=sample_space.values, name="X")
        values_Y = pd.Series([10, 20, 30], index=sample_space.values, name="Y")
        X = RandomVariable(values=values_X)
        Y = RandomVariable(values=values_Y)
        Z = X + Y
        expected_outputs = {"omega0": 11, "omega1": 22, "omega2": 33}
        assert Z.outputs == expected_outputs

    def test_arithmetic_mul_with_values_construction(self, sample_space):
        values_X = pd.Series([1, 2, 3], index=sample_space.values, name="X")
        X = RandomVariable(values=values_X)
        Z = X * 5
        expected_outputs = {"omega0": 5, "omega1": 10, "omega2": 15}
        assert Z.outputs == expected_outputs

    def test_arithmetic_sub_with_values_construction(self, sample_space):
        values_X = pd.Series([10, 20, 30], index=sample_space.values, name="X")
        X = RandomVariable(values=values_X)
        Z = X - 5
        expected_outputs = {"omega0": 5, "omega1": 15, "omega2": 25}
        assert Z.outputs == expected_outputs

    def test_arithmetic_div_with_values_construction(self, sample_space):
        values_X = pd.Series([10, 20, 30], index=sample_space.values, name="X")
        X = RandomVariable(values=values_X)
        Z = X / 10
        expected_outputs = {"omega0": 1.0, "omega1": 2.0, "omega2": 3.0}
        assert Z.outputs == expected_outputs

    def test_arithmetic_pow_with_values_construction(self, sample_space):
        values_X = pd.Series([1, 2, 3], index=sample_space.values, name="X")
        X = RandomVariable(values=values_X)
        Z = X**2
        expected_outputs = {"omega0": 1, "omega1": 4, "omega2": 9}
        assert Z.outputs == expected_outputs


class TestProbabilityMethods:
    @pytest.fixture
    def feature_embedding(self):
        import itertools

        sequences = list(itertools.product([0, 1], repeat=3))
        values = pd.DataFrame(sequences)
        return FeatureEmbedding(values=values, name="X")

    @pytest.fixture
    def fps(self, feature_embedding):
        def pmf(sample_features: SamplePointFeatures) -> float:
            num_ones = sample_features.sum()
            return 0.25**num_ones * 0.75 ** (3 - num_ones)

        return feature_embedding.add_probability_measure_from_features(pmf=pmf)

    @pytest.fixture
    def X_function(self):
        def function(sample_features: SamplePointFeatures) -> int:
            return sample_features.sum()

        return function

    def test_construction_on_prob_space(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        rv_range = X.range
        assert isinstance(rv_range, FeaturizedProbabilitySpace)
        expected_probabilities = {
            "x0": 0.75**3,
            "x1": 3 * 0.25 * 0.75**2,
            "x2": 3 * 0.25**2 * 0.75,
            "x3": 0.25**3,
        }
        actual_probabilities = {idx: rv_range.P(idx) for idx in rv_range.sample_space}
        for idx in expected_probabilities:
            assert abs(actual_probabilities[idx] - expected_probabilities[idx]) < 1e-10

    def test_P_method(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        assert abs(X.P(0) - 0.75**3) < 1e-10
        assert abs(X.P(1) - 3 * 0.25 * 0.75**2) < 1e-10
        assert abs(X.P(3) - 0.25**3) < 1e-10

    def test_P_method_without_probability_space_raises_error(self):
        sample_space = SampleSpace(["s0", "s1"])
        X = RandomVariable(domain=sample_space, outputs={"s0": 1, "s1": 2}, name="X")
        with pytest.raises(
            ValueError, match="does not have an associated ProbabilityMeasure"
        ):
            X.P(1)

    def test_P_method_with_invalid_value_raises_error(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        with pytest.raises(ValueError, match="not in the range"):
            X.P(99)

    def test_induced_probability_measure_sums_to_one(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        total_prob = sum(X.P(value) for value in [0, 1, 2, 3])
        assert abs(total_prob - 1.0) < 1e-10

    def test_unconditional_expectation(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        expected_expectation = (
            0 * 0.75**3
            + 1 * (3 * 0.25 * 0.75**2)
            + 2 * (3 * 0.25**2 * 0.75)
            + 3 * 0.25**3
        )
        actual_expectation = expectation(X)
        assert abs(actual_expectation - expected_expectation) < 1e-10

    def test_unconditional_expectation_without_probability_space_raises_error(self):
        sample_space = SampleSpace(["s0", "s1"])
        X = RandomVariable(domain=sample_space, outputs={"s0": 1, "s1": 2}, name="X")
        with pytest.raises(ValueError, match="must have a probability_space"):
            expectation(X)

    def test_expectation(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        atom_ids = dict(zip(X.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=X.probability_space.sample_space
        )
        sigma_algebra.probability_space = X.probability_space
        exp = expectation(rv=X, sigma_algebra=sigma_algebra)
        assert isinstance(exp, RandomVariable)
        assert exp.name == "E(X|F)"
        expected_outputs = {
            0: (0 * 0.75**3 + 1 * 0.25 * 0.75**2) / (0.75**3 + 0.25 * 0.75**2),
            1: (0 * 0.75**3 + 1 * 0.25 * 0.75**2) / (0.75**3 + 0.25 * 0.75**2),
            2: (1 * 0.25 * 0.75**2 + 2 * 0.25**2 * 0.75 + 1 * 0.25 * 0.75**2)
            / (0.25 * 0.75**2 + 0.25**2 * 0.75 + 0.25 * 0.75**2),
            3: (1 * 0.25 * 0.75**2 + 2 * 0.25**2 * 0.75 + 1 * 0.25 * 0.75**2)
            / (0.25 * 0.75**2 + 0.25**2 * 0.75 + 0.25 * 0.75**2),
            4: (1 * 0.25 * 0.75**2 + 2 * 0.25**2 * 0.75 + 1 * 0.25 * 0.75**2)
            / (0.25 * 0.75**2 + 0.25**2 * 0.75 + 0.25 * 0.75**2),
            5: 2 * 0.25**2 * 0.75 / (0.25**2 * 0.75),
            6: (2 * 0.25**2 * 0.75 + 3 * 0.25**3) / (0.25**2 * 0.75 + 0.25**3),
            7: (2 * 0.25**2 * 0.75 + 3 * 0.25**3) / (0.25**2 * 0.75 + 0.25**3),
        }
        for sample_id in exp.domain:
            assert (
                abs(exp.outputs[sample_id] - expected_outputs[sample_id])
                < 1e-10
            )

    def test_expectation_without_sigma_algebra_returns_float(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        result = expectation(X)
        assert isinstance(result, float)
        expected = (
            0 * 0.75**3
            + 1 * (3 * 0.25 * 0.75**2)
            + 2 * (3 * 0.25**2 * 0.75)
            + 3 * 0.25**3
        )
        assert abs(result - expected) < 1e-10

    def test_conditional_expectation_preserves_probability_space(self, fps, X_function):
        X = RandomVariable.from_features(
            function=X_function, feature_embedding=fps.feature_embedding
        )
        X.add_probability_measure_to_domain(fps.probability_measure)
        atom_ids = dict(zip(X.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=X.probability_space.sample_space
        )
        sigma_algebra.probability_space = X.probability_space
        exp = expectation(rv=X, sigma_algebra=sigma_algebra)
        print(X.probability_space)
        assert exp.probability_space == X.probability_space


class TestEdgeCases:
    def test_constant_random_variable(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": 5, "s1": 5, "s2": 5}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert len(X.range.domain) == 1
        sigma_alg = X.sigma_algebra
        events = sigma_alg.atom_id_to_event
        assert len(events) == 1
        assert len(list(events.values())[0].values) == 3

    def test_indicator_random_variable(self):
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        outputs = {"s0": 1, "s1": 0, "s2": 1, "s3": 0}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert len(X.range.domain) == 2
        assert X("s0") == 1
        assert X("s1") == 0

    def test_with_numeric_sample_indices(self):
        sample_space = SampleSpace([0, 1, 2])
        outputs = {0: "a", 1: "b", 2: "c"}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert X(0) == "a"
        assert X(1) == "b"
        assert X(2) == "c"

    def test_with_string_outputs(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        outputs = {"s0": "red", "s1": "green", "s2": "blue"}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="Color")
        assert X("s0") == "red"
        assert X("s1") == "green"

    def test_with_tuple_outputs(self):
        sample_space = SampleSpace(["s0", "s1"])
        outputs = {"s0": (1, 2), "s1": (3, 4)}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert X("s0") == (1, 2)
        assert X("s1") == (3, 4)

    def test_large_sample_space(self):
        n = 1000
        sample_space = SampleSpace([f"s{i}" for i in range(n)])
        outputs = {f"s{i}": i for i in range(n)}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        assert len(X.domain) == n
        assert X(f"s{500}") == 500

    def test_constant_random_variable_with_values(self):
        values = pd.Series([5, 5, 5], index=["s0", "s1", "s2"], name="X")
        X = RandomVariable(values=values)
        assert len(X.range.domain) == 1
        sigma_alg = X.sigma_algebra
        events = sigma_alg.atom_id_to_event
        assert len(events) == 1
        assert len(list(events.values())[0].values) == 3

    def test_with_string_outputs_values_construction(self):
        values = pd.Series(
            ["red", "green", "blue"], index=["s0", "s1", "s2"], name="Color"
        )
        X = RandomVariable(values=values)
        assert X("s0") == "red"
        assert X("s1") == "green"

    def test_with_tuple_outputs_values_construction(self):
        values = pd.Series([(1, 2), (3, 4)], index=["s0", "s1"], name="X")
        X = RandomVariable(values=values)
        assert X("s0") == (1, 2)
        assert X("s1") == (3, 4)

    def test_values_preserves_series_name(self):
        values = pd.Series([1, 2, 3], index=["s0", "s1", "s2"], name="CustomName")
        X = RandomVariable(values=values)
        assert X.name == "CustomName"

    def test_values_uses_default_name_if_series_unnamed(self):
        values = pd.Series([1, 2, 3], index=["s0", "s1", "s2"])
        X = RandomVariable(values=values, name="Y")
        assert X.name == "Y"


class TestIsMeasurable:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    @pytest.fixture
    def prob_space(self, sample_space):
        probabilities = {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25}
        return ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )

    def test_rv_is_measurable_wrt_its_own_sigma_algebra(self, prob_space):
        outputs = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=outputs, probability_space=prob_space, name="X"
        )
        assert X.is_measurable(X.sigma_algebra)

    def test_rv_is_measurable_wrt_power_set(self, prob_space):
        outputs = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=outputs, probability_space=prob_space, name="X"
        )
        power_set = SigmaAlgebra.power_set(sample_space=prob_space.sample_space)
        assert X.is_measurable(power_set)

    def test_rv_not_measurable_wrt_trivial(self, prob_space):
        outputs = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=outputs, probability_space=prob_space, name="X"
        )
        trivial = SigmaAlgebra.trivial(sample_space=prob_space.sample_space)
        assert not X.is_measurable(trivial)

    def test_constant_rv_measurable_wrt_trivial(self, prob_space):
        outputs = {"s0": 5, "s1": 5, "s2": 5, "s3": 5}
        X = RandomVariable.on_probability_space(
            outputs=outputs, probability_space=prob_space, name="X"
        )
        trivial = SigmaAlgebra.trivial(sample_space=prob_space.sample_space)
        assert X.is_measurable(trivial)

    def test_rv_measurable_wrt_coarser_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = SigmaAlgebra(
            sample_space=prob_space.sample_space, sample_id_to_atom_id=coarse_atom_ids
        )
        assert not X.is_measurable(coarse)

    def test_rv_not_measurable_wrt_finer_incompatible_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        incompatible_atom_ids = {"s0": 0, "s1": 1, "s2": 0, "s3": 2}
        incompatible = SigmaAlgebra(
            sample_space=prob_space.sample_space,
            sample_id_to_atom_id=incompatible_atom_ids,
        )
        assert not X.is_measurable(incompatible)

    def test_rv_measurable_wrt_finer_compatible_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        power_set = SigmaAlgebra.power_set(sample_space=prob_space.sample_space)
        assert X.is_measurable(power_set)

    def test_is_measurable_uses_probability_space_sigma_algebra_by_default(
        self, prob_space
    ):
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        custom_sigma = SigmaAlgebra(
            sample_space=prob_space.sample_space, sample_id_to_atom_id=coarse_atom_ids
        )
        prob_space._sigma_algebra = custom_sigma
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        assert X.is_measurable()

    def test_finer_rv_measurable_wrt_coarser_rv_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        Y_outputs = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        Y = RandomVariable.on_probability_space(
            outputs=Y_outputs, probability_space=prob_space, name="Y"
        )
        assert not Y.is_measurable(X.sigma_algebra)

    def test_coarser_rv_measurable_wrt_finer_rv_sigma_algebra(self, prob_space):
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        Y_outputs = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        Y = RandomVariable.on_probability_space(
            outputs=Y_outputs, probability_space=prob_space, name="Y"
        )
        assert X.is_measurable(Y.sigma_algebra)

    def test_function_of_measurable_rv_is_measurable(self, prob_space):
        X_outputs = {"s0": 1, "s1": 1, "s2": 2, "s3": 2}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        Y_outputs = {"s0": 2, "s1": 2, "s2": 4, "s3": 4}
        Y = RandomVariable.on_probability_space(
            outputs=Y_outputs, probability_space=prob_space, name="Y"
        )
        assert Y.sigma_algebra == X.sigma_algebra
        assert Y.is_measurable(X.sigma_algebra)

    def test_sum_of_measurable_rvs_is_measurable(self, prob_space):
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        coarse = SigmaAlgebra(
            sample_space=prob_space.sample_space, sample_id_to_atom_id=coarse_atom_ids
        )
        X_outputs = {"s0": 1, "s1": 1, "s2": 2, "s3": 2}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        Y_outputs = {"s0": 10, "s1": 10, "s2": 20, "s3": 20}
        Y = RandomVariable.on_probability_space(
            outputs=Y_outputs, probability_space=prob_space, name="Y"
        )
        Z = X + Y
        assert Z.is_measurable(coarse)

    def test_measurability_with_three_rvs(self, prob_space):
        C_outputs = {"s0": 5, "s1": 5, "s2": 5, "s3": 5}
        C = RandomVariable.on_probability_space(
            outputs=C_outputs, probability_space=prob_space, name="C"
        )
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        Y_outputs = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        Y = RandomVariable.on_probability_space(
            outputs=Y_outputs, probability_space=prob_space, name="Y"
        )
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
        sample_space = SampleSpace(["s0", "s1"])
        outputs = {"s0": 0, "s1": 1}
        X = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        with pytest.raises(ValueError):
            X.is_measurable()

    def test_measurability_respects_sigma_algebra_order(self, prob_space):
        trivial = SigmaAlgebra.trivial(sample_space=prob_space.sample_space)
        middle_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = SigmaAlgebra(
            sample_space=prob_space.sample_space, sample_id_to_atom_id=middle_atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space=prob_space.sample_space)
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = RandomVariable.on_probability_space(
            outputs=X_outputs, probability_space=prob_space, name="X"
        )
        assert not X.is_measurable(trivial)
        assert X.is_measurable(middle)
        assert X.is_measurable(power_set)
