import pytest

from sigalg.core import ProbabilitySpace, RandomVariable, SampleSpace, SigmaAlgebra
from sigalg.l2 import expectation


class TestUnconditionalExpectation:

    @pytest.fixture
    def numeric_rv(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        return RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )

    @pytest.fixture
    def string_rv(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "red", "s1": "green", "s2": "blue"}
        return RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="Color"
        )

    @pytest.fixture
    def tuple_rv(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.4, "s1": 0.6}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": (1, 2), "s1": (3, 4)}
        return RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="Tuple"
        )

    def test_unconditional_expectation_with_numeric_rv(self, numeric_rv):
        expected = 10 * 0.2 + 20 * 0.3 + 30 * 0.5
        actual = expectation(numeric_rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_without_probability_space(self):
        sample_space = SampleSpace(["s0", "s1"])
        outputs = {"s0": 1, "s1": 2}
        rv = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        with pytest.raises(ValueError, match="must have a probability_space"):
            expectation(rv)

    def test_unconditional_expectation_with_string_rv(self, string_rv):
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(string_rv)

    def test_unconditional_expectation_with_tuple_rv(self, tuple_rv):
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(tuple_rv)

    def test_unconditional_expectation_with_integer_rv(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.25, "s1": 0.25, "s2": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2, "s2": 3}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        expected = 1 * 0.25 + 2 * 0.25 + 3 * 0.5
        actual = expectation(rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_with_float_rv(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.6, "s1": 0.4}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1.5, "s1": 2.5}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        expected = 1.5 * 0.6 + 2.5 * 0.4
        actual = expectation(rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_with_negative_values(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.3, "s1": 0.4, "s2": 0.3}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": -10, "s1": 0, "s2": 10}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        expected = -10 * 0.3 + 0 * 0.4 + 10 * 0.3
        actual = expectation(rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_with_zero_values(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 0, "s1": 0}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        actual = expectation(rv)
        assert abs(actual - 0.0) < 1e-10


class TestExpectation:

    @pytest.fixture
    def simple_rv(self):
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2, "s2": 3, "s3": 4}
        return RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )

    def test_expectation_without_conditioning(self, simple_rv):
        result = expectation(simple_rv)
        expected = 1 * 0.1 + 2 * 0.2 + 3 * 0.3 + 4 * 0.4
        assert abs(result - expected) < 1e-10

    def test_expectation_with_trivial_sigma_algebra(self, simple_rv):
        sigma_algebra = SigmaAlgebra.trivial(sample_space=simple_rv.domain)
        result = expectation(simple_rv, sigma_algebra)
        assert isinstance(result, RandomVariable)
        expected_value = 1 * 0.1 + 2 * 0.2 + 3 * 0.3 + 4 * 0.4
        for sample_id in result.domain.data:
            assert abs(result.outputs[sample_id] - expected_value) < 1e-10

    def test_expectation_with_power_set_sigma_algebra(self, simple_rv):
        sigma_algebra = SigmaAlgebra.power_set(sample_space=simple_rv.domain)
        result = expectation(simple_rv, sigma_algebra)
        assert isinstance(result, RandomVariable)
        for sample_id in result.domain.data:
            assert abs(result.outputs[sample_id] - simple_rv.outputs[sample_id]) < 1e-10

    def test_expectation_with_custom_partition(self):
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30, "s3": 40}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id, sample_space=sample_space
        )
        result = expectation(rv, sigma_algebra)
        assert isinstance(result, RandomVariable)
        e1 = (10 * 0.1 + 20 * 0.2) / (0.1 + 0.2)
        e2 = (30 * 0.3 + 40 * 0.4) / (0.3 + 0.4)
        assert abs(result.outputs["s0"] - e1) < 1e-10
        assert abs(result.outputs["s1"] - e1) < 1e-10
        assert abs(result.outputs["s2"] - e2) < 1e-10
        assert abs(result.outputs["s3"] - e2) < 1e-10

    def test_expectation_preserves_name_format(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.4, "s1": 0.6}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="Y"
        )
        sample_id_to_atom_id = {"s0": 0, "s1": 1}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            name="G",
        )
        result = expectation(rv, sigma_algebra)
        assert result.name == "E(Y|G)"

    def test_expectation_without_probability_space_fails(self):
        sample_space = SampleSpace(["s0", "s1"])
        outputs = {"s0": 1, "s1": 2}
        rv = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        with pytest.raises(ValueError, match="must have a probability_space"):
            expectation(rv)

    def test_expectation_with_non_numeric_values_fails(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "red", "s1": "blue"}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="Color"
        )
        sigma_algebra = SigmaAlgebra.trivial(sample_space)
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(rv, sigma_algebra)

    def test_expectation_with_unequal_partition_probabilities(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.5, "s1": 0.3, "s2": 0.2}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 100, "s1": 200, "s2": 300}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id, sample_space=sample_space
        )
        result = expectation(rv, sigma_algebra)
        e1 = (100 * 0.5 + 200 * 0.3) / (0.5 + 0.3)
        e2 = 300
        assert abs(result.outputs["s0"] - e1) < 1e-10
        assert abs(result.outputs["s1"] - e1) < 1e-10
        assert abs(result.outputs["s2"] - e2) < 1e-10


class TestValidation:

    def test_expectation_with_invalid_rv_type(self):
        with pytest.raises(TypeError, match="rv must be a RandomVariable"):
            expectation("not a random variable")

    def test_expectation_with_invalid_rv_type_int(self):
        with pytest.raises(TypeError, match="rv must be a RandomVariable"):
            expectation(42)

    def test_expectation_with_invalid_rv_type_dict(self):
        with pytest.raises(TypeError, match="rv must be a RandomVariable"):
            expectation({"s0": 1, "s1": 2})

    def test_expectation_with_invalid_sigma_algebra_type(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            expectation(rv, sigma_algebra="not a sigma algebra")

    def test_expectation_with_invalid_sigma_algebra_type_int(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            expectation(rv, sigma_algebra=123)

    def test_expectation_without_probability_space(self):
        sample_space = SampleSpace(["s0", "s1"])
        outputs = {"s0": 1, "s1": 2}
        rv = RandomVariable(domain=sample_space, outputs=outputs, name="X")
        with pytest.raises(ValueError, match="must have a probability_space"):
            expectation(rv)

    def test_expectation_with_mismatched_sigma_algebra_sample_space(self):
        sample_space1 = SampleSpace(["s0", "s1"])
        probabilities1 = {"s0": 0.5, "s1": 0.5}
        prob_space1 = ProbabilitySpace.from_probabilities(
            sample_space=sample_space1, probabilities=probabilities1
        )
        outputs = {"s0": 1, "s1": 2}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space1, outputs=outputs, name="X"
        )
        sample_space2 = SampleSpace(["s2", "s3"])
        probabilities2 = {"s2": 0.5, "s3": 0.5}
        prob_space2 = ProbabilitySpace.from_probabilities(
            sample_space=sample_space2, probabilities=probabilities2
        )
        sigma_algebra = SigmaAlgebra.trivial(sample_space=prob_space2.sample_space)
        with pytest.raises(ValueError, match="sample_space must match"):
            expectation(rv, sigma_algebra)

    def test_expectation_with_non_numeric_values(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "red", "s1": "blue"}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(rv)

    def test_expectation_conditional_with_non_numeric_values(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "red", "s1": "blue"}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        sigma_algebra = SigmaAlgebra.trivial(sample_space=prob_space.sample_space)
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(rv, sigma_algebra)

    def test_expectation_with_boolean_values(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": True, "s1": False}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="X"
        )
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(rv)


class TestEdgeCases:

    def test_expectation_with_boolean_values_fails(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": True, "s1": False}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="Bool"
        )
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(rv)

    def test_expectation_with_string_values_fails(self):
        sample_space = SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "hello", "s1": "world"}
        rv = RandomVariable.on_probability_space(
            probability_space=prob_space, outputs=outputs, name="String"
        )
        with pytest.raises(TypeError, match="non-numeric values"):
            expectation(rv)
