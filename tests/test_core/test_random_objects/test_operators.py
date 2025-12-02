import pytest

import sigalg as sa


class TestUnconditionalExpectation:

    @pytest.fixture
    def numeric_rv(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        return sa.RandomVariable(
            probability_space=prob_space, outputs=outputs, name="X"
        )

    @pytest.fixture
    def string_rv(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "red", "s1": "green", "s2": "blue"}
        return sa.RandomVariable(
            probability_space=prob_space, outputs=outputs, name="Color"
        )

    @pytest.fixture
    def tuple_rv(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.4, "s1": 0.6}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": (1, 2), "s1": (3, 4)}
        return sa.RandomVariable(
            probability_space=prob_space, outputs=outputs, name="Tuple"
        )

    def test_unconditional_expectation_with_numeric_rv(self, numeric_rv):
        expected = 10 * 0.2 + 20 * 0.3 + 30 * 0.5
        actual = sa.unconditional_expectation(numeric_rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_without_probability_space(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        outputs = {"s0": 1, "s1": 2}
        rv = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        with pytest.raises(ValueError, match="must have a probability_space"):
            sa.unconditional_expectation(rv)

    def test_unconditional_expectation_with_string_rv(self, string_rv):
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.unconditional_expectation(string_rv)

    def test_unconditional_expectation_with_tuple_rv(self, tuple_rv):
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.unconditional_expectation(tuple_rv)

    def test_unconditional_expectation_with_mixed_numeric_string(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.3, "s1": 0.3, "s2": 0.4}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": "invalid"}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.unconditional_expectation(rv)

    def test_unconditional_expectation_with_integer_rv(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.25, "s1": 0.25, "s2": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2, "s2": 3}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        expected = 1 * 0.25 + 2 * 0.25 + 3 * 0.5
        actual = sa.unconditional_expectation(rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_with_float_rv(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.6, "s1": 0.4}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1.5, "s1": 2.5}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        expected = 1.5 * 0.6 + 2.5 * 0.4
        actual = sa.unconditional_expectation(rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_with_negative_values(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.3, "s1": 0.4, "s2": 0.3}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": -10, "s1": 0, "s2": 10}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        expected = -10 * 0.3 + 0 * 0.4 + 10 * 0.3
        actual = sa.unconditional_expectation(rv)
        assert abs(actual - expected) < 1e-10

    def test_unconditional_expectation_with_zero_values(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 0, "s1": 0}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        actual = sa.unconditional_expectation(rv)
        assert abs(actual - 0.0) < 1e-10


class TestExpectation:

    @pytest.fixture
    def simple_rv(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2, "s2": 3, "s3": 4}
        return sa.RandomVariable(
            probability_space=prob_space, outputs=outputs, name="X"
        )

    def test_expectation_without_conditioning(self, simple_rv):
        result = sa.expectation(simple_rv)
        expected = 1 * 0.1 + 2 * 0.2 + 3 * 0.3 + 4 * 0.4
        assert abs(result - expected) < 1e-10

    def test_expectation_with_trivial_sigma_algebra(self, simple_rv):
        sigma_algebra = sa.SigmaAlgebra.trivial(
            probability_space=simple_rv.probability_space
        )
        result = sa.expectation(simple_rv, sigma_algebra)
        assert isinstance(result, sa.RandomVariable)
        expected_value = 1 * 0.1 + 2 * 0.2 + 3 * 0.3 + 4 * 0.4
        for sample_id in result.domain.values:
            assert abs(result.outputs[sample_id] - expected_value) < 1e-10

    def test_expectation_with_power_set_sigma_algebra(self, simple_rv):
        sigma_algebra = sa.SigmaAlgebra.power_set(
            probability_space=simple_rv.probability_space
        )
        result = sa.expectation(simple_rv, sigma_algebra)
        assert isinstance(result, sa.RandomVariable)
        for sample_id in result.domain.values:
            assert abs(result.outputs[sample_id] - simple_rv.outputs[sample_id]) < 1e-10

    def test_expectation_with_custom_partition(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30, "s3": 40}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma_algebra = sa.SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id, probability_space=prob_space
        )
        result = sa.expectation(rv, sigma_algebra)
        assert isinstance(result, sa.RandomVariable)
        e1 = (10 * 0.1 + 20 * 0.2) / (0.1 + 0.2)
        e2 = (30 * 0.3 + 40 * 0.4) / (0.3 + 0.4)
        assert abs(result.outputs["s0"] - e1) < 1e-10
        assert abs(result.outputs["s1"] - e1) < 1e-10
        assert abs(result.outputs["s2"] - e2) < 1e-10
        assert abs(result.outputs["s3"] - e2) < 1e-10

    def test_expectation_preserves_name_format(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.4, "s1": 0.6}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 1, "s1": 2}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="Y")
        sample_id_to_atom_id = {"s0": 0, "s1": 1}
        sigma_algebra = sa.SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id,
            probability_space=prob_space,
            name="G",
        )
        result = sa.expectation(rv, sigma_algebra)
        assert result.name == "E(Y|G)"

    def test_expectation_without_probability_space_fails(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        outputs = {"s0": 1, "s1": 2}
        rv = sa.RandomVariable(domain=sample_space, outputs=outputs, name="X")
        with pytest.raises(ValueError, match="must have a probability_space"):
            sa.expectation(rv)

    def test_expectation_with_non_numeric_values_fails(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "red", "s1": "blue"}
        rv = sa.RandomVariable(
            probability_space=prob_space, outputs=outputs, name="Color"
        )
        sigma_algebra = sa.SigmaAlgebra.trivial(sample_space)
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.expectation(rv, sigma_algebra)

    def test_expectation_with_unequal_partition_probabilities(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.5, "s1": 0.3, "s2": 0.2}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 100, "s1": 200, "s2": 300}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1}
        sigma_algebra = sa.SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id, probability_space=prob_space
        )
        result = sa.expectation(rv, sigma_algebra)
        e1 = (100 * 0.5 + 200 * 0.3) / (0.5 + 0.3)
        e2 = 300
        assert abs(result.outputs["s0"] - e1) < 1e-10
        assert abs(result.outputs["s1"] - e1) < 1e-10
        assert abs(result.outputs["s2"] - e2) < 1e-10


class TestEdgeCases:

    def test_expectation_with_boolean_values_fails(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": True, "s1": False}
        rv = sa.RandomVariable(
            probability_space=prob_space, outputs=outputs, name="Bool"
        )
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.unconditional_expectation(rv)

    def test_expectation_with_string_values_fails(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": "hello", "s1": "world"}
        rv = sa.RandomVariable(
            probability_space=prob_space, outputs=outputs, name="String"
        )
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.unconditional_expectation(rv)

    def test_expectation_with_list_values_fails(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        probabilities = {"s0": 0.5, "s1": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        with pytest.raises(TypeError):
            outputs = {"s0": [1, 2], "s1": [3, 4]}
            _ = sa.RandomVariable(
                probability_space=prob_space, outputs=outputs, name="List"
            )
