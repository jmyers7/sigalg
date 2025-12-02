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
    def numeric_rv(self, fps):
        def function(sample_features: sa.SamplePointFeatures) -> int:
            return sample_features.sum()

        return sa.RandomVariable.from_features(fps=fps, function=function, name="X")

    @pytest.fixture
    def string_rv(self, fps):
        def function(sample_features: sa.SamplePointFeatures) -> str:
            num_ones = sample_features.sum()
            return f"count_{num_ones}"

        return sa.RandomVariable.from_features(fps=fps, function=function, name="Label")

    def test_expectation_without_sigma_algebra(self, numeric_rv):
        result = sa.expectation(numeric_rv)
        assert isinstance(result, float)
        expected = (
            0 * 0.75**3
            + 1 * (3 * 0.25 * 0.75**2)
            + 2 * (3 * 0.25**2 * 0.75)
            + 3 * 0.25**3
        )
        assert abs(result - expected) < 1e-10

    def test_expectation_with_sigma_algebra(self, numeric_rv):
        atom_ids = dict(zip(numeric_rv.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = sa.SigmaAlgebra(
            probability_space=numeric_rv.probability_space,
            sample_id_to_atom_id=atom_ids,
        )
        result = sa.expectation(rv=numeric_rv, sigma_algebra=sigma_algebra)
        assert isinstance(result, sa.RandomVariable)
        assert result.name == "E(X|F)"
        assert result.probability_space == numeric_rv.probability_space

    def test_expectation_without_sigma_algebra_with_string_rv(self, string_rv):
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.expectation(string_rv)

    def test_expectation_with_sigma_algebra_with_string_rv(self, string_rv):
        atom_ids = dict(zip(string_rv.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = sa.SigmaAlgebra(
            probability_space=string_rv.probability_space, sample_id_to_atom_id=atom_ids
        )
        with pytest.raises(TypeError, match="non-numeric values"):
            sa.expectation(rv=string_rv, sigma_algebra=sigma_algebra)

    def test_expectation_preserves_probability_space(self, numeric_rv):
        atom_ids = dict(zip(numeric_rv.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = sa.SigmaAlgebra(
            probability_space=numeric_rv.probability_space,
            sample_id_to_atom_id=atom_ids,
        )
        result = sa.expectation(rv=numeric_rv, sigma_algebra=sigma_algebra)
        assert result.probability_space == numeric_rv.probability_space

    def test_expectation_with_trivial_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        trivial_sigma = sa.SigmaAlgebra.trivial(probability_space=prob_space)
        result = sa.expectation(rv=rv, sigma_algebra=trivial_sigma)
        unconditional_exp = sa.unconditional_expectation(rv)
        for sample_id in rv.domain.values:
            assert abs(result(sample_id) - unconditional_exp) < 1e-10

    def test_expectation_with_power_set_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        rv = sa.RandomVariable(probability_space=prob_space, outputs=outputs, name="X")
        power_set_sigma = sa.SigmaAlgebra.power_set(probability_space=prob_space)
        result = sa.expectation(rv=rv, sigma_algebra=power_set_sigma)
        for sample_id in rv.domain.values:
            assert abs(result(sample_id) - rv(sample_id)) < 1e-10

    def test_expectation_returns_measurable_random_variable(self, numeric_rv):
        atom_ids = dict(zip(numeric_rv.domain, [0, 0, 1, 1, 1, 2, 3, 3]))
        sigma_algebra = sa.SigmaAlgebra(
            probability_space=numeric_rv.probability_space,
            sample_id_to_atom_id=atom_ids,
        )
        E = sa.expectation(rv=numeric_rv, sigma_algebra=sigma_algebra)
        sigma_algebra_E = E.sigma_algebra
        assert sa.is_sub_algebra(sub=sigma_algebra_E, super=sigma_algebra)


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
