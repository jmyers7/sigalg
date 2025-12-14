import pytest

from sigalg.core import (
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    SampleSpace,
    SigmaAlgebra,
)
from sigalg.l2 import L2


class TestConstructor:

    def test_constructor(self):
        sample_space = SampleSpace(["omega0", "omega1"])
        prob_space = ProbabilitySpace(sample_space=sample_space)
        L2_space = L2(prob_space)
        assert L2_space.probability_space == prob_space


class TestProperties:

    def test_properties(self):
        sample_space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id={"omega0": 0, "omega1": 1, "omega2": 1},
            sample_space=sample_space,
        )
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space,
            probabilities={"omega0": 0.2, "omega1": 0.5, "omega2": 0.3},
        )
        prob_space = ProbabilitySpace(
            sample_space=sample_space,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )
        L2_space = L2(prob_space)
        assert L2_space.probability_space == prob_space
        assert L2_space.sample_space == sample_space
        assert L2_space.sigma_algebra == sigma_algebra
        assert L2_space.probability_measure == prob_measure


class TestSetters:

    def test_setters(self):
        sample_space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma_algebra1 = SigmaAlgebra(
            sample_id_to_atom_id={"omega0": 0, "omega1": 1, "omega2": 1},
            sample_space=sample_space,
        )
        sigma_algebra2 = SigmaAlgebra(
            sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
            sample_space=sample_space,
        )
        prob_measure1 = ProbabilityMeasure(
            sample_space=sample_space,
            probabilities={"omega0": 0.2, "omega1": 0.5, "omega2": 0.3},
        )
        prob_measure2 = ProbabilityMeasure(
            sample_space=sample_space,
            probabilities={"omega0": 0.3, "omega1": 0.4, "omega2": 0.3},
        )
        prob_space1 = ProbabilitySpace(
            sample_space=sample_space,
            sigma_algebra=sigma_algebra1,
            probability_measure=prob_measure1,
        )
        prob_space2 = ProbabilitySpace(
            sample_space=sample_space,
            sigma_algebra=sigma_algebra2,
            probability_measure=prob_measure2,
        )
        L2_space = L2(prob_space1)
        assert L2_space.probability_space == prob_space1
        assert L2_space.sigma_algebra == sigma_algebra1
        assert L2_space.probability_measure == prob_measure1
        L2_space.sigma_algebra = sigma_algebra2
        L2_space.probability_measure = prob_measure2
        assert L2_space.probability_space == prob_space2
        assert L2_space.sigma_algebra == sigma_algebra2
        assert L2_space.probability_measure == prob_measure2


class TestHilbertSpaceMethods:

    @pytest.fixture
    def prob_space(self):
        sample_space = SampleSpace(["omega0", "omega1"])
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities={"omega0": 0.4, "omega1": 0.6}
        )
        return prob_space

    @pytest.fixture
    def X(self, prob_space):
        return RandomVariable.on_probability_space(
            outputs={"omega0": 1.0, "omega1": 2.0}, probability_space=prob_space
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable.on_probability_space(
            outputs={"omega0": 3.0, "omega1": 4.0}, probability_space=prob_space
        )

    @pytest.fixture
    def L2(self, prob_space):
        return L2(prob_space)

    def test_inner(self, L2, X, Y):
        inner_product = L2.inner(X, Y)
        expected_inner_product = 1.0 * 3.0 * 0.4 + 2.0 * 4.0 * 0.6
        assert inner_product == expected_inner_product

    def test_norm(self, L2, X):
        norm_X = L2.norm(X)
        expected_norm_X = (1.0**2 * 0.4 + 2.0**2 * 0.6) ** 0.5
        assert norm_X == expected_norm_X

    def test_distance(self, L2, X, Y):
        distance_XY = L2.distance(X, Y)
        expected_distance_XY = ((1.0 - 3.0) ** 2 * 0.4 + (2.0 - 4.0) ** 2 * 0.6) ** 0.5
        assert distance_XY == expected_distance_XY


class TestValidation:

    def test_invalid_probability_space(self):
        with pytest.raises(
            TypeError, match="probability_space must be a ProbabilitySpace instance."
        ):
            L2(probability_space="not_a_probability_space")

    def test_setting_invalid_sigma_algebra(self):
        sample_space = SampleSpace(["omega0", "omega1"])
        prob_space = ProbabilitySpace(sample_space=sample_space)
        L2_space = L2(prob_space)
        with pytest.raises(
            TypeError, match="sigma_algebra must be a SigmaAlgebra instance."
        ):
            L2_space.sigma_algebra = "not_a_sigma_algebra"

    def test_setting_invalid_probability_measure(self):
        sample_space = SampleSpace(["omega0", "omega1"])
        prob_space = ProbabilitySpace(sample_space=sample_space)
        L2_space = L2(prob_space)
        with pytest.raises(
            TypeError,
            match="probability_measure must be a ProbabilityMeasure instance.",
        ):
            L2_space.probability_measure = "not_a_probability_measure"

    def test_setting_invalid_name(self):
        sample_space = SampleSpace(["omega0", "omega1"])
        prob_space = ProbabilitySpace(sample_space=sample_space)
        L2_space = L2(prob_space)
        with pytest.raises(TypeError, match="name must be a string."):
            L2_space.name = 12345
