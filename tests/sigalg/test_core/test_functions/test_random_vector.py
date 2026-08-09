import pandas as pd
import pytest
from sigalg.core import (
    MeasureSpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test properties --------------------- #


class TestGeneratedSigAlg:
    def test_generated_sigma_algebra_property(self):
        """Test generated_sigma_algebra property of RandomVector."""
        Omega = SampleSpace.from_sequence(size=3)
        outputs_2d = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        outputs_1d = {0: 10, 1: 20, 2: 30}
        X = RandomVector.with_uniform(domain=Omega, mapping=outputs_2d)
        Y = RandomVector.with_uniform(domain=Omega, mapping=outputs_1d, name="Y")
        expected_sigma_algebra_2d = SigmaAlgebra(
            domain=Omega,
            mapping=outputs_2d,
            name="sigma(X)",
        )
        expected_sigma_algebra_1d = SigmaAlgebra(
            domain=Omega,
            mapping=outputs_1d,
            name="sigma(Y)",
        )

        assert X.generated_sig_alg == expected_sigma_algebra_2d
        assert Y.generated_sig_alg == expected_sigma_algebra_1d


class TestProbSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.8,
                1: 0.2,
            },
        )

    @pytest.fixture
    def mapping(self):
        return {
            0: (1, 2),
            1: (1, 2),
            2: (3, 4),
        }

    def test_prob_space_with_defaults(self, Omega, mapping):
        """Test that default probability space has power-set sigma-algebra and uniform probability measure."""
        X = RandomVector.with_uniform(domain=Omega, mapping=mapping)
        prob_space = ProbabilitySpace(domain=Omega)

        assert X.prob_space == prob_space
        assert X.prob_space.domain == Omega
        assert X.prob_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert X.prob_space.measure == ProbabilityMeasure.uniform(domain=Omega)

    def test_prob_space_with_custom_prob_measure(self, Omega, P, mapping):
        """Test constructor with custom probability measure sets sigma-algebra to the sigma-algebra of the probability measure."""
        X = RandomVector(domain=Omega, measure=P, mapping=mapping)
        prob_space = ProbabilitySpace(Omega, measure=P)

        assert X.prob_space == prob_space
        assert X.prob_space.domain == Omega
        assert X.prob_space.sig_alg == P.sig_alg
        assert X.prob_space.measure == P

    def test_prob_space_with_custom_sigma_algebra(self, Omega, F, mapping):
        """Test constructor with custom sigma-algebra sets the probability measure to uniform over the sigma-algebra."""
        X = RandomVector.with_uniform(domain=Omega, sig_alg=F, mapping=mapping)
        prob_space = ProbabilitySpace(Omega, F)

        assert X.prob_space == prob_space
        assert X.prob_space.domain == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.measure == ProbabilityMeasure.uniform(F)

    def test_prob_space_with_all_components(self, Omega, F, P, mapping):
        """Test constructor with all components."""
        prob_space = ProbabilitySpace(Omega, F, P)
        X = RandomVector(*prob_space, mapping=mapping)

        assert X.prob_space == prob_space
        assert X.prob_space.domain == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.measure == P


# --------------------- test data access methods --------------------- #


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.3,
                1: 0.2,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return MeasureSpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            },
        )

    def test_call_method_on_sample_points(self, Omega, X, Y):
        """Test calling on sample points."""
        for sample_point in Omega:
            pd.testing.assert_series_equal(X(sample_point), X.data.loc[sample_point])
            assert Y(sample_point) == Y.data.loc[sample_point]

    def test_call_method_on_atoms(self, F, X, Y):
        """Test calling on atoms."""
        for atom_id, atom in F.atom_id_to_atom.items():
            pd.testing.assert_series_equal(X(atom), X.atom_data.loc[atom_id])
            assert Y(atom) == Y.atom_data.loc[atom_id]


# --------------------- arithmetic --------------------- #


class TestArithmetic:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 4),
                2: (5, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (10, 20),
                1: (30, 40),
                2: (50, 60),
            },
        )

    def test_add_two_random_vectors(self, X, Y):
        """Test adding two RandomVectors."""
        Z = X + Y
        expected_data = pd.DataFrame(
            [(11, 22), (33, 44), (55, 66)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.measure_space == X.measure_space
        assert Z.name == "(X+Y)"

    def test_add_random_vector_and_scalar(self, X):
        """Test adding a scalar to a RandomVector."""
        Z = X + 10
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.measure_space == X.measure_space
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_vector(self, X):
        """Test adding a RandomVector to a scalar (reverse add)."""
        Z = 10 + X
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10+X)"

    def test_sub_two_random_vectors(self, X, Y):
        """Test subtracting two RandomVectors."""
        Z = X - Y
        expected_values = pd.DataFrame(
            [(-9, -18), (-27, -36), (-45, -54)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_values)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-Y)"

    def test_sub_random_vector_and_scalar(self, X):
        """Test subtracting a scalar from a RandomVector."""
        Z = X - 5
        expected_data = pd.DataFrame(
            [(-4, -3), (-2, -1), (0, 1)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_vector(self, X):
        """Test subtracting a RandomVector from a scalar (reverse sub)."""
        Z = 5 - X
        expected_data = pd.DataFrame(
            [(4, 3), (2, 1), (0, -1)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(5-X)"

    def test_mul_two_random_vectors(self, X, Y):
        """Test multiplying two RandomVectors."""
        Z = X * Y
        expected_data = pd.DataFrame(
            [(10, 40), (90, 160), (250, 360)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*Y)"

    def test_mul_random_vector_and_scalar(self, X):
        """Test multiplying a RandomVector by a scalar."""
        Z = X * 10
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_vector(self, X):
        """Test multiplying a scalar by a RandomVector (reverse mul)."""
        Z = 10 * X
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10*X)"

    def test_truediv_two_random_vectors(self, X, Y):
        """Test dividing two RandomVectors."""
        Z = Y / X
        expected_data = pd.DataFrame(
            [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(Y/X)"

    def test_truediv_random_vector_and_scalar(self, Y):
        """Test dividing a RandomVector by a scalar."""
        Z = Y / 10
        expected_data = pd.DataFrame(
            [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            index=Y.domain.data,
            columns=Y.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(Y/10)"

    def test_rtruediv_scalar_and_random_vector(self, Y):
        """Test dividing a scalar by a RandomVector (reverse div)."""
        Z = 10 / Y
        expected_data = pd.DataFrame(
            [(1.0, 1 / 2), (1 / 3, 1 / 4), (1 / 5, 1 / 6)],
            index=Y.domain.data,
            columns=Y.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(10/Y)"

    def test_pow_two_random_vectors(self, prob_space):
        """Test exponentiating two RandomVectors."""
        X = RandomVector(*prob_space, mapping={0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Y = RandomVector(
            *prob_space, name="Y", mapping={0: (2, 2), 1: (2, 2), 2: (2, 2)}
        )
        Z = X**Y
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self, X):
        """Test exponentiating a RandomVector by a scalar."""
        Z = X**2
        expected_data = pd.DataFrame(
            [(1, 4), (9, 16), (25, 36)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self, X):
        """Test exponentiating a scalar by a RandomVector (reverse pow)."""
        Z = 2**X
        expected_data = pd.DataFrame(
            [(2, 4), (8, 16), (32, 64)],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(2**X)"


class TestArithmeticWithRandomVariable:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 3,
                2: 5,
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 10,
                1: 30,
                2: 50,
            },
        )

    def test_add_two_random_variable(self, X, Y):
        """Test adding two RandomVariable."""
        Z = X + Y
        expected_data = pd.Series(
            [
                11,
                33,
                55,
            ],
            index=X.domain.data,
            name="(X+Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X+Y)"

    def test_add_random_variable_and_scalar(self, X):
        """Test adding a scalar to a RandomVariable."""
        Z = X + 10
        expected_data = pd.Series(
            [
                11,
                13,
                15,
            ],
            index=X.domain.data,
            name="(X+10)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_variable(self, X):
        """Test adding a RandomVariable to a scalar (reverse add)."""
        Z = 10 + X
        expected_data = pd.Series(
            [
                11,
                13,
                15,
            ],
            index=X.domain.data,
            name="(10+X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10+X)"

    def test_sub_two_random_variables(self, X, Y):
        """Test subtracting two RandomVariables."""
        Z = X - Y
        expected_values = pd.Series(
            [
                -9,
                -27,
                -45,
            ],
            index=X.domain.data,
            name="(X-Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-Y)"

    def test_sub_random_variable_and_scalar(self, X):
        """Test subtracting a scalar from a RandomVariable."""
        Z = X - 5
        expected_data = pd.Series(
            [
                -4,
                -2,
                0,
            ],
            index=X.domain.data,
            name="(X-5)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_variable(self, X):
        """Test subtracting a RandomVariable from a scalar (reverse sub)."""
        Z = 5 - X
        expected_data = pd.Series(
            [
                4,
                2,
                0,
            ],
            index=X.domain.data,
            name="(5-X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(5-X)"

    def test_mul_two_random_variables(self, X, Y):
        """Test multiplying two RandomVariables."""
        Z = X * Y
        expected_data = pd.Series(
            [
                10,
                90,
                250,
            ],
            index=X.domain.data,
            name="(X*Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*Y)"

    def test_mul_random_variable_and_scalar(self, X):
        """Test multiplying a RandomVariable by a scalar."""
        Z = X * 10
        expected_data = pd.Series(
            [
                10,
                30,
                50,
            ],
            index=X.domain.data,
            name="(X*10)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_variable(self, X):
        """Test multiplying a scalar by a RandomVariable (reverse mul)."""
        Z = 10 * X
        expected_data = pd.Series(
            [
                10,
                30,
                50,
            ],
            index=X.domain.data,
            name="(10*X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10*X)"

    def test_truediv_two_random_variables(self, X, Y):
        """Test dividing two RandomVariables."""
        Z = Y / X
        expected_data = pd.Series(
            [
                10.0,
                10.0,
                10.0,
            ],
            index=X.domain.data,
            name="(Y/X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(Y/X)"

    def test_truediv_random_variable_and_scalar(self, Y):
        """Test dividing a RandomVariable by a scalar."""
        Z = Y / 10
        expected_data = pd.Series(
            [
                1.0,
                3.0,
                5.0,
            ],
            index=Y.domain.data,
            name="(Y/10)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(Y/10)"

    def test_rtruediv_scalar_and_random_variable(self, Y):
        """Test dividing a scalar by a RandomVariable (reverse div)."""
        Z = 10 / Y
        expected_data = pd.Series(
            [
                1.0,
                1 / 3,
                1 / 5,
            ],
            index=Y.domain.data,
            name="(10/Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(10/Y)"

    def test_pow_two_random_variables(self, prob_space):
        """Test exponentiating two RandomVariable."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 4,
                2: 6,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 2,
            },
        )
        Z = X**Y
        expected_data = pd.Series(
            [
                4,
                16,
                36,
            ],
            index=X.domain.data,
            name="(X**Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self, X):
        """Test exponentiating a RandomVariable by a scalar."""
        Z = X**2
        expected_data = pd.Series(
            [
                1,
                9,
                25,
            ],
            index=X.domain.data,
            name="(X**2)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self, X):
        """Test exponentiating a scalar by a RandomVariable (reverse pow)."""
        Z = 2**X
        expected_data = pd.Series(
            [
                2,
                8,
                32,
            ],
            index=X.domain.data,
            name="(2**X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(2**X)"


# --------------------- comparison --------------------- #


class TestComparisonOperators:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    def test_lt_two_random_vectors(self, prob_space):
        """Test less than comparison of two RandomVectors."""
        X = RandomVector(*prob_space, mapping={0: (1, 2), 1: (2, 3), 2: (3, 4)})
        Y = RandomVector(
            *prob_space, name="Y", mapping={0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
        )
        Z = X < Y
        expected_data = pd.DataFrame(
            [[False, True], [False, True], [False, False]],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X < Y)"
        assert Z.prob_space == X.prob_space

    def test_le_two_random_vectors(self, prob_space):
        """Test less than or equal comparison of two RandomVectors."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (-1, 3),
                1: (2, 4),
                2: (3, 4),
            },
        )
        Z = X <= Y
        expected_data = pd.DataFrame(
            [[False, True], [True, True], [True, True]],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X <= Y)"
        assert Z.prob_space == X.prob_space

    def test_gt_two_random_vectors(self, prob_space):
        """Test greater than comparison of two RandomVectors."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (-1, 3),
                1: (2, 4),
                2: (3, 4),
            },
        )
        Z = X > Y
        expected_data = pd.DataFrame(
            [[True, False], [True, True], [False, False]],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X > Y)"
        assert Z.prob_space == X.prob_space

    def test_ge_two_random_vectors(self, prob_space):
        """Test greater than or equal comparison of two RandomVectors."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (-1, 3),
                1: (2, 4),
                2: (3, 4),
            },
        )
        Z = X >= Y
        expected_data = pd.DataFrame(
            [[True, False], [True, True], [True, True]],
            index=X.domain.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X >= Y)"
        assert Z.prob_space == X.prob_space

    def test_lt_random_variables(self, prob_space):
        """Test less than comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X < Y
        expected_data = pd.Series(
            [True, False, False],
            index=X.domain.data,
            name="(X < Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X < Y)"
        assert Z.prob_space == X.prob_space

    def test_le_random_variables(self, prob_space):
        """Test less than or equal comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X <= Y
        expected_data = pd.Series(
            [True, True, False],
            index=X.domain.data,
            name="(X <= Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X <= Y)"
        assert Z.prob_space == X.prob_space

    def test_gt_random_variables(self, prob_space):
        """Test greater than comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X > Y
        expected_data = pd.Series(
            [False, False, True],
            index=X.domain.data,
            name="(X > Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X > Y)"
        assert Z.prob_space == X.prob_space

    def test_ge_random_variables(self, prob_space):
        """Test greater than or equal comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X >= Y
        expected_data = pd.Series(
            [False, True, True],
            index=X.domain.data,
            name="(X >= Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X >= Y)"
        assert Z.prob_space == X.prob_space

    def test_lt_random_vector_and_scalar(self, prob_space):
        """Test less than comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 5),
            },
        )
        results = [X < 5, 5 > X]
        expected_data = pd.DataFrame(
            [[True, True], [True, False], [True, False]],
            index=X.domain.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X < 5)"
            assert result.prob_space == X.prob_space

    def test_le_random_vector_and_scalar(self, prob_space):
        """Test less than or equal comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 5),
            },
        )
        results = [X <= 5, 5 >= X]
        expected_data = pd.DataFrame(
            [[True, True], [True, True], [True, True]],
            index=X.domain.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X <= 5)"
            assert result.prob_space == X.prob_space

    def test_gt_random_vector_and_scalar(self, prob_space):
        """Test greater than comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 6),
            },
        )
        results = [X > 5, 5 < X]
        expected_data = pd.DataFrame(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            index=X.domain.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X > 5)"
            assert result.prob_space == X.prob_space

    def test_ge_random_vector_and_scalar(self, prob_space):
        """Test greater than or equal comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 6),
            },
        )
        results = [X >= 5, 5 <= X]
        expected_data = pd.DataFrame(
            [
                [False, False],
                [False, True],
                [False, True],
            ],
            index=X.domain.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X >= 5)"
            assert result.prob_space == X.prob_space

    def test_lt_random_variable_and_scalar(self, prob_space):
        """Test less than comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X < 5, 5 > X]
        expected_data = pd.Series(
            [
                True,
                False,
                False,
            ],
            index=X.domain.data,
            name="(X < 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X < 5)"
            assert result.prob_space == X.prob_space

    def test_le_random_variable_and_scalar(self, prob_space):
        """Test less than or equal comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X <= 5, 5 >= X]
        expected_data = pd.Series(
            [
                True,
                True,
                False,
            ],
            index=X.domain.data,
            name="(X <= 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X <= 5)"
            assert result.prob_space == X.prob_space

    def test_gt_random_variable_and_scalar(self, prob_space):
        """Test greater than comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X > 5, 5 < X]
        expected_data = pd.Series(
            [
                False,
                False,
                True,
            ],
            index=X.domain.data,
            name="(X > 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X > 5)"
            assert result.prob_space == X.prob_space

    def test_ge_random_variable_and_scalar(self, prob_space):
        """Test greater than or equal comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X >= 5, 5 <= X]
        expected_data = pd.Series(
            [
                False,
                True,
                True,
            ],
            index=X.domain.data,
            name="(X >= 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X >= 5)"
            assert result.prob_space == X.prob_space


class TestBooleanMethods:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    def test_all_returns_true_when_all_true(self, prob_space):
        """Test that all() returns True when all values are True."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (0, 1),
                1: (1, 2),
                2: (2, 3),
            },
        )
        Z = X > Y

        assert Z.all() is True

    def test_all_returns_false_when_some_false(self, prob_space):
        """Test that all() returns False when some values are False."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (1, 1),
                1: (1, 2),
                2: (2, 3),
            },
        )
        Z = X > Y

        assert Z.all() is False

    def test_any_returns_true_when_some_true(self, prob_space):
        """Test that any() returns True when at least one value is True."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 6),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 5),
            },
        )
        Z = X > Y

        assert Z.any() is True

    def test_any_returns_false_when_all_false(self, prob_space):
        """Test that any() returns False when all values are False."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Z = X > Y

        assert Z.any() is False

    def test_all_with_random_variable(self, prob_space):
        """Test all() method with RandomVariable."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 0,
                1: 1,
                2: 2,
            },
        )
        Z = X > Y

        assert Z.all() is True

    def test_any_with_random_variable(self, prob_space):
        """Test any() method with RandomVariable."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 2,
                2: 2,
            },
        )
        Z = X > Y

        assert Z.any() is True
