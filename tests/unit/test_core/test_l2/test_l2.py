import pandas as pd
import pytest

from sigalg.core import (
    L2,
    ProbabilityMeasure,
    MeasureSpace,
    RandomVariable,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_no_parameters(self):
        """Test the base constructor for an empty L2 space."""
        H = L2()
        prob_space = MeasureSpace()

        assert H.prob_space == prob_space
        assert H.sample_space is None
        assert H.sig_alg is None
        assert H.prob_measure is None
        assert H.name == "H"
        assert H.basis is None
        assert H.basis_df is None

    def test_all_parameters(self):
        """Test the base constructor with all parameters specified."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 1,
                1: 2,
                2: 0,
                3: 0,
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.0,
                1: 0.55,
                2: 0.45,
            },
        )
        K = L2(Omega, F, P, name="K")
        prob_space = MeasureSpace(Omega, F, P)

        assert K.prob_space == prob_space
        assert K.sample_space is Omega
        assert K.sig_alg is F
        assert K.prob_measure is P
        assert K.name == "K"
        assert K.basis is not None
        assert K.basis_df is not None


# --------------------- test properties --------------------- #


class TestBasisDF:
    def test_with_1_diml_atom_ids_and_atom_with_0_prob(self):
        """Test the `basis_df` property when the sigma-algebra has 1-dimensional atom IDs and one atom has zero probability."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.0,
                1: 0.55,
                2: 0.45,
            },
        )
        H = L2(Omega, F, P)
        _, (_, A_1), (_, A_2) = F
        I_1 = RandomVariable.indicator_of(A_1)
        I_2 = RandomVariable.indicator_of(A_2)
        phi_1 = I_1 / P(A_1) ** 0.5
        phi_2 = I_2 / P(A_2) ** 0.5
        expected_basis_df = pd.concat([phi_1.data, phi_2.data], axis=1)
        expected_basis_df.columns = [1, 2]

        pd.testing.assert_frame_equal(H.basis_df, expected_basis_df)

    def test_with_1_diml_atom_ids_not_in_ascending_order(self):
        """Test the `basis_df` property when the sigma-algebra has 1-dimensional atom IDs that are not in ascending order."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 2,
                1: 0,
                2: 1,
                3: 1,
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.45,
                1: 0.0,
                2: 0.55,
            },
        )
        H = L2(Omega, F, P)
        (_, A_2), (_, A_0), _ = F
        I_2 = RandomVariable.indicator_of(A_2)
        I_0 = RandomVariable.indicator_of(A_0)
        phi_2 = I_2 / P(A_2) ** 0.5
        phi_0 = I_0 / P(A_0) ** 0.5
        expected_basis_df = pd.concat([phi_0.data, phi_2.data], axis=1)
        expected_basis_df.columns = [0, 2]

        pd.testing.assert_frame_equal(H.basis_df, expected_basis_df)

    def test_with_2_diml_atom_ids_and_atom_with_0_prob(self):
        """Test the `basis_df` property when the sigma-algebra has 2-dimensional atom IDs and one atom has zero probability."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: (0, 1),
                1: (2, 3),
                2: (4, 5),
                3: (4, 5),
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (0, 1): 0.0,
                (2, 3): 0.55,
                (4, 5): 0.45,
            },
        )
        H = L2(Omega, F, P)
        _, (_, A_1), (_, A_2) = F
        I_1 = RandomVariable.indicator_of(A_1)
        I_2 = RandomVariable.indicator_of(A_2)
        phi_1 = I_1 / P(A_1) ** 0.5
        phi_2 = I_2 / P(A_2) ** 0.5
        expected_basis_df = pd.concat([phi_1.data, phi_2.data], axis=1)
        expected_basis_df.columns = [(2, 3), (4, 5)]

        pd.testing.assert_frame_equal(H.basis_df, expected_basis_df)

    def test_with_2_diml_atom_ids_and_2_diml_sample_space(self):
        """Test the `basis_df` property when the sigma-algebra has 2-dimensional atom IDs and the sample space is 2-dimensional."""

        Omega = SampleSpace.cartesian_product(
            [[0, 1], ["a", "b"]], variable_names=["number", "letter"]
        )
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                (0, "a"): (3, 4),
                (0, "b"): (2, 3),
                (1, "a"): (2, 3),
                (1, "b"): (3, 4),
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (2, 3): 0.45,
                (3, 4): 0.55,
            },
        )
        H = L2(Omega, F, P)
        (_, A_34), (_, A_23) = F
        I_34 = RandomVariable.indicator_of(A_34)
        I_23 = RandomVariable.indicator_of(A_23)
        phi_34 = I_34 / P(A_34) ** 0.5
        phi_23 = I_23 / P(A_23) ** 0.5
        expected_basis_df = pd.concat([phi_23.data, phi_34.data], axis=1)
        expected_basis_df.columns = [(2, 3), (3, 4)]

        pd.testing.assert_frame_equal(H.basis_df, expected_basis_df)


class TestBasisAndDim:
    def test_with_1_diml_atom_ids_and_atom_with_0_prob(self):
        """Test the `basis_df` property when the sigma-algebra has 1-dimensional atom IDs and one atom has zero probability."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.0,
                1: 0.55,
                2: 0.45,
            },
        )
        H = L2(Omega, F, P)
        phi_1, phi_2 = H.basis.values()

        assert H.inner(phi_1, phi_2) == 0.0
        assert H.norm(phi_1) == 1.0
        assert H.norm(phi_2) == 1.0
        assert H.dim == 2

    def test_with_1_diml_atom_ids_not_in_ascending_order(self):
        """Test the `basis_df` property when the sigma-algebra has 1-dimensional atom IDs that are not in ascending order."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 2,
                1: 0,
                2: 1,
                3: 1,
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.45,
                1: 0.0,
                2: 0.55,
            },
        )
        H = L2(Omega, F, P)
        phi_2, phi_0 = H.basis.values()

        assert H.inner(phi_0, phi_2) == 0.0
        assert H.norm(phi_0) == 1.0
        assert H.norm(phi_2) == 1.0
        assert H.dim == 2

    def test_with_2_diml_atom_ids_and_atom_with_0_prob(self):
        """Test the `basis_df` property when the sigma-algebra has 2-dimensional atom IDs and one atom has zero probability."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: (0, 1),
                1: (2, 3),
                2: (4, 5),
                3: (4, 5),
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (0, 1): 0.0,
                (2, 3): 0.55,
                (4, 5): 0.45,
            },
        )
        H = L2(Omega, F, P)
        phi_23, phi_45 = H.basis.values()

        assert H.inner(phi_23, phi_45) == 0.0
        assert H.norm(phi_23) == 1.0
        assert H.norm(phi_45) == 1.0
        assert H.dim == 2

    def test_with_2_diml_atom_ids_and_2_diml_sample_space(self):
        """Test the `basis_df` property when the sigma-algebra has 2-dimensional atom IDs and the sample space is 2-dimensional."""

        Omega = SampleSpace.cartesian_product(
            [[0, 1], ["a", "b"]], variable_names=["number", "letter"]
        )
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                (0, "a"): (3, 4),
                (0, "b"): (2, 3),
                (1, "a"): (2, 3),
                (1, "b"): (3, 4),
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (2, 3): 0.45,
                (3, 4): 0.55,
            },
        )
        H = L2(Omega, F, P)
        phi_34, phi_23 = H.basis.values()

        assert H.inner(phi_23, phi_34) == 0.0
        assert H.norm(phi_23) == 1.0
        assert H.norm(phi_34) == 1.0
        assert H.dim == 2


# --------------------- test Hilbert space methods --------------------- #


class TestL2FourierCoefficients:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=5)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: (0, 1),
                1: (0, 1),
                2: (1, 2),
                3: (2, 3),
                4: (2, 3),
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (0, 1): 0.7,
                (1, 2): 0.3,
                (2, 3): 0.0,
            },
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(Omega, F, P)

    def test_reconstruct_rv(self, H, P):
        """Test that Fourier coefficients can reconstruct the random variable."""
        X = RandomVariable(
            *H.prob_space,
            name="X",
            mapping={
                0: 2,
                1: 2,
                2: 3,
                3: -4,
                4: -4,
            },
        )
        c = H.fourier_coefficients(X)
        phi = H.basis
        I = c.keys()
        X_reconstructed = sum(c[i] * phi[i] for i in I)

        assert P.equal_almost_surely(X, X_reconstructed)


class TestL2Inner:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=5)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: (0, 1),
                1: (0, 1),
                2: (1, 2),
                3: (2, 3),
                4: (2, 3),
            },
            variable_names=["F_0", "F_1"],
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (0, 1): 0.7,
                (1, 2): 0.3,
                (2, 3): 0.0,
            },
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(Omega, F, P)

    @pytest.fixture
    def X(self, H):
        return RandomVariable(
            *H.prob_space,
            name="X",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: -4,
                4: -4,
            },
        )

    @pytest.fixture
    def Y(self, H):
        return RandomVariable(
            *H.prob_space,
            name="Y",
            mapping={
                0: 4,
                1: 4,
                2: 6,
                3: -8,
                4: -8,
            },
        )

    def test_inner_product_two_random_variables(self, H, X, Y, P):
        """Test inner product of two random variables in L2-space."""
        expected_inner = (
            X(0) * Y(0) * P(F_0=0, F_1=1)
            + X(2) * Y(2) * P(F_0=1, F_1=2)
            + X(3) * Y(3) * P(F_0=2, F_1=3)
            + X(4) * Y(4) * P(F_0=2, F_1=3)
        )

        assert H.inner(X, Y) == expected_inner

    def test_symmetry(self, X, Y, H):
        """Test that inner product is symmetric."""
        assert H.inner(X, Y) == H.inner(Y, X)

    def test_bilinearity(self, X, Y, H):
        """Test that inner product is bilinear."""
        a = 2
        b = -3
        Z = RandomVariable(
            *H.prob_space,
            name="Z",
            mapping={
                0: 5,
                1: 5,
                2: 7,
                3: -10,
                4: -10,
            },
        )

        assert H.inner(a * X + b * Y, Z) == pytest.approx(
            a * H.inner(X, Z) + b * H.inner(Y, Z)
        )


class TestL2Norm:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=5)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: (0, 1),
                1: (0, 1),
                2: (1, 2),
                3: (2, 3),
                4: (2, 3),
            },
            variable_names=["F_0", "F_1"],
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (0, 1): 0.7,
                (1, 2): 0.3,
                (2, 3): 0.0,
            },
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(Omega, F, P)

    @pytest.fixture
    def X(self, H):
        return RandomVariable(
            *H.prob_space,
            name="X",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: -4,
                4: -4,
            },
        )

    def test_norm_of_rv(self, H, X, P):
        """Test the norm of a random variable."""
        expected_norm = (
            X(0) ** 2 * P(F_0=0, F_1=1)
            + X(2) ** 2 * P(F_0=1, F_1=2)
            + X(3) ** 2 * P(F_0=2, F_1=3)
            + X(4) ** 2 * P(F_0=2, F_1=3)
        ) ** 0.5

        assert H.norm(X) == expected_norm

    def test_norm_of_almost_zero_rv_is_zero(self, H):
        Z = RandomVariable(
            *H.prob_space,
            name="Z",
            mapping={
                0: 0,
                1: 0,
                2: 0,
                3: 4,
                4: 4,
            },
        )

        assert H.norm(Z) == 0

    def test_homogeneity(self, H, X):
        """Test that norm is homogeneous: ||aX|| = |a| * ||X||."""
        a = -3
        assert H.norm(a * X) == pytest.approx(abs(a) * H.norm(X))

    def test_triangle_inequality(self, H, X):
        """Test that norm satisfies triangle inequality: ||X + Y|| <= ||X|| + ||Y||."""
        Y = RandomVariable(
            *H.prob_space,
            name="Y",
            mapping={
                0: 4,
                1: 4,
                2: 6,
                3: -8,
                4: -8,
            },
        )

        assert H.norm(X + Y) <= H.norm(X) + H.norm(Y)


class TestL2Metric:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=5)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: (0, 1),
                1: (0, 1),
                2: (1, 2),
                3: (2, 3),
                4: (2, 3),
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (0, 1): 0.7,
                (1, 2): 0.3,
                (2, 3): 0.0,
            },
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(Omega, F, P)

    @pytest.fixture
    def X(self, H):
        return RandomVariable(
            *H.prob_space,
            name="X",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: -4,
                4: -4,
            },
        )

    @pytest.fixture
    def Y(self, H):
        return RandomVariable(
            *H.prob_space,
            name="Y",
            mapping={
                0: 4,
                1: 4,
                2: 6,
                3: -8,
                4: -8,
            },
        )

    def test_metric(self, X, Y, H):
        """Test that the metric is the norm of the difference."""
        expected_distance = H.norm(X - Y)

        assert H.metric(X, Y) == expected_distance

    def test_symmetry(self, X, Y, H):
        """Test that the metric is symmetric."""
        assert H.metric(X, Y) == H.metric(Y, X)

    def test_triangle_inequality(self, X, Y, H):
        """Test that the metric satisfies triangle inequality."""
        Z = RandomVariable(
            *H.prob_space,
            name="Z",
            mapping={
                0: 5,
                1: 5,
                2: 7,
                3: -10,
                4: -10,
            },
        )

        assert H.metric(X, Z) <= H.metric(X, Y) + H.metric(Y, Z)


class TestL2Proj:
    @pytest.fixture
    def H(self):
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: (0, 1),
                1: (1, 2),
                2: (2, 3),
                3: (2, 3),
            },
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                (0, 1): 0.2,
                (1, 2): 0.8,
                (2, 3): 0.0,
            },
        )
        return L2(Omega, F, P)

    @pytest.fixture
    def X(self, H):
        return RandomVariable(
            *H.prob_space,
            mapping={
                0: -2,
                1: 4,
                2: 5,
                3: 5,
            },
        )

    @pytest.fixture
    def Y(self, H):
        return RandomVariable(
            *H.prob_space,
            name="Y",
            mapping={
                0: -5,
                1: 1,
                2: 2,
                3: 2,
            },
        )

    def test_proj_onto_constant(self, X, H):
        """Test projection onto a constant yields the (unconditional) expectation."""
        one = RandomVariable.from_constant(*H.prob_space, name="one", constant=1)
        proj, coeffs, dim = H.proj(rv=X, subspace=[one])
        expected_proj = X.expectation()

        assert dim == 1
        assert proj == expected_proj
        assert coeffs[0] == pytest.approx(expected_proj.item())

    def test_proj_identity(self, X, H):
        """Test that projecting onto a space containing the variable gives the variable."""
        proj, coeffs, dim = H.proj(rv=X, subspace=[X])

        assert dim == 1
        assert proj == X
        assert coeffs[0] == pytest.approx(1.0)

    def test_proj_centered(self, X, H):
        """Test projection of a centered random variable onto the constant function should be zero."""
        one = RandomVariable.from_constant(*H.prob_space, name="one", constant=1)
        X_centered = X - X.expectation()
        proj, coeffs, dim = H.proj(rv=X_centered, subspace=[one])

        assert dim == 1
        assert proj.item() == pytest.approx(0.0)
        assert coeffs[0] == pytest.approx(0.0)

    def test_proj_polynomial_regression(self, X, Y, H):
        """Test polynomial regression example."""
        proj, c, _ = H.proj(rv=Y, subspace=[X**0, X**1, X**2])
        expected_proj = sum([c[k] * X**k for k in range(3)])

        assert proj == expected_proj

    def test_orthogonality(self, X, Y, H):
        """Test orthogonality of the residual to the subspace."""
        proj, _, _ = H.proj(rv=Y, subspace=[X**0, X**1, X**2])
        residual = Y - proj

        for span_rv in [X**0, X**1, X**2]:
            assert H.inner(residual, span_rv) == pytest.approx(0.0)
