import numpy as np
import pytest

from sigalg.core import (
    ProbabilityMeasure,
    RandomVariable,
    SampleSpace,
    SigmaAlgebra,
)
from sigalg.l2 import L2


class TestL2Constructor:
    def test_constructor_with_all_parameters(self):
        """Test L2 constructor with all parameters specified."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        H = L2(
            sample_space=Omega,
            sig_alg=F,
            prob_measure=P,
            name="H",
        )

        assert H.sample_space == Omega
        assert H.sig_alg == F
        assert H.prob_measure == P
        assert H.name == "H"

    def test_constructor_with_defaults(self):
        """Test L2 constructor with default sigma algebra and probability measure."""
        Omega = SampleSpace().from_sequence(size=3)
        H = L2(sample_space=Omega)

        assert H.sample_space == Omega
        assert H.sig_alg == SigmaAlgebra.power_set(sample_space=Omega)
        assert H.prob_measure == ProbabilityMeasure.uniform(sig_alg=SigmaAlgebra.power_set(Omega))
        assert H.name == "H"


class TestL2Basis:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    def test_basis_returns_dict(self, Omega, F, P):
        """Test that basis returns a dictionary."""
        H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        basis = H.basis

        assert isinstance(basis, dict)

    def test_basis_has_correct_number_of_vectors(self, Omega, F, P):
        """Test that basis has one vector per atom with nonzero probability."""
        H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        basis = H.basis

        assert len(basis) == 2

    def test_basis_vectors_are_random_variables(self, Omega, F, P):
        """Test that all basis vectors are RandomVariable instances."""
        H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        basis = H.basis

        for basis_vec in basis.values():
            assert isinstance(basis_vec, RandomVariable)

    def test_basis_vectors_are_orthonormal(self, Omega, F, P):
        """Test that basis vectors are orthonormal."""
        H = L2(sample_space=Omega, sig_alg=F, prob_measure=P)
        basis_list = list(H.basis.values())

        for i, v_i in enumerate(basis_list):
            for j, v_j in enumerate(basis_list):
                inner_prod = H.inner(v_i, v_j)
                if i == j:
                    assert abs(inner_prod - 1.0) < 1e-9
                else:
                    assert abs(inner_prod) < 1e-9

    def test_basis_with_zero_probability_atom(self, Omega, F):
        """Test that basis excludes atoms with zero probability."""
        Q = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict({0: 0.2, 1: 0.8, 2: 0.0})
        H = L2(sample_space=Omega, sig_alg=F, prob_measure=Q)
        basis = H.basis

        assert len(basis) == 1


class TestL2Properties:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    def test_name_property(self, Omega):
        """Test that name property returns the correct name."""
        H = L2(sample_space=Omega, name="MyL2Space")

        assert H.name == "MyL2Space"

    def test_name_setter(self, Omega):
        """Test that name can be updated via setter."""
        H = L2(sample_space=Omega, name="H")
        H.name = "NewName"

        assert H.name == "NewName"


class TestL2Integrate:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sig_alg=F, prob_measure=P)

    def test_integrate_random_variable_in_l2(self, H, Omega):
        """Test integration of a random variable in the L2-space."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        integral = H.integrate(X)
        expected = 1.6

        assert abs(integral - expected) < 1e-9


class TestL2FourierCoefficients:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sig_alg=F, prob_measure=P)

    def test_fourier_coefficients_returns_dict(self, H, Omega):
        """Test that fourier_coefficients returns a dictionary."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 2, 1: 2, 2: 3})
        coeffs = H.fourier_coefficients(X)

        assert isinstance(coeffs, dict)

    def test_fourier_coefficients_has_correct_keys(self, H, Omega):
        """Test that coefficient keys match basis vector names."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 2, 1: 2, 2: 3})
        coeffs = H.fourier_coefficients(X)

        assert set(coeffs.keys()) == set(H.basis.keys())

    def test_fourier_coefficients_values_are_real(self, H, Omega):
        """Test that all coefficients are real numbers."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 2, 1: 2, 2: 3})
        coeffs = H.fourier_coefficients(X)

        for coeff in coeffs.values():
            assert isinstance(coeff, (int, float, np.number))

    def test_fourier_coefficients_reconstruct_rv(self, H, Omega):
        """Test that Fourier coefficients can reconstruct the random variable."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 2, 1: 2, 2: 3})
        coeffs = H.fourier_coefficients(X)
        X_reconstructed = sum(
            coeff * basis_vec
            for coeff, basis_vec in zip(coeffs.values(), H.basis.values(), strict=False)
        )

        assert np.allclose(X_reconstructed.data, X.data)

    def test_fourier_coefficients_random_variable_not_in_l2_raises(self, Omega, H):
        """Test that computing coefficients for non-measurable RV raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.fourier_coefficients(X)

    def test_fourier_coefficients_with_zero_probability_atom(self, Omega, F):
        """Test Fourier coefficients when an atom has zero probability."""
        Q = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict({0: 0.2, 1: 0.8, 2: 0.0})
        H = L2(sample_space=Omega, sig_alg=F, prob_measure=Q)
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 2, 1: 2, 2: 3})
        coeffs = H.fourier_coefficients(X)

        assert len(coeffs) == 1


class TestL2Contains:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sig_alg=F, prob_measure=P)

    def test_contains_measurable_random_variable(self, H, Omega):
        """Test that measurable random variable is in L2-space."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})

        assert X in H

    def test_contains_non_measurable_random_variable(self, H, Omega):
        """Test that non-measurable random variable is not in L2-space."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})

        assert X not in H

    def test_contains_basis_vector(self, H):
        """Test that basis vectors are in the L2-space."""
        for basis_vec in H.basis.values():
            assert basis_vec in H

    def test_contains_invalid_type_raises(self, H):
        """Test that checking membership of non-RandomVariable raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of RandomVariable"):
            "not a random variable" in H  # noqa: B015

    def test_contains_different_domain_raises(self, H):
        """Test that checking RV with different domain raises ValueError."""
        different_omega = SampleSpace().from_sequence(size=4)
        X = RandomVariable(domain=different_omega, name="X").from_dict(
            {0: 1, 1: 2, 2: 3, 3: 4}
        )

        with pytest.raises(ValueError, match="domain.*must match"):
            X in H  # noqa: B015


class TestL2Inner:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sig_alg=F, prob_measure=P)

    def test_inner_product_two_random_variables(self, H, Omega):
        """Test inner product of two random variables in L2-space."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})
        inner_prod = H.inner(X, Y)
        expected = 8.2

        assert abs(inner_prod - expected) < 1e-9

    def test_inner_product_is_symmetric(self, H, Omega):
        """Test that inner product is symmetric."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})

        assert abs(H.inner(X, Y) - H.inner(Y, X)) < 1e-9

    def test_inner_product_orthogonal_vectors(self, H, F):
        """Test that orthogonal vectors have zero inner product."""
        A, B = F.to_atoms()
        I_A = RandomVariable.indicator_of(A)
        I_B = RandomVariable.indicator_of(B)
        inner_prod = H.inner(I_A, I_B)

        assert abs(inner_prod) < 1e-9

    def test_inner_product_with_self(self, H, Omega):
        """Test inner product of a vector with itself."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        inner_prod = H.inner(X, X)
        expected = 0.2 + 0.5 + 2.7

        assert abs(inner_prod - expected) < 1e-9

    def test_inner_product_first_not_in_l2_raises(self, H, Omega):
        """Test that inner product with first RV not in L2 raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.inner(X, Y)

    def test_inner_product_second_not_in_l2_raises(self, H, Omega):
        """Test that inner product with second RV not in L2 raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 0, 1: 1, 2: 2})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.inner(X, Y)


class TestL2Norm:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sig_alg=F, prob_measure=P)

    def test_norm_returns_nonnegative(self, H, Omega):
        """Test that norm returns a non-negative value."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        norm = H.norm(X)

        assert norm >= 0

    def test_norm_squared_equals_inner_product(self, H, Omega):
        """Test that ||X||^2 = <X, X>."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        norm_squared = H.norm(X) ** 2
        inner_prod = H.inner(X, X)

        assert abs(norm_squared - inner_prod) < 1e-9

    def test_norm_of_indicator_function(self, H, F):
        """Test norm of indicator function equals sqrt of probability."""
        A, _ = F.to_atoms()
        I_A = RandomVariable.indicator_of(A)
        norm = H.norm(I_A)
        expected_norm = H.prob_measure(A) ** 0.5

        assert abs(norm - expected_norm) < 1e-9

    def test_norm_of_basis_vector_equals_one(self, H):
        """Test that basis vectors have unit norm."""
        for basis_vec in H.basis.values():
            norm = H.norm(basis_vec)
            assert abs(norm - 1.0) < 1e-9

    def test_norm_not_in_l2_raises(self, H, Omega):
        """Test that computing norm of non-measurable RV raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.norm(X)


class TestL2Metric:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sig_alg=F, prob_measure=P)

    def test_metric_returns_nonnegative(self, H, Omega):
        """Test that metric returns a non-negative value."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})
        dist = H.metric(X, Y)

        assert dist >= 0

    def test_metric_to_self_is_zero(self, H, Omega):
        """Test that metric from a vector to itself is zero."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        dist = H.metric(X, X)

        assert abs(dist) < 1e-9

    def test_metric_is_symmetric(self, H, Omega):
        """Test that metric is symmetric."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})

        assert abs(H.metric(X, Y) - H.metric(Y, X)) < 1e-9

    def test_metric_equals_norm_of_difference(self, H, Omega):
        """Test that d(X, Y) = ||X - Y||."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})
        dist = H.metric(X, Y)
        norm_diff = H.norm(X - Y)

        assert abs(dist - norm_diff) < 1e-9

    def test_metric_satisfies_triangle_inequality(self, H, Omega):
        """Test that metric satisfies triangle inequality: d(X,Z) <= d(X,Y) + d(Y,Z)."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 2, 1: 2, 2: 4})
        Z = RandomVariable(domain=Omega, name="Z").from_dict({0: 4, 1: 4, 2: 6})

        d_XZ = H.metric(X, Z)
        d_XY = H.metric(X, Y)
        d_YZ = H.metric(Y, Z)

        assert d_XZ <= d_XY + d_YZ + 1e-9

    def test_metric_first_not_in_l2_raises(self, H, Omega):
        """Test that metric with first RV not in L2 raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.metric(X, Y)

    def test_metric_second_not_in_l2_raises(self, H, Omega):
        """Test that metric with second RV not in L2 raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 0, 1: 1, 2: 2})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.metric(X, Y)


class TestL2Proj:
    @pytest.fixture
    def H(self):
        Omega = SampleSpace().from_sequence(size=4)
        P = ProbabilityMeasure(sig_alg=SigmaAlgebra.power_set(Omega)).from_dict(
            {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.2}
        )
        return L2(sample_space=Omega, prob_measure=P)

    def test_proj_onto_constant(self, H):
        """Test projection onto constant function (mean)."""
        Omega = H.sample_space
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )
        one = RandomVariable(domain=Omega, name="one").from_constant(1)
        proj, coeffs, dim = H.proj(rv=X, subspace=[one])
        expected_proj = X.expectation(prob_measure=H.prob_measure)

        assert dim == 1
        assert proj == expected_proj
        assert np.isclose(coeffs[0], expected_proj.item())

    def test_proj_identity(self, H):
        """Test that projecting onto a space containing the vector gives the vector."""
        Omega = H.sample_space
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )
        proj, coeffs, dim = H.proj(rv=X, subspace=[X])

        assert dim == 1
        assert proj == X
        assert np.isclose(coeffs[0], 1.0)

    def test_proj_centered(self, H):
        """Test projection of a centered random variable onto the constant function should be zero."""
        Omega = H.sample_space
        P = H.prob_measure
        one = RandomVariable(domain=Omega, name="one").from_constant(1)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )
        X_centered = X - X.expectation(prob_measure=P)
        proj, coeffs, dim = H.proj(rv=X_centered, subspace=[one])

        assert dim == 1
        assert np.isclose(proj.item(), 0.0, atol=1e-10)
        assert np.isclose(coeffs[0], 0.0, atol=1e-10)

    def test_proj_linear_dependent(self, H):
        """Test projection with linearly dependent spanning set."""
        Omega = H.sample_space
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 1.0, 1: 3.0, 2: 2.0, 3: 5.0}
        )
        Z = 2 * X
        _, _, dim = H.proj(rv=Y, subspace=[X, Y, Z])

        assert dim == 2

    def test_proj_polynomial_regression(self, H):
        """Test polynomial regression example."""
        Omega = H.sample_space
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 2.0, 1: 3.0, 2: 5.0, 3: 7.0}
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 1.0, 1: 3.0, 2: 2.0, 3: 4.0}
        )
        one = RandomVariable(domain=Omega, name="one").from_constant(1)
        proj, u, dim = H.proj(rv=Y, subspace=[one, X, X**2])
        expected_proj = sum([u[k] * X**k for k in range(dim)])

        assert proj == expected_proj

    def test_orthogonality(self, H):
        """Test orthogonality of the residual to the subspace."""
        Omega = H.sample_space
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 2.0, 1: 3.0, 2: 5.0, 3: 7.0}
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 1.0, 1: 3.0, 2: 2.0, 3: 4.0}
        )
        one = RandomVariable(domain=Omega, name="one").from_constant(1)
        proj, _, _ = H.proj(rv=Y, subspace=[one, X, X**2])
        residual = Y - proj

        for span_rv in [one, X, X**2]:
            inner_prod = H.inner(residual, span_rv)
            assert abs(inner_prod) < 1e-9

    def test_proj_raises_on_empty_subspace(self, H):
        """Test that empty subspace raises ValueError."""
        Omega = H.sample_space
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )

        with pytest.raises(ValueError, match="nonempty"):
            H.proj(rv=X, subspace=[])

    def test_proj_raises_on_rv_not_in_space(self, H):
        """Test that rv not in L2 space raises ValueError."""
        Omega = H.sample_space
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 0, 3: 1})
        H.sig_alg = F
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )
        one = RandomVariable(domain=Omega, name="one").from_constant(1)

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.proj(rv=X, subspace=[one])

    def test_proj_raises_on_subspace_rv_not_in_space(self, H):
        """Test that subspace rv not in L2 space raises ValueError."""
        Omega = H.sample_space
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 0, 3: 1})
        H.sig_alg = F
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 1.0, 2: 1.0, 3: 4.0}
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )

        with pytest.raises(ValueError, match="All random variables.*must be"):
            H.proj(rv=X, subspace=[Y])

    def test_proj_name_generation(self, H):
        """Test that projection gets appropriate name."""
        Omega = H.sample_space
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        )
        X_unnamed = (
            RandomVariable(domain=Omega)
            .from_dict({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0})
            .with_name(None)
        )
        one = RandomVariable(domain=Omega, name="one").from_constant(1)
        proj, _, _ = H.proj(rv=X, subspace=[one])
        proj_unnamed, _, _ = H.proj(rv=X_unnamed, subspace=[one])

        assert proj.name == "X_proj"
        assert proj_unnamed.name == "proj"
