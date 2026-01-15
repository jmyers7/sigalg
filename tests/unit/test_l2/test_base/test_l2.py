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
        P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        H = L2(
            sample_space=Omega,
            sigma_algebra=F,
            probability_measure=P,
            name="H",
        )

        assert H.sample_space == Omega
        assert H.sigma_algebra == F
        assert H.probability_measure == P
        assert H.name == "H"

    def test_constructor_with_defaults(self):
        """Test L2 constructor with default sigma algebra and probability measure."""
        Omega = SampleSpace().from_sequence(size=3)
        H = L2(sample_space=Omega)

        assert H.sample_space == Omega
        assert H.sigma_algebra == SigmaAlgebra.power_set(sample_space=Omega)
        assert H.probability_measure == ProbabilityMeasure.uniform(sample_space=Omega)
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
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    def test_basis_returns_dict(self, Omega, F, P):
        """Test that basis returns a dictionary."""
        H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)
        basis = H.basis

        assert isinstance(basis, dict)

    def test_basis_has_correct_number_of_vectors(self, Omega, F, P):
        """Test that basis has one vector per atom with nonzero probability."""
        H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)
        basis = H.basis

        assert len(basis) == 2

    def test_basis_vectors_are_random_variables(self, Omega, F, P):
        """Test that all basis vectors are RandomVariable instances."""
        H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)
        basis = H.basis

        for basis_vec in basis.values():
            assert isinstance(basis_vec, RandomVariable)

    def test_basis_vectors_are_orthonormal(self, Omega, F, P):
        """Test that basis vectors are orthonormal."""
        H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)
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
        Q = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.8, 2: 0.0})
        H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=Q)
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
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    def test_probability_space_property(self, Omega, F, P):
        """Test that probability_space property returns a ProbabilitySpace."""
        from sigalg.core.base.probability_space import ProbabilitySpace

        H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)

        assert isinstance(H.probability_space, ProbabilitySpace)
        assert H.probability_space.sample_space == Omega
        assert H.probability_space.sigma_algebra == F
        assert H.probability_space.probability_measure == P

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
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)

    def test_integrate_random_variable_in_l2(self, H, Omega):
        """Test integration of a random variable in the L2-space."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        integral = H.integrate(X)
        expected = 1.6

        assert abs(integral - expected) < 1e-9

    def test_integrate_random_variable_not_in_l2_raises(self, Omega, H):
        """Test that integrating a non-measurable random variable raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.integrate(X)


class TestL2FourierCoefficients:

    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)

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
            for coeff, basis_vec in zip(coeffs.values(), H.basis.values())
        )

        assert np.allclose(X_reconstructed.data, X.data)

    def test_fourier_coefficients_random_variable_not_in_l2_raises(self, Omega, H):
        """Test that computing coefficients for non-measurable RV raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.fourier_coefficients(X)

    def test_fourier_coefficients_with_zero_probability_atom(self, Omega, F):
        """Test Fourier coefficients when an atom has zero probability."""
        Q = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.8, 2: 0.0})
        H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=Q)
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
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)

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
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)

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
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)

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
        expected_norm = H.probability_measure(A) ** 0.5

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


class TestL2Distance:

    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.5, 2: 0.3}
        )

    @pytest.fixture
    def H(self, Omega, F, P):
        return L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)

    def test_distance_returns_nonnegative(self, H, Omega):
        """Test that distance returns a non-negative value."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})
        dist = H.distance(X, Y)

        assert dist >= 0

    def test_distance_to_self_is_zero(self, H, Omega):
        """Test that distance from a vector to itself is zero."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        dist = H.distance(X, X)

        assert abs(dist) < 1e-9

    def test_distance_is_symmetric(self, H, Omega):
        """Test that distance is symmetric."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})

        assert abs(H.distance(X, Y) - H.distance(Y, X)) < 1e-9

    def test_distance_equals_norm_of_difference(self, H, Omega):
        """Test that d(X, Y) = ||X - Y||."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})
        dist = H.distance(X, Y)
        norm_diff = H.norm(X - Y)

        assert abs(dist - norm_diff) < 1e-9

    def test_distance_satisfies_triangle_inequality(self, H, Omega):
        """Test that distance satisfies triangle inequality: d(X,Z) <= d(X,Y) + d(Y,Z)."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 2, 1: 2, 2: 4})
        Z = RandomVariable(domain=Omega, name="Z").from_dict({0: 4, 1: 4, 2: 6})

        d_XZ = H.distance(X, Z)
        d_XY = H.distance(X, Y)
        d_YZ = H.distance(Y, Z)

        assert d_XZ <= d_XY + d_YZ + 1e-9

    def test_distance_first_not_in_l2_raises(self, H, Omega):
        """Test that distance with first RV not in L2 raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 0, 1: 1, 2: 2})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 4, 1: 4, 2: 6})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.distance(X, Y)

    def test_distance_second_not_in_l2_raises(self, H, Omega):
        """Test that distance with second RV not in L2 raises ValueError."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 1, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 0, 1: 1, 2: 2})

        with pytest.raises(ValueError, match="must be in the L2-space"):
            H.distance(X, Y)
