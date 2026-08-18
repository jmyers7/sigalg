import pytest
from sigalg.core import Domain, Function, MeasurableVector, SigmaAlgebra

# --------------------- arithmetic --------------------- #


class TestArithmeticReturnTypes:
    @pytest.fixture
    def X(self):
        return Domain.from_sequence(size=4)

    @pytest.fixture
    def F(self, X):
        return SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2])))

    @pytest.fixture
    def f(self, X, F):
        return MeasurableVector.from_rand(domain=X, sig_alg=F, dim=2)

    def test_sum_with_scalar_creates_measurable_vector(self, f, F):
        """Test that sum with a scalar creates a measurable vector."""
        sum = f + 4

        assert isinstance(sum, MeasurableVector)
        assert sum.sig_alg is F

    def test_sum_with_measurable_vector_creates_measurable_vector(self, X, F, f):
        """Test that the sum with a measurable vector creates another measurable vector."""
        g = MeasurableVector.from_rand(domain=X, sig_alg=F, dim=2, name="g")
        sum = f + g

        assert isinstance(sum, MeasurableVector)
        assert sum.sig_alg is F

    def test_sum_with_measurable_function_instance_creates_measurable_vector(
        self, X, f, F
    ):
        """Test that summing with a measurable Function instance creates a measurable vector."""
        h = Function(
            domain=X, mapping=dict(zip(X, [(2, 1), (4, 1), (4, 1), (-3, 1)])), name="h"
        )
        sum = f + h

        assert isinstance(sum, MeasurableVector)
        assert sum.sig_alg is F

    def test_sum_with_non_measurable_function_instance_creates_function(self, X, f):
        """Test that summing with a non-measurable Function instance creates a Function instance."""
        k = Function(
            domain=X, mapping=dict(zip(X, [(1, 1), (2, 1), (3, 1), (4, 1)])), name="k"
        )

        assert isinstance(f + k, Function)
        assert not isinstance(f + k, MeasurableVector)
