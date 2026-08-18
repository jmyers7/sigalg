import pytest
from sigalg.core import Domain, Function, MeasurableFunction, SigmaAlgebra

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
        return MeasurableFunction.from_rand(domain=X, sig_alg=F)

    def test_sum_with_scalar_creates_measurable_function(self, f, F):
        """Test that sum with a scalar creates a measurable function."""
        sum = f + 4

        assert isinstance(sum, MeasurableFunction)
        assert sum.sig_alg is F

    def test_sum_with_measurable_function_creates_measurable_function(self, X, F, f):
        """Test that the sum with a measurable function creates another measurable function."""
        g = MeasurableFunction.from_rand(domain=X, sig_alg=F, name="g")
        sum = f + g

        assert isinstance(sum, MeasurableFunction)
        assert sum.sig_alg is F

    def test_sum_with_measurable_function_instance_creates_measurable_function(
        self, X, f, F
    ):
        """Test that summing with a measurable Function instance creates a measurable function."""
        h = Function(domain=X, mapping=dict(zip(X, [2, 4, 4, -3])), name="h")
        sum = f + h

        assert isinstance(sum, MeasurableFunction)
        assert sum.sig_alg is F

    def test_sum_with_non_measurable_function_instance_creates_function(self, X, f):
        """Test that summing with a non-measurable Function instance creates a Function instance."""
        k = Function(domain=X, mapping=dict(zip(X, [1, 2, 3, 4])), name="k")

        assert isinstance(f + k, Function)
        assert not isinstance(f + k, MeasurableFunction)
