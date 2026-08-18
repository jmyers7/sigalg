import pytest
from sigalg.core import (
    Function,
    MeasurableFunction,
    ProbabilityMeasure,
    RandomVariable,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- arithmetic --------------------- #


class TestArithmeticReturnTypes:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(domain=Omega, mapping=dict(zip(Omega, [0, 1, 1, 2])))

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F, mapping=dict(zip(F.atom_space, [0.1, 0.4, 0.5]))
        )

    @pytest.fixture
    def X(self, Omega, F, P):
        return RandomVariable.from_rand(domain=Omega, sig_alg=F, measure=P)

    def test_sum_with_scalar_creates_random_variable(self, X, F, P):
        """Test that sum with a scalar creates a random variable."""
        sum = X + 4

        assert isinstance(sum, RandomVariable)
        assert sum.sig_alg is F
        assert sum.measure is P

    def test_sum_with_random_variable_creates_random_variable(self, Omega, F, P, X):
        """Test that the sum with a random variable creates another random variable."""
        Y = RandomVariable.from_rand(domain=Omega, sig_alg=F, measure=P, name="Y")
        sum = X + Y

        assert isinstance(sum, RandomVariable)
        assert sum.sig_alg is F
        assert sum.measure is P

    def test_sum_with_measurable_function_without_measure_creates_random_variable(
        self, Omega, F, P, X
    ):
        """Test that the sum with a measurable function without a measure creates a random variable."""
        g = MeasurableFunction.from_rand(domain=Omega, sig_alg=F, name="g")
        sum = X + g

        assert isinstance(sum, RandomVariable)
        assert sum.sig_alg is F
        assert sum.measure is P

    def test_sum_with_measurable_function_instance_creates_random_variable(
        self, Omega, F, P, X
    ):
        """Test that summing with a measurable Function instance creates a random variable."""
        h = Function(
            domain=Omega,
            mapping=dict(
                zip(
                    Omega,
                    [2, 4, 4, -3],
                )
            ),
            name="h",
        )
        sum = X + h

        assert isinstance(sum, RandomVariable)
        assert sum.sig_alg is F
        assert sum.measure is P

    def test_sum_with_non_measurable_function_instance_creates_function(self, Omega, X):
        """Test that summing with a non-measurable Function instance creates a Function instance."""
        k = Function(
            domain=Omega,
            mapping=dict(
                zip(
                    Omega,
                    [1, 2, 3, 4],
                )
            ),
            name="k",
        )
        sum = X + k

        assert isinstance(sum, Function)
        assert not isinstance(sum, MeasurableFunction)
        assert not isinstance(sum, RandomVariable)
