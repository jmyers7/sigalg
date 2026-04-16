import numpy as np
import pandas as pd
import pytest

from sigalg.core import (
    Operators,
    ProbabilityMeasure,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)

correlation = Operators.correlation
covariance = Operators.covariance
expectation = Operators.expectation
integrate = Operators.integrate
pushforward = Operators.pushforward
std = Operators.std
variance = Operators.variance


class TestIntegrate:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.5,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVector(domain=Omega, name="X").from_dict(
            {
                0: (1, 2),
                1: (2, 1),
                2: (3, 4),
            }
        )

    @pytest.fixture
    def Y(self, Omega):
        return RandomVariable(domain=Omega, name="Y").from_dict(
            {
                0: 1,
                1: 1,
                2: 0,
            }
        )

    def test_integrate_random_vector_with_prob_measure_parameter(self, X, P):
        """Test integration of a 2D random vector with an explicit probability measure."""
        integral = integrate(rv=X, probability_measure=P)
        int_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        int_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4
        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["X_0", "X_1"], name="feature"),
            name="integral(X)",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_variable_with_prob_measure_parameter(self, Y, P):
        """Test integration of a random variable with an explicit probability measure."""
        integral = integrate(rv=Y, probability_measure=P)
        expected_integral = 0.5

        assert np.abs(integral - expected_integral) < 1e-9

    def test_integrate_random_vector_with_rv_prob_measure(self, X, P):
        """Test integration of a 2D random vector using the probability measure carried by the random vector."""
        X.with_probability_measure(probability_measure=P)
        integral = integrate(rv=X)
        int_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        int_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4
        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["X_0", "X_1"], name="feature"),
            name="integral(X)",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_variable_with_rv_prob_measure(self, Y, P):
        """Test integration of a random variable using the probability measure carried by the random variable."""
        Y.with_probability_measure(probability_measure=P)
        integral = integrate(rv=Y)
        expected_integral = 0.5

        assert np.abs(integral - expected_integral) < 1e-9

    def test_integrate_with_explicit_probability_measure_overrides_rv_measure(
        self, Omega, X, P
    ):
        """Test that explicit probability measure overrides the one carried by rv."""
        Q = ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.5,
            }
        )
        X.with_probability_measure(probability_measure=P)
        integral = integrate(rv=X, probability_measure=Q)
        int_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        int_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4

        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["X_0", "X_1"], name="feature"),
            name="integral(X)",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_vector_with_event(self, X, P, Omega):
        """Test integration of a random vector over an event."""
        A = Omega.get_event([0, 1])
        integral = integrate(rv=X, probability_measure=P, event=A)
        int_X0 = 0.2 * 1 + 0.3 * 2
        int_X1 = 0.2 * 2 + 0.3 * 1
        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["X_0", "X_1"], name="feature"),
            name="integral(X)",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_variable_with_event(self, Y, P, Omega):
        """Test integration of a random variable over an event."""
        A = Omega.get_event([0, 2])
        integral = integrate(rv=Y, probability_measure=P, event=A)
        expected_integral = Y(0) * P(0) + Y(2) * P(2)

        assert np.abs(integral - expected_integral) < 1e-9


class TestExpectation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.5,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVector(domain=Omega, name="X").from_dict(
            {
                0: (1, 2),
                1: (2, 1),
                2: (3, 4),
            }
        )

    @pytest.fixture
    def Z(self, Omega):
        return RandomVariable(domain=Omega, name="Z").from_dict(
            {
                0: 1,
                1: -2,
                2: 3,
            }
        )

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="F").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    def test_unconditional_expectation_random_vector(self, Omega, X, P):
        """Test unconditional expectation of a 2D random vector."""
        E_X = expectation(X, probability_measure=P)
        E_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        E_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4
        expected_data = pd.DataFrame(
            [
                [E_X0, E_X1],
                [E_X0, E_X1],
                [E_X0, E_X1],
            ],
            index=Omega.data,
            columns=pd.Index(["E(X)_0", "E(X)_1"], name="expectation"),
        )

        pd.testing.assert_frame_equal(E_X.data, expected_data)
        assert E_X.name == "E(X)"

    def test_unconditional_expectation_random_variable(self, Omega, Z, P):
        """Test unconditional expectation of a random variable."""
        E_Z = expectation(Z, probability_measure=P)
        expected_E_Z = 0.2 * 1 + 0.3 * (-2) + 0.5 * 3
        expected_data = pd.Series(
            [expected_E_Z] * len(Omega), index=Omega.data, name="E(Z)"
        )

        pd.testing.assert_series_equal(E_Z.data, expected_data)
        assert E_Z.name == "E(Z)"

    def test_conditional_expectation_random_vector(self, Omega, X, P, F):
        """Test conditional expectation of a random vector given a sigma algebra."""
        E_X_F = expectation(rv=X, sigma_algebra=F, probability_measure=P)

        E_X0_atom_0 = (0.2 * 1 + 0.3 * 2) / 0.5
        E_X1_atom_0 = (0.2 * 2 + 0.3 * 1) / 0.5
        E_X0_atom_1 = (0.3 * 3) / 0.3
        E_X1_atom_1 = (0.3 * 4) / 0.3
        expected_data = pd.DataFrame(
            [
                [E_X0_atom_0, E_X1_atom_0],
                [E_X0_atom_0, E_X1_atom_0],
                [E_X0_atom_1, E_X1_atom_1],
            ],
            index=Omega.data,
            columns=pd.Index(["E(X|F)_0", "E(X|F)_1"], name="expectation"),
        )

        pd.testing.assert_frame_equal(E_X_F.data, expected_data)
        assert E_X_F.name == "E(X|F)"

    def test_conditional_expectation_random_variable(self, Omega, Z, P, F):
        """Test conditional expectation of a random variable given a sigma algebra."""
        E_Z_F = expectation(rv=Z, sigma_algebra=F, probability_measure=P)
        E_Z_atom_0 = (0.2 * 1 + 0.3 * (-2)) / 0.5
        E_Z_atom_1 = (0.5 * 3) / 0.5

        expected_data = pd.Series(
            [E_Z_atom_0, E_Z_atom_0, E_Z_atom_1],
            index=Omega.data,
            name="E(Z|F)",
        )

        pd.testing.assert_series_equal(E_Z_F.data, expected_data)
        assert E_Z_F.name == "E(Z|F)"

    def test_expectation_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            expectation("not a random vector")

    def test_expectation_invalid_sigma_algebra_type_raises(self):
        """Test that invalid sigma algebra type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})
        P = ProbabilityMeasure.uniform(domain)

        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            expectation(X, sigma_algebra="not a sigma algebra", probability_measure=P)

    def test_expectation_invalid_probability_measure_type_raises(self):
        """Test that invalid probability measure type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})

        with pytest.raises(
            TypeError, match="probability_measure must be a ProbabilityMeasure"
        ):
            expectation(X, probability_measure="not a probability measure")


class TestVariance:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.5,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVector(domain=Omega, name="X").from_dict(
            {
                0: (1, 2),
                1: (2, 1),
                2: (3, 4),
            }
        )

    @pytest.fixture
    def Z(self, Omega):
        return RandomVariable(domain=Omega, name="Z").from_dict(
            {
                0: 1,
                1: -2,
                2: 3,
            }
        )

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="F").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    def test_unconditional_variance_random_vector(self, Omega, X, P):
        """Test unconditional variance of a 2D random vector."""
        V_X = variance(X, probability_measure=P)

        E_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        E_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4
        V_X0 = 0.2 * (1 - E_X0) ** 2 + 0.3 * (2 - E_X0) ** 2 + 0.5 * (3 - E_X0) ** 2
        V_X1 = 0.2 * (2 - E_X1) ** 2 + 0.3 * (1 - E_X1) ** 2 + 0.5 * (4 - E_X1) ** 2
        expected_data = pd.DataFrame.from_dict(
            dict.fromkeys(Omega, (V_X0, V_X1)),
            orient="index",
            columns=pd.Index(["V(X)_0", "V(X)_1"], name="variance"),
        )
        expected_data.index.name = Omega.data.name

        pd.testing.assert_frame_equal(V_X.data, expected_data)
        assert V_X.name == "V(X)"

    def test_unconditional_variance_random_variable(self, Omega, Z, P):
        """Test unconditional variance of a random variable."""
        V_Z = variance(Z, probability_measure=P)
        E_Z = 0.2 * 1 + 0.3 * (-2) + 0.5 * 3
        expected_V_Z = (
            0.2 * (1 - E_Z) ** 2 + 0.3 * (-2 - E_Z) ** 2 + 0.5 * (3 - E_Z) ** 2
        )
        expected_data = pd.Series(
            [expected_V_Z] * len(Omega),
            index=Omega.data,
            name="V(Z)",
        )

        pd.testing.assert_series_equal(V_Z.data, expected_data)
        assert V_Z.name == "V(Z)"

    def test_conditional_variance_random_vector(self, Omega, X, P, F):
        """Test conditional variance of a random vector given a sigma algebra."""
        V_X_F = variance(rv=X, sigma_algebra=F, probability_measure=P)
        E_X0_atom_0 = (0.2 * 1 + 0.3 * 2) / 0.5
        E_X1_atom_0 = (0.2 * 2 + 0.3 * 1) / 0.5
        E_X0_atom_1 = (0.5 * 3) / 0.5
        E_X1_atom_1 = (0.5 * 4) / 0.5
        V_X0_atom_0 = (
            0.2 * (1 - E_X0_atom_0) ** 2 + 0.3 * (2 - E_X0_atom_0) ** 2
        ) / 0.5
        V_X1_atom_0 = (
            0.2 * (2 - E_X1_atom_0) ** 2 + 0.3 * (1 - E_X1_atom_0) ** 2
        ) / 0.5
        V_X0_atom_1 = 0.5 * (3 - E_X0_atom_1) ** 2 / 0.5
        V_X1_atom_1 = 0.5 * (4 - E_X1_atom_1) ** 2 / 0.5
        expected_data = pd.DataFrame(
            [
                [V_X0_atom_0, V_X1_atom_0],
                [V_X0_atom_0, V_X1_atom_0],
                [V_X0_atom_1, V_X1_atom_1],
            ],
            index=Omega.data,
            columns=pd.Index(["V(X|F)_0", "V(X|F)_1"], name="variance"),
        )

        pd.testing.assert_frame_equal(V_X_F.data, expected_data)
        assert V_X_F.name == "V(X|F)"

    def test_conditional_variance_random_variable(self, Omega, Z, P, F):
        """Test conditional variance of a random variable given a sigma algebra."""
        V_Z_F = variance(Z, sigma_algebra=F, probability_measure=P)
        E_Z_atom_0 = (0.2 * 1 + 0.3 * (-2)) / 0.5
        E_Z_atom_1 = (0.5 * 3) / 0.5
        V_Z_atom_0 = (0.2 * (1 - E_Z_atom_0) ** 2 + 0.3 * (-2 - E_Z_atom_0) ** 2) / 0.5
        V_Z_atom_1 = (0.5 * (3 - E_Z_atom_1) ** 2) / 0.5
        expected_data = pd.Series(
            [
                V_Z_atom_0,
                V_Z_atom_0,
                V_Z_atom_1,
            ],
            index=Omega.data,
            name="V(Z|F)",
        )

        pd.testing.assert_series_equal(V_Z_F.data, expected_data)
        assert V_Z_F.name == "V(Z|F)"

    def test_variance_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            variance("not a random vector")

    def test_variance_invalid_sigma_algebra_type_raises(self):
        """Test that invalid sigma algebra type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})
        P = ProbabilityMeasure.uniform(domain)

        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            variance(X, sigma_algebra="not a sigma algebra", probability_measure=P)


class TestStandardDeviation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.5,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVector(domain=Omega, name="X").from_dict(
            {
                0: (1, 2),
                1: (2, 1),
                2: (3, 4),
            }
        )

    @pytest.fixture
    def Z(self, Omega):
        return RandomVariable(domain=Omega, name="Z").from_dict(
            {
                0: 1,
                1: -2,
                2: 3,
            }
        )

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="F").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    def test_unconditional_std_random_vector(self, Omega, X, P):
        """Test unconditional standard deviation of a random vector."""
        std_X = std(X, probability_measure=P)
        V_X = variance(X, probability_measure=P)
        expected_std_X0 = V_X.data.iloc[0, 0] ** 0.5
        expected_std_X1 = V_X.data.iloc[0, 1] ** 0.5
        expected_data = pd.DataFrame(
            [
                [expected_std_X0, expected_std_X1],
                [expected_std_X0, expected_std_X1],
                [expected_std_X0, expected_std_X1],
            ],
            index=Omega.data,
            columns=pd.Index(["std(X)_0", "std(X)_1"], name="std"),
        )

        pd.testing.assert_frame_equal(std_X.data, expected_data)
        assert std_X.name == "std(X)"

    def test_unconditional_std_random_variable(self, Omega, Z, P):
        """Test unconditional standard deviation of a random variable."""
        std_Z = std(Z, probability_measure=P)
        V_Z = variance(Z, probability_measure=P)
        expected_std = V_Z.data.iloc[0] ** 0.5
        expected_data = pd.Series(
            [expected_std] * len(Omega),
            index=Omega.data,
            name="std(Z)",
        )

        pd.testing.assert_series_equal(std_Z.data, expected_data)
        assert std_Z.name == "std(Z)"

    def test_conditional_std_random_variable(self, Omega, Z, P, F):
        """Test conditional standard deviation of a random variable."""
        std_Z_F = std(Z, sigma_algebra=F, probability_measure=P)
        V_Z_F = variance(Z, sigma_algebra=F, probability_measure=P)
        std_Z_atom_0 = V_Z_F.data.iloc[0] ** 0.5
        std_Z_atom_1 = V_Z_F.data.iloc[2] ** 0.5
        expected_data = pd.Series(
            [std_Z_atom_0, std_Z_atom_0, std_Z_atom_1],
            index=Omega.data,
            name="std(Z|F)",
        )

        pd.testing.assert_series_equal(std_Z_F.data, expected_data)
        assert std_Z_F.name == "std(Z|F)"

    def test_std_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            std("not a random vector")


class TestCovariance:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.5,
            }
        )

    def test_covariance_two_random_vectors(self, Omega, P):
        """Test covariance of two 2D random vectors returns a matrix."""
        X = RandomVector(domain=Omega, name="X").from_dict(
            {
                0: (1, 2),
                1: (2, 1),
                2: (3, 4),
            }
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {
                0: (3, -2),
                1: (1, 5),
                2: (6, 8),
            }
        )
        cov = covariance(X, Y, probability_measure=P)
        E_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        E_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4
        E_Y0 = 0.2 * 3 + 0.3 * 1 + 0.5 * 6
        E_Y1 = 0.2 * (-2) + 0.3 * 5 + 0.5 * 8
        cov_00 = (
            0.2 * (1 - E_X0) * (3 - E_Y0)
            + 0.3 * (2 - E_X0) * (1 - E_Y0)
            + 0.5 * (3 - E_X0) * (6 - E_Y0)
        )
        cov_01 = (
            0.2 * (1 - E_X0) * (-2 - E_Y1)
            + 0.3 * (2 - E_X0) * (5 - E_Y1)
            + 0.5 * (3 - E_X0) * (8 - E_Y1)
        )
        cov_10 = (
            0.2 * (2 - E_X1) * (3 - E_Y0)
            + 0.3 * (1 - E_X1) * (1 - E_Y0)
            + 0.5 * (4 - E_X1) * (6 - E_Y0)
        )
        cov_11 = (
            0.2 * (2 - E_X1) * (-2 - E_Y1)
            + 0.3 * (1 - E_X1) * (5 - E_Y1)
            + 0.5 * (4 - E_X1) * (8 - E_Y1)
        )
        expected_data = pd.DataFrame(
            [
                [cov_00, cov_01],
                [cov_10, cov_11],
            ],
            index=X.data.columns,
            columns=Y.data.columns,
        )

        pd.testing.assert_frame_equal(cov, expected_data)

    def test_covariance_two_random_variables(self, Omega, P):
        """Test covariance of two random variables returns a scalar."""
        Z = RandomVariable(domain=Omega, name="Z").from_dict(
            {
                0: 1,
                1: -2,
                2: 3,
            }
        )
        W = RandomVariable(domain=Omega, name="W").from_dict(
            {
                0: 5,
                1: 6,
                2: 1,
            }
        )
        cov = covariance(Z, W, probability_measure=P)
        E_Z = 0.2 * 1 + 0.3 * (-2) + 0.5 * 3
        E_W = 0.2 * 5 + 0.3 * 6 + 0.5 * 1
        expected_cov = (
            0.2 * (1 - E_Z) * (5 - E_W)
            + 0.3 * (-2 - E_Z) * (6 - E_W)
            + 0.5 * (3 - E_Z) * (1 - E_W)
        )

        assert abs(cov - expected_cov) < 1e-9

    def test_covariance_single_random_vector(self, Omega, P):
        """Test covariance of a single random vector with itself."""
        X = RandomVector(domain=Omega, name="X").from_dict(
            {
                0: (1, 2),
                1: (2, 1),
                2: (3, 4),
            }
        )
        cov = covariance(X, probability_measure=P)
        E_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        E_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4
        cov_00 = 0.2 * (1 - E_X0) ** 2 + 0.3 * (2 - E_X0) ** 2 + 0.5 * (3 - E_X0) ** 2
        cov_01 = (
            0.2 * (1 - E_X0) * (2 - E_X1)
            + 0.3 * (2 - E_X0) * (1 - E_X1)
            + 0.5 * (3 - E_X0) * (4 - E_X1)
        )
        cov_10 = (
            0.2 * (2 - E_X1) * (1 - E_X0)
            + 0.3 * (1 - E_X1) * (2 - E_X0)
            + 0.5 * (4 - E_X1) * (3 - E_X0)
        )
        cov_11 = 0.2 * (2 - E_X1) ** 2 + 0.3 * (1 - E_X1) ** 2 + 0.5 * (4 - E_X1) ** 2
        expected_data = pd.DataFrame(
            [
                [cov_00, cov_01],
                [cov_10, cov_11],
            ],
            index=X.data.columns,
            columns=X.data.columns,
        )

        pd.testing.assert_frame_equal(cov, expected_data)

    def test_covariance_single_random_variable(self, Omega, P):
        """Test covariance of a single random variable with itself."""
        Z = RandomVariable(domain=Omega, name="Z").from_dict(
            {
                0: 1,
                1: -2,
                2: 3,
            }
        )
        cov = covariance(Z, probability_measure=P)
        E_Z = 0.2 * 1 + 0.3 * (-2) + 0.5 * 3
        expected_cov = (
            0.2 * (1 - E_Z) ** 2 + 0.3 * (-2 - E_Z) ** 2 + 0.5 * (3 - E_Z) ** 2
        )

        assert abs(cov - expected_cov) < 1e-9

    def test_covariance_invalid_rv1_type_raises(self):
        """Test that invalid rv1 type raises TypeError."""
        with pytest.raises(TypeError, match="rv1 must be a RandomVector"):
            covariance("not a random vector")

    def test_covariance_invalid_rv2_type_raises(self):
        """Test that invalid rv2 type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})
        P = ProbabilityMeasure.uniform(domain)

        with pytest.raises(TypeError, match="rv2 must be a RandomVector"):
            covariance(X, rv2="not a random vector", probability_measure=P)

    def test_covariance_different_domains_raises(self):
        """Test that random vectors with different domains raise ValueError."""
        Omega1 = SampleSpace().from_sequence(size=2, prefix="s")
        Omega2 = SampleSpace().from_sequence(size=2, prefix="t")
        X = RandomVariable(domain=Omega1, name="X").from_dict({"s_0": 1, "s_1": 2})
        Y = RandomVariable(domain=Omega2, name="Y").from_dict({"t_0": 3, "t_1": 4})
        P = ProbabilityMeasure.uniform(Omega1)

        with pytest.raises(ValueError, match="same domain"):
            covariance(X, Y, probability_measure=P)

    def test_covariance_different_dimensions_raises(self):
        """Test that random vectors with different dimensions raise ValueError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        )
        Z = RandomVariable(domain=Omega, name="Z").from_dict({0: 1, 1: 2, 2: 3})
        P = ProbabilityMeasure.uniform(Omega)

        with pytest.raises(ValueError, match="same dimension"):
            covariance(X, Z, probability_measure=P)


class TestCorrelation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.5,
            }
        )

    def test_correlation_two_random_vectors(self, Omega, P):
        """Test correlation of two 2D random vectors returns a matrix."""
        X = RandomVector(domain=Omega, name="X").from_dict(
            {
                0: (1, 2),
                1: (2, 1),
                2: (3, 4),
            }
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {
                0: (3, -2),
                1: (1, 5),
                2: (6, 8),
            }
        )
        corr = correlation(X, Y, probability_measure=P)
        E_X0 = 0.2 * 1 + 0.3 * 2 + 0.5 * 3
        E_X1 = 0.2 * 2 + 0.3 * 1 + 0.5 * 4
        E_Y0 = 0.2 * 3 + 0.3 * 1 + 0.5 * 6
        E_Y1 = 0.2 * (-2) + 0.3 * 5 + 0.5 * 8
        std_X0 = (
            0.2 * (1 - E_X0) ** 2 + 0.3 * (2 - E_X0) ** 2 + 0.5 * (3 - E_X0) ** 2
        ) ** 0.5
        std_X1 = (
            0.2 * (2 - E_X1) ** 2 + 0.3 * (1 - E_X1) ** 2 + 0.5 * (4 - E_X1) ** 2
        ) ** 0.5
        std_Y0 = (
            0.2 * (3 - E_Y0) ** 2 + 0.3 * (1 - E_Y0) ** 2 + 0.5 * (6 - E_Y0) ** 2
        ) ** 0.5
        std_Y1 = (
            0.2 * (-2 - E_Y1) ** 2 + 0.3 * (5 - E_Y1) ** 2 + 0.5 * (8 - E_Y1) ** 2
        ) ** 0.5
        corr_00 = (
            0.2 * (1 - E_X0) * (3 - E_Y0)
            + 0.3 * (2 - E_X0) * (1 - E_Y0)
            + 0.5 * (3 - E_X0) * (6 - E_Y0)
        ) / (std_X0 * std_Y0)
        corr_01 = (
            0.2 * (1 - E_X0) * (-2 - E_Y1)
            + 0.3 * (2 - E_X0) * (5 - E_Y1)
            + 0.5 * (3 - E_X0) * (8 - E_Y1)
        ) / (std_X0 * std_Y1)
        corr_10 = (
            0.2 * (2 - E_X1) * (3 - E_Y0)
            + 0.3 * (1 - E_X1) * (1 - E_Y0)
            + 0.5 * (4 - E_X1) * (6 - E_Y0)
        ) / (std_X1 * std_Y0)
        corr_11 = (
            0.2 * (2 - E_X1) * (-2 - E_Y1)
            + 0.3 * (1 - E_X1) * (5 - E_Y1)
            + 0.5 * (4 - E_X1) * (8 - E_Y1)
        ) / (std_X1 * std_Y1)
        expected_data = pd.DataFrame(
            [
                [corr_00, corr_01],
                [corr_10, corr_11],
            ],
            index=X.data.columns,
            columns=Y.data.columns,
        )

        pd.testing.assert_frame_equal(corr, expected_data)

    def test_correlation_two_random_variables(self, Omega, P):
        """Test correlation of two random variables returns a scalar."""
        Z = RandomVariable(domain=Omega, name="Z").from_dict(
            {
                0: 1,
                1: -2,
                2: 3,
            }
        )
        W = RandomVariable(domain=Omega, name="W").from_dict(
            {
                0: 5,
                1: 6,
                2: 1,
            }
        )
        corr = correlation(Z, W, probability_measure=P)
        E_Z = 0.2 * 1 + 0.3 * (-2) + 0.5 * 3
        E_W = 0.2 * 5 + 0.3 * 6 + 0.5 * 1
        std_Z = (
            0.2 * (1 - E_Z) ** 2 + 0.3 * (-2 - E_Z) ** 2 + 0.5 * (3 - E_Z) ** 2
        ) ** 0.5
        std_W = (
            0.2 * (5 - E_W) ** 2 + 0.3 * (6 - E_W) ** 2 + 0.5 * (1 - E_W) ** 2
        ) ** 0.5
        expected_corr = (
            0.2 * (1 - E_Z) * (5 - E_W)
            + 0.3 * (-2 - E_Z) * (6 - E_W)
            + 0.5 * (3 - E_Z) * (1 - E_W)
        ) / (std_Z * std_W)

        assert abs(corr - expected_corr) < 1e-9

    def test_correlation_equals_one_for_perfect_positive_correlation(self, Omega, P):
        """Test correlation equals 1 for perfectly positively correlated variables."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 2, 1: 4, 2: 6})
        corr = correlation(X, Y, probability_measure=P)

        assert abs(corr - 1.0) < 1e-9

    def test_correlation_equals_negative_one_for_perfect_negative_correlation(
        self, Omega, P
    ):
        """Test correlation equals -1 for perfectly negatively correlated variables."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 6, 1: 4, 2: 2})
        corr = correlation(X, Y, probability_measure=P)

        assert abs(corr - (-1.0)) < 1e-9


class TestProbabilityTheorems:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.2,
                2: 0.5,
                3: 0.1,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVariable(domain=Omega, name="X").from_dict(
            {
                0: 1,
                1: 2,
                2: 3,
                3: 4,
            }
        )

    @pytest.fixture
    def Y(self, Omega):
        return RandomVariable(domain=Omega, name="Y").from_dict(
            {
                0: 10,
                1: 10,
                2: 20,
                3: 20,
            }
        )

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="F").from_dict(
            {
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            }
        )

    def test_tower_law_for_conditional_expectation(self, Omega, P, X, F):
        """Test law of total expectation: E(E(X|F)|G) = E(X|G), if F is finer than G."""
        G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            }
        )
        E_X_F = expectation(X, sigma_algebra=F, probability_measure=P)
        E_E_X_F_G = expectation(E_X_F, sigma_algebra=G, probability_measure=P)
        E_X_G = expectation(X, sigma_algebra=G, probability_measure=P)

        assert np.allclose(E_E_X_F_G.data, E_X_G.data)

    def test_law_of_total_variance(self, Omega, P, X, F):
        """Test law of total variance: V(X) = E(V(X|F)) + V(E(X|F))."""
        V_X = variance(X, probability_measure=P).data.iloc[0]
        V_X_F = variance(X, sigma_algebra=F, probability_measure=P)
        E_V_X_F = expectation(V_X_F, probability_measure=P).data.iloc[0]
        E_X_F = expectation(X, sigma_algebra=F, probability_measure=P)
        V_E_X_F = variance(E_X_F, probability_measure=P).data.iloc[0]

        assert np.allclose(V_X, E_V_X_F + V_E_X_F)

    def test_expectation_of_measurable_random_variable_equals_itself(self, P, F, Y):
        """Test E(X|F) = X if X is F-measurable."""
        E_Y_F = expectation(Y, sigma_algebra=F, probability_measure=P)

        assert np.allclose(E_Y_F.data, Y.data)

    def test_linearity_of_expectation(self, P, X, Y, F):
        """Test linearity of conditional expectation: E(aX + bY|F) = aE(X|F) + bE(Y|F)."""
        a = 3
        b = 5
        E_aX_plus_bY_F = expectation(
            a * X + b * Y, sigma_algebra=F, probability_measure=P
        )
        E_X_F = expectation(X, sigma_algebra=F, probability_measure=P)
        E_Y_F = expectation(Y, sigma_algebra=F, probability_measure=P)
        expected_E_aX_plus_bY_F = a * E_X_F + b * E_Y_F

        assert np.allclose(E_aX_plus_bY_F.data, expected_E_aX_plus_bY_F.data)

    def test_pullout_property_of_conditional_expectation(self, P, X, Y, F):
        """Test pullout property: E(XY|F) = Y E(X|F) if Y is F-measurable."""
        E_X_F = expectation(X, sigma_algebra=F, probability_measure=P)
        E_XY_F = expectation(X * Y, sigma_algebra=F, probability_measure=P)
        expected_E_XY_F = Y * E_X_F

        assert np.allclose(E_XY_F.data, expected_E_XY_F.data)

    def test_independence_and_expectation(self, Omega):
        """Test that if X is independent of F, then E(X|F) = E(X)."""
        P = ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.75**2,  # TT
                1: 0.75 * 0.25,  # TH
                2: 0.75 * 0.25,  # HT
                3: 0.25**2,  # HH
            }
        )
        F = SigmaAlgebra(sample_space=Omega, name="F").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            }
        )
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {
                0: 0,
                1: 1,
                2: 0,
                3: 1,
            }
        )
        E_X_F = expectation(X, sigma_algebra=F, probability_measure=P)
        E_X = expectation(X, probability_measure=P)

        assert np.allclose(E_X_F.data, E_X.data)

    def test_expectation_of_centered_random_variable_is_zero(self, P, X):
        """Test E(X - E(X)) = 0."""
        E_X = expectation(X, probability_measure=P)
        centered_X = X - E_X
        E_centered_X = expectation(centered_X, probability_measure=P)

        assert np.allclose(E_centered_X.data, 0)

    def test_variance_formula_with_squared_expectation(self, P, X):
        """Test V(X) = E(X^2) - E(X)^2."""
        V_X = variance(X, probability_measure=P)
        E_X = expectation(X, probability_measure=P)
        E_X_squared = expectation(X**2, probability_measure=P)

        assert np.allclose(V_X.data, (E_X_squared - E_X**2).data)

    def test_covariance_symmetry(self, P, X, Y):
        """Test Cov(X, Y) = Cov(Y, X)."""
        cov_XY = covariance(X, Y, probability_measure=P)
        cov_YX = covariance(Y, X, probability_measure=P)

        assert abs(cov_XY - cov_YX) < 1e-9

    def test_variance_is_covariance_with_self(self, P, X):
        """Test V(X) = Cov(X, X)."""
        V_X = variance(X, probability_measure=P)
        cov = covariance(X, X, probability_measure=P)

        assert np.allclose(V_X.data, cov)

    def test_covariance_bilinear(self, P, X, Y):
        """Test covariance is bilinear: Cov(aX + bY, Z) = aCov(X, Z) + bCov(Y, Z)."""
        a = 2
        b = 4
        Z = RandomVariable(domain=X.domain, name="Z").from_dict(
            {
                0: 5,
                1: 6,
                2: 7,
                3: 8,
            }
        )
        cov_aX_plus_bY_Z = covariance(a * X + b * Y, Z, probability_measure=P)
        cov_X_Z = covariance(X, Z, probability_measure=P)
        cov_Y_Z = covariance(Y, Z, probability_measure=P)
        expected_cov_aX_plus_bY_Z = a * cov_X_Z + b * cov_Y_Z

        assert np.allclose(cov_aX_plus_bY_Z, expected_cov_aX_plus_bY_Z)
