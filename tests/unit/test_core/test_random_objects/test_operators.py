import numpy as np
import pandas as pd
import pytest

from sigalg.core import (
    Index,
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
    def A(self, Omega):
        return Omega.get_event([0, 1])

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
                2: -1,
            }
        )

    def test_integrate_random_vector_with_prob_measure_parameter(self, X, P, A):
        """Test integration of a 2D random vector with an explicit probability measure."""
        X0, X1 = X.components
        integral = integrate(rv=X, probability_measure=P)
        int_X0 = sum([X0(omega) * P(omega) for omega in X.domain])
        int_X1 = sum([X1(omega) * P(omega) for omega in X.domain])
        expected_int_X = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )
        integral_A = integrate(rv=X, probability_measure=P, event=A)
        int_X0_A = sum([X0(omega) * P(omega) for omega in A])
        int_X1_A = sum([X1(omega) * P(omega) for omega in A])
        expected_int_X_A = pd.Series(
            [int_X0_A, int_X1_A],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )

        pd.testing.assert_series_equal(integral, expected_int_X)
        pd.testing.assert_series_equal(integral_A, expected_int_X_A)

    def test_integrate_random_variable_with_prob_measure_parameter(self, Y, P, A):
        """Test integration of a random variable with an explicit probability measure."""
        integral = integrate(rv=Y, probability_measure=P)
        int_Y = sum(Y(omega) * P(omega) for omega in Y.domain)
        integral_A = integrate(rv=Y, probability_measure=P, event=A)
        int_Y_A = sum(Y(omega) * P(omega) for omega in A)

        assert np.abs(integral - int_Y) < 1e-9
        assert np.abs(integral_A - int_Y_A) < 1e-9

    def test_integrate_random_vector_with_rv_prob_measure(self, X, P, A):
        """Test integration of a 2D random vector using the probability measure carried by the random vector."""
        X.with_probability_measure(probability_measure=P)
        integral = integrate(rv=X)
        X0, X1 = X.components
        int_X0 = sum(X0(omega) * P(omega) for omega in X.domain)
        int_X1 = sum(X1(omega) * P(omega) for omega in X.domain)
        expected_int_X = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )
        integral_A = integrate(rv=X, probability_measure=P, event=A)
        int_X0_A = sum([X0(omega) * P(omega) for omega in A])
        int_X1_A = sum([X1(omega) * P(omega) for omega in A])
        expected_int_X_A = pd.Series(
            [int_X0_A, int_X1_A],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )

        pd.testing.assert_series_equal(integral, expected_int_X)
        pd.testing.assert_series_equal(integral_A, expected_int_X_A)

    def test_integrate_random_variable_with_rv_prob_measure(self, Y, P, A):
        """Test integration of a random variable using the probability measure carried by the random variable."""
        Y.with_probability_measure(probability_measure=P)
        integral = integrate(rv=Y)
        int_Y = sum(Y(omega) * P(omega) for omega in Y.domain)
        integral_A = integrate(rv=Y, event=A)
        int_Y_A = sum(Y(omega) * P(omega) for omega in A)

        assert np.abs(integral - int_Y) < 1e-9
        assert np.abs(integral_A - int_Y_A) < 1e-9

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
        X0, X1 = X.components
        int_X0 = sum(X0(omega) * Q(omega) for omega in X.domain)
        int_X1 = sum(X1(omega) * Q(omega) for omega in X.domain)

        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )

        pd.testing.assert_series_equal(integral, expected_integral)


class TestExpectation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.15,
                2: 0.65,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVariable(domain=Omega).from_dict(
            {
                0: -1,
                1: 2,
                2: 4,
            }
        )

    @pytest.fixture
    def Y(self, Omega):
        return RandomVector(domain=Omega, name="Y").from_dict(
            {
                0: (1, 2),
                1: (-1, 3),
                2: (4, 0),
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
    def G(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
            }
        )

    def test_expectation_random_variable(self, Omega, X, G, P):
        """Test the expectation of a random variable."""
        # Test passing the probability measure
        exp_cond = Operators.expectation(rv=X, sigma_algebra=G, probability_measure=P)
        int = Operators.integrate
        atoms = G.to_atoms()
        I = RandomVariable.indicator_of  # noqa: E741
        expected_exp_cond = sum(
            [(int(X, P, B) / P(B)) * I(B) for B in atoms]
        ).with_name("E(X|G)")

        pd.testing.assert_series_equal(exp_cond.data, expected_exp_cond.data)
        assert exp_cond.name == "E(X|G)"

        # Test setting the probability measure on the random variable
        X.probability_measure = P
        exp_cond = Operators.expectation(rv=X, sigma_algebra=G)

        pd.testing.assert_series_equal(exp_cond.data, expected_exp_cond.data)
        assert exp_cond.name == "E(X|G)"

        # Test unconditional expectation
        exp_uncond = Operators.expectation(rv=X)
        expected_exp_uncond = (
            RandomVariable(domain=Omega).from_constant(int(X)).with_name("E(X)")
        )

        pd.testing.assert_series_equal(exp_uncond.data, expected_exp_uncond.data)
        assert exp_uncond.name == "E(X)"

    def test_expectation_random_vector(self, Omega, Y, G, P):
        """Test the expectation of a random variable."""
        # Test passing the probability measure
        exp_cond = Operators.expectation(rv=Y, sigma_algebra=G, probability_measure=P)
        int = Operators.integrate
        atoms = G.to_atoms()
        I = RandomVariable.indicator_of  # noqa: E741

        Y0, Y1 = Y.components
        expected_exp_cond_Y0 = sum(
            [(int(Y0, P, B) / P(B)) * I(B) for B in atoms]
        ).with_name("E(Y_0|G)")
        expected_exp_cond_Y1 = sum(
            [(int(Y1, P, B) / P(B)) * I(B) for B in atoms]
        ).with_name("E(Y_1|G)")

        expected_data = pd.concat(
            [expected_exp_cond_Y0.data, expected_exp_cond_Y1.data], axis=1
        )
        expected_data.columns.name = "expectation"

        pd.testing.assert_frame_equal(exp_cond.data, expected_data)
        assert exp_cond.name == "E(Y|G)"

        # Test setting the probability measure on the random vector
        Y.probability_measure = P
        exp_cond = Operators.expectation(rv=Y, sigma_algebra=G)

        pd.testing.assert_frame_equal(exp_cond.data, expected_data)
        assert exp_cond.name == "E(Y|G)"

        # Test unconditional expectation
        exp_uncond = Operators.expectation(rv=Y)
        expected_exp_uncond = (
            RandomVector(domain=Omega).from_constant(tuple(int(Y))).with_name("E(Y)")
        )
        expected_exp_uncond.index = Index(
            name="index", data_name="expectation"
        ).from_list(["E(Y_0)", "E(Y_1)"])

        pd.testing.assert_frame_equal(exp_uncond.data, expected_exp_uncond.data)
        assert exp_uncond.name == "E(Y)"

    def test_sum_of_atom_expectations_formula(self, X, Y, G, P):
        """Test whether the conditional expectation is the linear combination of the indicator functions of the atoms with weights given by restricted expectations."""
        exp = Operators.expectation
        I = RandomVariable.indicator_of  # noqa: E741

        # Test for random variable
        X.probability_measure = P
        exp_cond = exp(rv=X, sigma_algebra=G)

        exp_linear_combo = sum(
            [exp(X(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("E(X|G)")

        pd.testing.assert_series_equal(exp_cond.data, exp_linear_combo.data)
        assert exp_cond.name == "E(X|G)"

        # Test for random vector
        Y.probability_measure = P
        Y0, Y1 = Y.components
        exp_cond = exp(rv=Y, sigma_algebra=G)

        exp_linear_combo_Y0 = sum(
            [exp(Y0(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("E(Y_0|G)")
        exp_linear_combo_Y1 = sum(
            [exp(Y1(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("E(Y_1|G)")
        expected_data = pd.concat(
            [exp_linear_combo_Y0.data, exp_linear_combo_Y1.data], axis=1
        )
        expected_data.columns.name = "expectation"

        pd.testing.assert_frame_equal(exp_cond.data, expected_data)
        assert exp_cond.name == "E(Y|G)"

    def test_linearity_of_expectation(self, X, Z, P, G):
        """Test the linearity of expectation."""
        a = 2
        b = -3
        X.probability_measure = P
        Z.probability_measure = P
        exp = Operators.expectation

        assert exp(a * X + b * Z, G) == a * exp(X, G) + b * exp(Z, G)

    def test_factoring_out_measurable_functions(self, Omega, X, G, P):
        """Test that functions measurable with respect to the sigma-algebra can be factored out of the expectation."""

        C = RandomVariable(domain=Omega, name="C").from_dict(
            {
                0: 2,
                1: -1,
                2: -1,
            }
        )
        C.probability_measure = P
        X.probability_measure = P
        exp = Operators.expectation

        assert exp(C * X, G) == C * exp(X, G)

    def test_iterated_expectation(self):
        """Test the law of iterated expectation."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            }
        )
        G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 1,
            }
        )
        P = ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.1,
                1: 0.15,
                2: 0.25,
                3: 0.5,
            }
        )
        X = RandomVariable(domain=Omega).from_dict(
            {
                0: -1,
                1: 2,
                2: -3,
                3: 2,
            }
        )

        X.probability_measure = P
        exp = Operators.expectation

        assert G < F
        assert exp(exp(X, F), G) == exp(X, G)

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
                1: 0.15,
                2: 0.65,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVariable(domain=Omega).from_dict(
            {
                0: -1,
                1: 2,
                2: 4,
            }
        )

    @pytest.fixture
    def Y(self, Omega):
        return RandomVector(domain=Omega, name="Y").from_dict(
            {
                0: (1, 2),
                1: (-1, 3),
                2: (4, 0),
            }
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
            }
        )

    def test_variance_random_variable(self, Omega, X, G, P):
        """Test the variance of a random variable."""
        # Test passing the probability measure
        var_cond = Operators.variance(rv=X, sigma_algebra=G, probability_measure=P)
        exp = Operators.expectation
        expected_var = exp((X - exp(X, G, P)) ** 2, G, P).with_name("V(X|G)")

        pd.testing.assert_series_equal(var_cond.data, expected_var.data)
        assert var_cond.name == "V(X|G)"

        # Test setting the probability measure on the random variable
        X.probability_measure = P
        var_cond = Operators.variance(rv=X, sigma_algebra=G)

        pd.testing.assert_series_equal(var_cond.data, expected_var.data)
        assert var_cond.name == "V(X|G)"

        # Test unconditional expectation
        exp_uncond = Operators.variance(rv=X)
        var_X = exp((X - exp(X)) ** 2).item()
        expected_var_uncond = (
            RandomVariable(domain=Omega).from_constant(var_X).with_name("V(X)")
        )

        pd.testing.assert_series_equal(exp_uncond.data, expected_var_uncond.data)
        assert exp_uncond.name == "V(X)"

    def test_variance_random_vector(self, Omega, Y, G, P):
        """Test the expectation of a random variable."""
        # Test passing the probability measure
        var_cond = Operators.variance(rv=Y, sigma_algebra=G, probability_measure=P)
        exp = Operators.expectation

        Y0, Y1 = Y.components
        expected_var_cond_Y0 = exp((Y0 - exp(Y0, G, P)) ** 2, G, P).with_name(
            "V(Y_0|G)"
        )
        expected_var_cond_Y1 = exp((Y1 - exp(Y1, G, P)) ** 2, G, P).with_name(
            "V(Y_1|G)"
        )

        expected_data = pd.concat(
            [expected_var_cond_Y0.data, expected_var_cond_Y1.data], axis=1
        )
        expected_data.columns.name = "variance"

        pd.testing.assert_frame_equal(var_cond.data, expected_data)
        assert var_cond.name == "V(Y|G)"

        # Test setting the probability measure on the random vector
        Y.probability_measure = P
        var_cond = Operators.variance(rv=Y, sigma_algebra=G)

        pd.testing.assert_frame_equal(var_cond.data, expected_data)
        assert var_cond.name == "V(Y|G)"

        # Test unconditional expectation
        var_uncond = Operators.variance(rv=Y)
        var_Y = exp((Y - exp(Y)) ** 2).item()
        expected_var_uncond = (
            RandomVector(domain=Omega).from_constant(tuple(var_Y)).with_name("V(Y)")
        )
        expected_var_uncond.index = Index(name="index", data_name="variance").from_list(
            ["V(Y_0)", "V(Y_1)"]
        )

        pd.testing.assert_frame_equal(var_uncond.data, expected_var_uncond.data)
        assert var_uncond.name == "V(Y)"

    def test_total_variance(self):
        """Test the law of total variance."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            }
        )
        P = ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.1,
                1: 0.15,
                2: 0.25,
                3: 0.5,
            }
        )
        X = RandomVariable(domain=Omega).from_dict(
            {
                0: -1,
                1: 2,
                2: -3,
                3: 2,
            }
        )

        X.probability_measure = P
        exp = Operators.expectation
        var = Operators.variance

        assert var(X) == exp(var(X, F)) + var(exp(X, F))

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
