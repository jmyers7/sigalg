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
        integral = Operators.integrate(rv=X, probability_measure=P)
        int_X0 = sum([X0(omega) * P(omega) for omega in X.domain])
        int_X1 = sum([X1(omega) * P(omega) for omega in X.domain])
        expected_int_X = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )
        integral_A = Operators.integrate(rv=X, probability_measure=P, event=A)
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
        integral = Operators.integrate(rv=Y, probability_measure=P)
        int_Y = sum(Y(omega) * P(omega) for omega in Y.domain)
        integral_A = Operators.integrate(rv=Y, probability_measure=P, event=A)
        int_Y_A = sum(Y(omega) * P(omega) for omega in A)

        assert np.abs(integral - int_Y) < 1e-9
        assert np.abs(integral_A - int_Y_A) < 1e-9

    def test_integrate_random_vector_with_rv_prob_measure(self, X, P, A):
        """Test integration of a 2D random vector using the probability measure carried by the random vector."""
        X.with_probability_measure(probability_measure=P)
        integral = Operators.integrate(rv=X)
        X0, X1 = X.components
        int_X0 = sum(X0(omega) * P(omega) for omega in X.domain)
        int_X1 = sum(X1(omega) * P(omega) for omega in X.domain)
        expected_int_X = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )
        integral_A = Operators.integrate(rv=X, probability_measure=P, event=A)
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
        integral = Operators.integrate(rv=Y)
        int_Y = sum(Y(omega) * P(omega) for omega in Y.domain)
        integral_A = Operators.integrate(rv=Y, event=A)
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
        integral = Operators.integrate(rv=X, probability_measure=Q)
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
        """Test the expectation of a random vector."""
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

    def test_expectation_of_measurable_random_variable_equals_itself(self, Omega, G, P):
        """Test E(X|G) = X if X is G-measurable."""
        exp = Operators.expectation
        W = RandomVector(domain=Omega, name="W").from_dict(
            {
                0: (1, 2),
                1: (3, 4),
                2: (3, 4),
            }
        )
        W.probability_measure = P

        assert exp(W, G) == W

    def test_independence_and_expectation(self):
        """Test that if X is independent of F, then E(X|F) = E(X)."""
        exp = Operators.expectation

        Omega = SampleSpace().from_sequence(size=4)
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
        X.probability_measure = P

        assert exp(X) == exp(X, F)

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
            Operators.expectation("not a random vector")

    def test_expectation_invalid_sigma_algebra_type_raises(self):
        """Test that invalid sigma algebra type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})
        P = ProbabilityMeasure.uniform(domain)

        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            Operators.expectation(
                X, sigma_algebra="not a sigma algebra", probability_measure=P
            )

    def test_expectation_invalid_probability_measure_type_raises(self):
        """Test that invalid probability measure type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})

        with pytest.raises(
            TypeError, match="probability_measure must be a ProbabilityMeasure"
        ):
            Operators.expectation(X, probability_measure="not a probability measure")


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
        """Test the variance of a random vector."""
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

    def test_sum_of_atom_variances_formula(self, X, Y, G, P):
        """Test whether the conditional variance is the linear combination of the indicator functions of the atoms with weights given by restricted variances."""
        var = Operators.variance
        I = RandomVariable.indicator_of  # noqa: E741

        # Test for random variable
        X.probability_measure = P
        var_cond = var(rv=X, sigma_algebra=G)

        var_linear_combo = sum(
            [var(X(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("V(X|G)")

        pd.testing.assert_series_equal(var_cond.data, var_linear_combo.data)
        assert var_cond.name == "V(X|G)"

        # Test for random vector
        Y.probability_measure = P
        Y0, Y1 = Y.components
        var_cond = var(rv=Y, sigma_algebra=G)

        var_linear_combo_Y0 = sum(
            [var(Y0(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("V(Y_0|G)")
        var_linear_combo_Y1 = sum(
            [var(Y1(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("V(Y_1|G)")
        expected_data = pd.concat(
            [var_linear_combo_Y0.data, var_linear_combo_Y1.data], axis=1
        )
        expected_data.columns.name = "variance"

        pd.testing.assert_frame_equal(var_cond.data, expected_data)
        assert var_cond.name == "V(Y|G)"

    def test_variance_formula_with_squared_expectation(self, P, X, G):
        """Test V(X) = E(X^2) - E(X)^2."""
        X.probability_measure = P
        var = Operators.variance
        exp = Operators.expectation

        assert var(X, G) == exp(X**2, G) - exp(X, G) ** 2

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
            Operators.variance("not a random vector")

    def test_variance_invalid_sigma_algebra_type_raises(self):
        """Test that invalid sigma algebra type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})
        P = ProbabilityMeasure.uniform(domain)

        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            Operators.variance(
                X, sigma_algebra="not a sigma algebra", probability_measure=P
            )


class TestStandardDeviation:
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

    def test_std_random_variable(self, Omega, X, G, P):
        """Test the standard deviation of a random variable."""
        # Test passing the probability measure
        std = Operators.std
        std_cond = std(rv=X, sigma_algebra=G, probability_measure=P)
        expected_std = (
            Operators.variance(rv=X, sigma_algebra=G, probability_measure=P) ** 0.5
        ).with_name("std(X|G)")

        pd.testing.assert_series_equal(std_cond.data, expected_std.data)
        assert std_cond.name == "std(X|G)"

        # Test setting the probability measure on the random variable
        X.probability_measure = P
        std_cond = std(rv=X, sigma_algebra=G)

        pd.testing.assert_series_equal(std_cond.data, expected_std.data)
        assert std_cond.name == "std(X|G)"

        # Test unconditional expectation
        std_uncond = std(rv=X)
        std_X = (Operators.variance(X) ** 0.5).item()
        expected_std_uncond = (
            RandomVariable(domain=Omega).from_constant(std_X).with_name("std(X)")
        )

        pd.testing.assert_series_equal(std_uncond.data, expected_std_uncond.data)
        assert std_uncond.name == "std(X)"

    def test_std_random_vector(self, Omega, Y, G, P):
        """Test the standard deviation of a random vector."""
        # Test passing the probability measure
        std = Operators.std
        std_cond = std(rv=Y, sigma_algebra=G, probability_measure=P)

        Y0, Y1 = Y.components
        expected_std_cond_Y0 = (Operators.variance(Y0, G, P) ** 0.5).with_name(
            "std(Y_0|G)"
        )
        expected_std_cond_Y1 = (Operators.variance(Y1, G, P) ** 0.5).with_name(
            "std(Y_1|G)"
        )

        expected_data = pd.concat(
            [expected_std_cond_Y0.data, expected_std_cond_Y1.data], axis=1
        )
        expected_data.columns.name = "std"

        pd.testing.assert_frame_equal(std_cond.data, expected_data)
        assert std_cond.name == "std(Y|G)"

        # Test setting the probability measure on the random vector
        Y.probability_measure = P
        std_cond = std(rv=Y, sigma_algebra=G)

        pd.testing.assert_frame_equal(std_cond.data, expected_data)
        assert std_cond.name == "std(Y|G)"

        # Test unconditional expectation
        std_uncond = std(rv=Y)
        std_Y = (Operators.variance(Y) ** 0.5).item()
        expected_std_uncond = (
            RandomVector(domain=Omega).from_constant(tuple(std_Y)).with_name("std(Y)")
        )
        expected_std_uncond.index = Index(name="index", data_name="std").from_list(
            ["std(Y_0)", "std(Y_1)"]
        )

        pd.testing.assert_frame_equal(std_uncond.data, expected_std_uncond.data)
        assert std_uncond.name == "std(Y)"

    def test_squared_std_equals_variance(self, X, G, P):
        """Test that std(X|G)^2 = V(X|G)."""
        std = Operators.std
        var = Operators.variance
        X.probability_measure = P

        assert std(X, G) ** 2 == var(X, G)

    def test_sum_of_atom_std_formula(self, X, Y, G, P):
        """Test whether the conditional standard deviation is the linear combination of the indicator functions of the atoms with weights given by restricted standard deviations."""
        std = Operators.std
        I = RandomVariable.indicator_of  # noqa: E741

        # Test for random variable
        X.probability_measure = P
        std_cond = std(rv=X, sigma_algebra=G)

        std_linear_combo = sum(
            [std(X(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("std(X|G)")

        pd.testing.assert_series_equal(std_cond.data, std_linear_combo.data)
        assert std_cond.name == "std(X|G)"

        # Test for random vector
        Y.probability_measure = P
        Y0, Y1 = Y.components
        std_cond = std(rv=Y, sigma_algebra=G)

        std_linear_combo_Y0 = sum(
            [std(Y0(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("std(Y_0|G)")
        std_linear_combo_Y1 = sum(
            [std(Y1(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("std(Y_1|G)")
        expected_data = pd.concat(
            [std_linear_combo_Y0.data, std_linear_combo_Y1.data], axis=1
        )
        expected_data.columns.name = "std"

        pd.testing.assert_frame_equal(std_cond.data, expected_data)
        assert std_cond.name == "std(Y|G)"

    def test_std_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            Operators.std("not a random vector")

    def test_std_invalid_sigma_algebra_type_raises(self):
        """Test that invalid sigma algebra type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})
        P = ProbabilityMeasure.uniform(domain)

        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            Operators.std(X, sigma_algebra="not a sigma algebra", probability_measure=P)


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
        cov = Operators.covariance(X, Y, probability_measure=P)
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
        cov = Operators.covariance(Z, W, probability_measure=P)
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
        cov = Operators.covariance(X, probability_measure=P)
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
        cov = Operators.covariance(Z, probability_measure=P)
        E_Z = 0.2 * 1 + 0.3 * (-2) + 0.5 * 3
        expected_cov = (
            0.2 * (1 - E_Z) ** 2 + 0.3 * (-2 - E_Z) ** 2 + 0.5 * (3 - E_Z) ** 2
        )

        assert abs(cov - expected_cov) < 1e-9

    def test_covariance_invalid_rv1_type_raises(self):
        """Test that invalid rv1 type raises TypeError."""
        with pytest.raises(TypeError, match="rv1 must be a RandomVector"):
            Operators.covariance("not a random vector")

    def test_covariance_invalid_rv2_type_raises(self):
        """Test that invalid rv2 type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})
        P = ProbabilityMeasure.uniform(domain)

        with pytest.raises(TypeError, match="rv2 must be a RandomVector"):
            Operators.covariance(X, rv2="not a random vector", probability_measure=P)

    def test_covariance_different_domains_raises(self):
        """Test that random vectors with different domains raise ValueError."""
        Omega1 = SampleSpace().from_sequence(size=2, prefix="s")
        Omega2 = SampleSpace().from_sequence(size=2, prefix="t")
        X = RandomVariable(domain=Omega1, name="X").from_dict({"s_0": 1, "s_1": 2})
        Y = RandomVariable(domain=Omega2, name="Y").from_dict({"t_0": 3, "t_1": 4})
        P = ProbabilityMeasure.uniform(Omega1)

        with pytest.raises(ValueError, match="same domain"):
            Operators.covariance(X, Y, probability_measure=P)

    def test_covariance_different_dimensions_raises(self):
        """Test that random vectors with different dimensions raise ValueError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        )
        Z = RandomVariable(domain=Omega, name="Z").from_dict({0: 1, 1: 2, 2: 3})
        P = ProbabilityMeasure.uniform(Omega)

        with pytest.raises(ValueError, match="same dimension"):
            Operators.covariance(X, Z, probability_measure=P)


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
        corr = Operators.correlation(X, Y, probability_measure=P)
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
        corr = Operators.correlation(Z, W, probability_measure=P)
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
        corr = Operators.correlation(X, Y, probability_measure=P)

        assert abs(corr - 1.0) < 1e-9

    def test_correlation_equals_negative_one_for_perfect_negative_correlation(
        self, Omega, P
    ):
        """Test correlation equals -1 for perfectly negatively correlated variables."""
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 6, 1: 4, 2: 2})
        corr = Operators.correlation(X, Y, probability_measure=P)

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

    def test_covariance_symmetry(self, P, X, Y):
        """Test Cov(X, Y) = Cov(Y, X)."""
        cov_XY = Operators.covariance(X, Y, probability_measure=P)
        cov_YX = Operators.covariance(Y, X, probability_measure=P)

        assert abs(cov_XY - cov_YX) < 1e-9

    def test_variance_is_covariance_with_self(self, P, X):
        """Test V(X) = Cov(X, X)."""
        V_X = Operators.variance(X, probability_measure=P)
        cov = Operators.covariance(X, X, probability_measure=P)

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
        cov_aX_plus_bY_Z = Operators.covariance(a * X + b * Y, Z, probability_measure=P)
        cov_X_Z = Operators.covariance(X, Z, probability_measure=P)
        cov_Y_Z = Operators.covariance(Y, Z, probability_measure=P)
        expected_cov_aX_plus_bY_Z = a * cov_X_Z + b * cov_Y_Z

        assert np.allclose(cov_aX_plus_bY_Z, expected_cov_aX_plus_bY_Z)
