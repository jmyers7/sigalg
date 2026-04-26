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
        integral = Operators.integrate(rv=X, prob_measure=P)
        int_X0 = sum([X0(omega) * P(omega) for omega in X.domain])
        int_X1 = sum([X1(omega) * P(omega) for omega in X.domain])
        expected_int_X = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )
        integral_A = Operators.integrate(rv=X, prob_measure=P, event=A)
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
        integral = Operators.integrate(rv=Y, prob_measure=P)
        int_Y = sum(Y(omega) * P(omega) for omega in Y.domain)
        integral_A = Operators.integrate(rv=Y, prob_measure=P, event=A)
        int_Y_A = sum(Y(omega) * P(omega) for omega in A)

        assert np.abs(integral - int_Y) < 1e-9
        assert np.abs(integral_A - int_Y_A) < 1e-9

    def test_integrate_random_vector_with_rv_prob_measure(self, X, P, A):
        """Test integration of a 2D random vector using the probability measure carried by the random vector."""
        X.with_probability_measure(prob_measure=P)
        integral = Operators.integrate(rv=X)
        X0, X1 = X.components
        int_X0 = sum(X0(omega) * P(omega) for omega in X.domain)
        int_X1 = sum(X1(omega) * P(omega) for omega in X.domain)
        expected_int_X = pd.Series(
            [int_X0, int_X1],
            index=pd.Index(["integral(X_0)", "integral(X_1)"], name="integral"),
            name="integral(X)",
        )
        integral_A = Operators.integrate(rv=X, prob_measure=P, event=A)
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
        Y.with_probability_measure(prob_measure=P)
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
        X.with_probability_measure(prob_measure=P)
        integral = Operators.integrate(rv=X, prob_measure=Q)
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
        exp_cond = Operators.expectation(rv=X, sig_alg=G, prob_measure=P)
        int = Operators.integrate
        atoms = G.to_atoms()
        I = RandomVariable.indicator_of
        expected_exp_cond = sum(
            [(int(X, P, B) / P(B)) * I(B) for B in atoms]
        ).with_name("E(X|G)")

        pd.testing.assert_series_equal(exp_cond.data, expected_exp_cond.data)
        assert exp_cond.name == "E(X|G)"

        # Test setting the probability measure on the random variable
        X.prob_measure = P
        exp_cond = Operators.expectation(rv=X, sig_alg=G)

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
        exp_cond = Operators.expectation(rv=Y, sig_alg=G, prob_measure=P)
        int = Operators.integrate
        atoms = G.to_atoms()
        I = RandomVariable.indicator_of

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
        Y.prob_measure = P
        exp_cond = Operators.expectation(rv=Y, sig_alg=G)

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
        I = RandomVariable.indicator_of

        # Test for random variable
        X.prob_measure = P
        exp_cond = exp(rv=X, sig_alg=G)

        exp_linear_combo = sum(
            [exp(X(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("E(X|G)")

        pd.testing.assert_series_equal(exp_cond.data, exp_linear_combo.data)
        assert exp_cond.name == "E(X|G)"

        # Test for random vector
        Y.prob_measure = P
        Y0, Y1 = Y.components
        exp_cond = exp(rv=Y, sig_alg=G)

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
        X.prob_measure = P
        Z.prob_measure = P
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
        C.prob_measure = P
        X.prob_measure = P
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
        W.prob_measure = P

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
        X.prob_measure = P

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

        X.prob_measure = P
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

        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra"):
            Operators.expectation(
                X, sig_alg="not a sigma algebra", prob_measure=P
            )

    def test_expectation_invalid_probability_measure_type_raises(self):
        """Test that invalid probability measure type raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=domain, name="X").from_dict({0: 1, 1: 2})

        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure"
        ):
            Operators.expectation(X, prob_measure="not a probability measure")


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
        var_cond = Operators.variance(rv=X, sig_alg=G, prob_measure=P)
        exp = Operators.expectation
        expected_var = exp((X - exp(X, G, P)) ** 2, G, P).with_name("V(X|G)")

        pd.testing.assert_series_equal(var_cond.data, expected_var.data)
        assert var_cond.name == "V(X|G)"

        # Test setting the probability measure on the random variable
        X.prob_measure = P
        var_cond = Operators.variance(rv=X, sig_alg=G)

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
        var_cond = Operators.variance(rv=Y, sig_alg=G, prob_measure=P)
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
        Y.prob_measure = P
        var_cond = Operators.variance(rv=Y, sig_alg=G)

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
        I = RandomVariable.indicator_of

        # Test for random variable
        X.prob_measure = P
        var_cond = var(rv=X, sig_alg=G)

        var_linear_combo = sum(
            [var(X(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("V(X|G)")

        pd.testing.assert_series_equal(var_cond.data, var_linear_combo.data)
        assert var_cond.name == "V(X|G)"

        # Test for random vector
        Y.prob_measure = P
        Y0, Y1 = Y.components
        var_cond = var(rv=Y, sig_alg=G)

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
        X.prob_measure = P
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

        X.prob_measure = P
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

        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra"):
            Operators.variance(
                X, sig_alg="not a sigma algebra", prob_measure=P
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
        std_cond = std(rv=X, sig_alg=G, prob_measure=P)
        expected_std = (
            Operators.variance(rv=X, sig_alg=G, prob_measure=P) ** 0.5
        ).with_name("std(X|G)")

        pd.testing.assert_series_equal(std_cond.data, expected_std.data)
        assert std_cond.name == "std(X|G)"

        # Test setting the probability measure on the random variable
        X.prob_measure = P
        std_cond = std(rv=X, sig_alg=G)

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
        std_cond = std(rv=Y, sig_alg=G, prob_measure=P)

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
        Y.prob_measure = P
        std_cond = std(rv=Y, sig_alg=G)

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
        X.prob_measure = P

        assert std(X, G) ** 2 == var(X, G)

    def test_sum_of_atom_std_formula(self, X, Y, G, P):
        """Test whether the conditional standard deviation is the linear combination of the indicator functions of the atoms with weights given by restricted standard deviations."""
        std = Operators.std
        I = RandomVariable.indicator_of

        # Test for random variable
        X.prob_measure = P
        std_cond = std(rv=X, sig_alg=G)

        std_linear_combo = sum(
            [std(X(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("std(X|G)")

        pd.testing.assert_series_equal(std_cond.data, std_linear_combo.data)
        assert std_cond.name == "std(X|G)"

        # Test for random vector
        Y.prob_measure = P
        Y0, Y1 = Y.components
        std_cond = std(rv=Y, sig_alg=G)

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

        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra"):
            Operators.std(X, sig_alg="not a sigma algebra", prob_measure=P)


class TestCovariance:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=5)

    @pytest.fixture
    def P(self, Omega):
        rng = np.random.default_rng(42)
        return ProbabilityMeasure(sample_space=Omega).from_rand(random_state=rng)

    @pytest.fixture
    def X(self, Omega):
        rng = np.random.default_rng(42)
        return RandomVariable(domain=Omega).from_randint(
            low=-20, high=21, random_state=rng
        )

    @pytest.fixture
    def Y(self, Omega):
        rng = np.random.default_rng(43)
        return RandomVariable(domain=Omega, name="Y").from_randint(
            low=-10, high=11, random_state=rng
        )

    @pytest.fixture
    def Z(self, Omega):
        rng = np.random.default_rng(44)
        return RandomVariable(domain=Omega, name="Z").from_randint(
            low=-20, high=21, random_state=rng
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
            }
        )

    def test_covariance_with_prob_measure_parameter(self, X, Y, P):
        """Test covariance of two random variables with an explicit probability measure."""
        cov = Operators.cov
        exp = Operators.expectation
        covar = cov(X, Y, prob_measure=P)
        expected_covar = (
            exp(X * Y, prob_measure=P)
            - exp(X, prob_measure=P) * exp(Y, prob_measure=P)
        ).with_name("cov(X, Y)")

        pd.testing.assert_series_equal(covar.data, expected_covar.data)
        assert covar.name == "cov(X, Y)"

    def test_covariance_with_rv_prob_measure(self, X, Y, P):
        """Test covariance using the probability measure carried by the random variables."""
        cov = Operators.cov
        exp = Operators.expectation
        X.prob_measure = P
        Y.prob_measure = P
        covar = cov(X, Y)
        expected_covar = (exp(X * Y) - exp(X) * exp(Y)).with_name("cov(X, Y)")

        pd.testing.assert_series_equal(covar.data, expected_covar.data)
        assert covar.name == "cov(X, Y)"

    def test_conditional_covariance(self, X, Y, G, P):
        """Test conditional covariance with respect to a sigma-algebra."""
        cov = Operators.cov
        exp = Operators.expectation
        X.prob_measure = P
        Y.prob_measure = P
        covar_cond = cov(X, Y, G)
        expected_covar_cond = (exp(X * Y, G) - exp(X, G) * exp(Y, G)).with_name(
            "cov(X, Y|G)"
        )

        pd.testing.assert_series_equal(covar_cond.data, expected_covar_cond.data)
        assert covar_cond.name == "cov(X, Y|G)"

    def test_sum_of_atom_covariances_formula(self, X, Y, G, P):
        """Test whether the conditional covariance is the linear combination of the indicator functions of the atoms with weights given by restricted covariances."""
        cov = Operators.cov
        I = RandomVariable.indicator_of
        X.prob_measure = P
        Y.prob_measure = P
        covar_cond = cov(X, Y, G)

        covar_linear_combo = sum(
            [cov(X(atom), Y(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("cov(X, Y|G)")

        pd.testing.assert_series_equal(covar_cond.data, covar_linear_combo.data)
        assert covar_cond.name == "cov(X, Y|G)"

    def test_alternate_formula_for_covariance(self, X, Y, G, P):
        """Test the alternate formula cov(X, Y|G) = E[(X - E(X|G))(Y - E(Y|G))|G]."""
        cov = Operators.cov
        exp = Operators.expectation
        X.prob_measure = P
        Y.prob_measure = P

        covar = cov(X, Y, G)
        alternate = exp((X - exp(X, G)) * (Y - exp(Y, G)), G).with_name("cov(X, Y|G)")

        pd.testing.assert_series_equal(covar.data, alternate.data)

    def test_symmetry_of_covariance(self, X, Y, G, P):
        """Test that cov(X, Y|G) = cov(Y, X|G)."""
        cov = Operators.cov
        X.prob_measure = P
        Y.prob_measure = P

        assert cov(X, Y, G) == cov(Y, X, G)

    def test_bilinearity_of_covariance(self, X, Y, Z, G, P):
        """Test the bilinearity property of covariance."""
        a = 3
        cov = Operators.cov
        X.prob_measure = P
        Y.prob_measure = P
        Z.prob_measure = P

        assert cov(a * X + Y, Z, G) == a * cov(X, Z, G) + cov(Y, Z, G)

    def test_covariance_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv1 and rv2 must be RandomVariables"):
            Operators.cov("not a random variable", "also not")

    def test_covariance_different_domains_raises(self, Omega):
        """Test that random variables with different domains raise ValueError."""
        Omega2 = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega).from_dict({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        Y = RandomVariable(domain=Omega2, name="Y").from_dict({0: 1, 1: 2, 2: 3})

        with pytest.raises(ValueError, match="rv1 and rv2 must have the same domain"):
            Operators.cov(X, Y)

    def test_covariance_mismatched_probability_measures_raises(self, Omega):
        """Test that mismatched probability measures raise ValueError when not explicitly passed."""
        P1 = ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        )
        P2 = ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2}
        )
        X = RandomVariable(domain=Omega).from_dict({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        )
        X.prob_measure = P1
        Y.prob_measure = P2

        with pytest.raises(
            ValueError,
            match="If prob_measure is not passed, then the probability measures on the random variables will be used. But they are not equal.",
        ):
            Operators.cov(X, Y)


class TestCorrelation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=5)

    @pytest.fixture
    def P(self, Omega):
        rng = np.random.default_rng(42)
        return ProbabilityMeasure(sample_space=Omega).from_rand(random_state=rng)

    @pytest.fixture
    def X(self, Omega):
        rng = np.random.default_rng(42)
        return RandomVariable(domain=Omega).from_randint(
            low=-20, high=21, random_state=rng
        )

    @pytest.fixture
    def Y(self, Omega):
        rng = np.random.default_rng(43)
        return RandomVariable(domain=Omega, name="Y").from_randint(
            low=-10, high=11, random_state=rng
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
            }
        )

    def test_correlation_with_prob_measure_parameter(self, X, Y, P):
        """Test correlation of two random variables with an explicit probability measure."""
        corr = Operators.corr
        cov = Operators.cov
        std = Operators.std
        correlation = corr(X, Y, prob_measure=P)
        expected_correlation = (
            cov(X, Y, prob_measure=P)
            / (std(X, prob_measure=P) * std(Y, prob_measure=P))
        ).with_name("corr(X, Y)")

        pd.testing.assert_series_equal(correlation.data, expected_correlation.data)
        assert correlation.name == "corr(X, Y)"

    def test_correlation_with_rv_prob_measure(self, X, Y, P):
        """Test correlation using the probability measure carried by the random variables."""
        corr = Operators.corr
        cov = Operators.cov
        std = Operators.std
        X.prob_measure = P
        Y.prob_measure = P
        correlation = corr(X, Y)
        expected_correlation = (cov(X, Y) / (std(X) * std(Y))).with_name("corr(X, Y)")

        pd.testing.assert_series_equal(correlation.data, expected_correlation.data)
        assert correlation.name == "corr(X, Y)"

    def test_conditional_correlation(self, X, Y, G, P):
        """Test conditional correlation with respect to a sigma-algebra."""
        corr = Operators.corr
        cov = Operators.cov
        std = Operators.std
        X.prob_measure = P
        Y.prob_measure = P
        correlation_cond = corr(X, Y, G)
        expected_correlation_cond = (cov(X, Y, G) / (std(X, G) * std(Y, G))).with_name(
            "corr(X, Y|G)"
        )

        pd.testing.assert_series_equal(
            correlation_cond.data, expected_correlation_cond.data
        )
        assert correlation_cond.name == "corr(X, Y|G)"

    def test_sum_of_atom_correlations_formula(self, X, Y, G, P):
        """Test whether the conditional correlation is the linear combination of the indicator functions of the atoms with weights given by restricted correlations."""
        corr = Operators.corr
        I = RandomVariable.indicator_of
        X.prob_measure = P
        Y.prob_measure = P
        correlation_cond = corr(X, Y, G)

        corr_linear_combo = sum(
            [corr(X(atom), Y(atom)).item() * I(atom) for atom in G.to_atoms()]
        ).with_name("corr(X, Y|G)")

        pd.testing.assert_series_equal(correlation_cond.data, corr_linear_combo.data)
        assert correlation_cond.name == "corr(X, Y|G)"

    def test_perfectly_correlated_random_variables(self):
        """Test that perfectly correlated random variables have correlation ±1."""
        rng = np.random.default_rng(42)
        Omega = SampleSpace().from_sequence(size=4)
        X = RandomVariable(domain=Omega).from_dict(
            {
                0: -1,  # on the line y = x
                1: 1,  # on the line y = x
                2: -1,  # on the line y = -x
                3: 1,  # on the line y = -x
            }
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {
                0: -1,  # on the line y = x
                1: 1,  # on the line y = x
                2: 1,  # on the line y = -x
                3: -1,  # on the line y = -x
            }
        )
        P = ProbabilityMeasure(sample_space=Omega).from_rand(random_state=rng)
        X.prob_measure = P
        Y.prob_measure = P

        G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            }
        )

        corr = Operators.corr
        correlation = corr(X, Y, G)

        for omega in Omega:
            assert np.abs(np.abs(correlation(omega)) - 1.0) < 1e-9

    def test_independence_implies_uncorrelated(self):
        """Test that independent random variables are uncorrelated."""
        from scipy.stats import bernoulli

        from sigalg.core import Time
        from sigalg.processes import IIDProcess

        coin_flip = IIDProcess(
            distribution=bernoulli(p=0.7),
            support=[0, 1],
            time=Time.discrete(length=1),
            name="coin_flip",
        ).from_enumeration()

        X, Y = coin_flip
        X.with_name("X")
        Y.with_name("Y")

        corr = Operators.corr
        correlation = corr(X, Y)

        assert np.abs(correlation.item()) < 1e-9

    def test_correlation_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv1 and rv2 must be RandomVariables"):
            Operators.corr("not a random variable", "also not")

    def test_correlation_different_domains_raises(self, Omega):
        """Test that random variables with different domains raise ValueError."""
        Omega2 = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega).from_dict({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        Y = RandomVariable(domain=Omega2, name="Y").from_dict({0: 1, 1: 2, 2: 3})

        with pytest.raises(ValueError, match="rv1 and rv2 must have the same domain"):
            Operators.corr(X, Y)

    def test_correlation_mismatched_probability_measures_raises(self, Omega):
        """Test that mismatched probability measures raise ValueError when not explicitly passed."""
        P1 = ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
        )
        P2 = ProbabilityMeasure(sample_space=Omega).from_dict(
            {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2}
        )
        X = RandomVariable(domain=Omega).from_dict({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        )
        X.prob_measure = P1
        Y.prob_measure = P2

        with pytest.raises(
            ValueError,
            match="If prob_measure is not passed, then the probability measures on the random variables will be used. But they are not equal.",
        ):
            Operators.corr(X, Y)


class TestPushforward:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.15,
                1: 0.35,
                2: 0.1,
                3: 0.4,
            }
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVector(domain=Omega).from_dict(
            {
                0: (1, 2),
                1: (1, 2),
                2: (3, -1),
                3: (0, 1),
            }
        )

    @pytest.fixture
    def Y(self, Omega):
        return RandomVariable(domain=Omega, name="Y").from_dict(
            {
                0: 1,
                1: 1,
                2: -1,
                3: 2,
            }
        )

    def test_pushforward_random_vector_with_prob_measure_parameter(self, X, P):
        """Test pushforward of a probability measure along a 2D random vector with an explicit probability measure."""
        pushforward = Operators.pushforward(rv=X, prob_measure=P)

        assert isinstance(pushforward, ProbabilityMeasure)
        assert pushforward.sample_space == X.range.sample_space
        assert pushforward.name == "P_X"
        assert np.abs(pushforward((1, 2)) - 0.5) < 1e-9
        assert np.abs(pushforward((3, -1)) - 0.1) < 1e-9
        assert np.abs(pushforward((0, 1)) - 0.4) < 1e-9

    def test_pushforward_random_variable_with_prob_measure_parameter(self, Y, P):
        """Test pushforward of a probability measure along a random variable with an explicit probability measure."""
        pushforward = Operators.pushforward(rv=Y, prob_measure=P)

        assert isinstance(pushforward, ProbabilityMeasure)
        assert pushforward.sample_space == Y.range.sample_space
        assert pushforward.name == "P_Y"
        assert np.abs(pushforward(1) - 0.5) < 1e-9
        assert np.abs(pushforward(-1) - 0.1) < 1e-9
        assert np.abs(pushforward(2) - 0.4) < 1e-9

    def test_pushforward_random_vector_with_rv_prob_measure(self, X, P):
        """Test pushforward using the probability measure carried by the random vector."""
        X.with_probability_measure(prob_measure=P)
        pushforward = Operators.pushforward(rv=X)

        assert isinstance(pushforward, ProbabilityMeasure)
        assert pushforward.sample_space == X.range.sample_space
        assert np.abs(pushforward((1, 2)) - 0.5) < 1e-9
        assert np.abs(pushforward((3, -1)) - 0.1) < 1e-9
        assert np.abs(pushforward((0, 1)) - 0.4) < 1e-9

    def test_pushforward_random_variable_with_rv_prob_measure(self, Y, P):
        """Test pushforward using the probability measure carried by the random variable."""
        Y.with_probability_measure(prob_measure=P)
        pushforward = Operators.pushforward(rv=Y)

        assert isinstance(pushforward, ProbabilityMeasure)
        assert pushforward.sample_space == Y.range.sample_space
        assert np.abs(pushforward(1) - 0.5) < 1e-9
        assert np.abs(pushforward(-1) - 0.1) < 1e-9
        assert np.abs(pushforward(2) - 0.4) < 1e-9

    def test_pushforward_with_explicit_probability_measure_overrides_rv_measure(
        self, Omega, X, P
    ):
        """Test that explicit probability measure overrides the one carried by rv."""
        Q = ProbabilityMeasure(sample_space=Omega).from_dict(
            {
                0: 0.2,
                1: 0.3,
                2: 0.1,
                3: 0.4,
            }
        )
        X.with_probability_measure(prob_measure=P)
        pushforward = Operators.pushforward(rv=X, prob_measure=Q)

        assert np.abs(pushforward((1, 2)) - 0.5) < 1e-9
        assert np.abs(pushforward((3, -1)) - 0.1) < 1e-9
        assert np.abs(pushforward((0, 1)) - 0.4) < 1e-9

    def test_pushforward_probability_sums_to_one(self, X, P):
        """Test that the pushforward measure is a valid probability measure (sums to 1)."""
        pushforward = Operators.pushforward(rv=X, prob_measure=P)

        total_probability = sum(
            pushforward(point) for point in pushforward.sample_space
        )
        assert np.abs(total_probability - 1.0) < 1e-9

    def test_pushforward_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            Operators.pushforward("not a random vector")

    def test_pushforward_invalid_probability_measure_type_raises(self, X):
        """Test that invalid probability measure type raises TypeError."""
        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure"
        ):
            Operators.pushforward(X, prob_measure="not a probability measure")

    def test_pushforward_mismatched_sample_space_raises(self, Omega, X):
        """Test that probability measure on different sample space raises ValueError."""
        Omega2 = SampleSpace().from_sequence(size=3)
        Q = ProbabilityMeasure(sample_space=Omega2).from_dict({0: 0.3, 1: 0.3, 2: 0.4})

        with pytest.raises(
            ValueError,
            match="rv must be defined on the sample space of prob_measure",
        ):
            Operators.pushforward(X, prob_measure=Q)
