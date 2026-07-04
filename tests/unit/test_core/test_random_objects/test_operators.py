import numpy as np
import pandas as pd
import pytest

from sigalg.core import (
    Operators,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)


class TestIntegrate:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.3,
                1: 0.2,
                2: 0.5,
            },
        )

    @pytest.fixture
    def Q(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            name="Q",
            mapping={
                0: 0.1,
                1: 0.2,
                2: 0.7,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def A(self, prob_space):
        return prob_space.get_event([0, 1, 2, 3])

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            },
        )

    def test_integrate_random_vector(self, X, F, P):
        """Test integration of a 2D random vector."""
        X0, X1 = X.components
        integral = Operators.integrate(rv=X)
        int_X0 = sum(X0(atom) * P(atom) for atom in F.to_atoms)
        int_X1 = sum(X1(atom) * P(atom) for atom in F.to_atoms)
        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=X.index.data,
            name="int X dP",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_vector_on_event(self, X, F, P, A):
        """Test integration of a random vector on an event."""
        integral = Operators.integrate(rv=X, event=A)
        X0, X1 = X.components
        int_X0 = sum([(X0 * A.indicator)(atom) * P(atom) for atom in F.to_atoms])
        int_X1 = sum([(X1 * A.indicator)(atom) * P(atom) for atom in F.to_atoms])
        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=X.index.data,
            name="int_A X dP",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_variable(self, Y, F, P):
        """Test integration of a random variable."""
        integral = Operators.integrate(rv=Y)
        expected_integral = sum(Y(atom) * P(atom) for atom in F.to_atoms)

        assert np.abs(integral - expected_integral) < 1e-9

    def test_integrate_random_variable_on_event(self, Y, F, P, A):
        """Test integration of a random variable on an event."""
        integral = Operators.integrate(rv=Y, event=A)
        expected_integral = sum(
            (Y * A.indicator)(atom) * P(atom) for atom in F.to_atoms
        )

        assert np.abs(integral - expected_integral) < 1e-9

    def test_integrate_random_vector_with_prob_measure_parameter(self, X, F, Q):
        """Test integration of a random vector with a specified probability measure."""
        integral = Operators.integrate(rv=X, prob_measure=Q)
        X0, X1 = X.components
        int_X0 = sum(X0(atom) * Q(atom) for atom in F.to_atoms)
        int_X1 = sum(X1(atom) * Q(atom) for atom in F.to_atoms)
        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=X.index.data,
            name="int X dQ",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_vector_on_event_with_prob_measure_parameter(
        self, X, F, Q, A
    ):
        """Test integration of a random vector on an event with a specified probability measure."""
        integral = Operators.integrate(rv=X, event=A, prob_measure=Q)
        X0, X1 = X.components
        int_X0 = sum((X0 * A.indicator)(atom) * Q(atom) for atom in F.to_atoms)
        int_X1 = sum((X1 * A.indicator)(atom) * Q(atom) for atom in F.to_atoms)
        expected_integral = pd.Series(
            [int_X0, int_X1],
            index=X.index.data,
            name="int_A X dQ",
        )

        pd.testing.assert_series_equal(integral, expected_integral)

    def test_integrate_random_variable_with_prob_measure_parameter(self, Y, F, Q):
        """Test integration of a random variable with a specified probability measure."""
        integral = Operators.integrate(rv=Y, prob_measure=Q)
        expected_integral = sum(Y(atom) * Q(atom) for atom in F.to_atoms)

        assert np.abs(integral - expected_integral) < 1e-9

    def test_integrate_random_variable_on_event_with_prob_measure_parameter(
        self, Y, F, Q, A
    ):
        """Test integration of a random variable on an event with a specified probability measure."""
        integral = Operators.integrate(rv=Y, event=A, prob_measure=Q)
        expected_integral = sum(
            (Y * A.indicator)(atom) * Q(atom) for atom in F.to_atoms
        )

        assert np.abs(integral - expected_integral) < 1e-9

    def test_invalid_rv_raises(self):
        """Test that passing an invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            Operators.integrate(rv="not a random vector")

    def test_invalid_prob_measure_raises(self, X):
        """Test that passing an invalid probability measure raises TypeError."""
        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure"
        ):
            Operators.integrate(rv=X, prob_measure="not a probability measure")

    def test_invalid_event_raises(self, X):
        """Test that passing an invalid event raises TypeError."""
        with pytest.raises(TypeError, match="event must be an Event"):
            Operators.integrate(rv=X, event="not an event")

    def test_non_measurable_event_raises(self, X, Omega):
        """Test that passing an event that is not measurable with respect to the sigma-algebra raises ValueError."""
        non_measurable_event = SigmaAlgebra.power_set(Omega).get_event([1, 2])
        with pytest.raises(
            ValueError,
            match="the event must be an element of the sigma-algebra of the random vector",
        ):
            Operators.integrate(rv=X, event=non_measurable_event)


class TestExpectation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
            },
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.3,
                1: 0.2,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def Q(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            name="Q",
            mapping={
                0: 0.0,
                1: 0.3,
                2: 0.7,
            },
        )

    @pytest.fixture
    def F2(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="F2",
            mapping={
                0: (0, 1),
                1: (0, 1),
                2: (1, 2),
                3: (1, 2),
                4: (2, 3),
                5: (2, 3),
            },
        )

    @pytest.fixture
    def G2(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G2",
            mapping={
                0: (0, -1),
                1: (0, -1),
                2: (1, 2),
                3: (1, 2),
                4: (1, 2),
                5: (1, 2),
            },
        )

    @pytest.fixture
    def P2(self, F2):
        return ProbabilityMeasure(
            sig_alg=F2,
            name="P2",
            mapping={
                (0, 1): 0.3,
                (1, 2): 0.2,
                (2, 3): 0.5,
            },
        )

    @pytest.fixture
    def prob_space2(self, Omega, F2, P2):
        return ProbabilitySpace(Omega, F2, P2)

    @pytest.fixture
    def Q2(self, F2):
        return ProbabilityMeasure(
            sig_alg=F2,
            name="Q2",
            mapping={
                (0, 1): 0.0,
                (1, 2): 0.3,
                (2, 3): 0.7,
            },
        )

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            },
        )

    @pytest.fixture
    def X2(self, prob_space2):
        return RandomVector(
            *prob_space2,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            },
        )

    @pytest.fixture
    def Z(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (0, -2),
                1: (0, -2),
                2: (-3, 1),
                3: (-3, 1),
                4: (2, 6),
                5: (2, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            },
        )

    @pytest.fixture
    def Y2(self, prob_space2):
        return RandomVariable(
            *prob_space2,
            name="Y2",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            },
        )

    def test_unconditional_expectation_random_vector(self, X, F, P, prob_space):
        """Test the unconditional expectation of a random vector."""
        expectation = Operators.expectation(rv=X)
        X0, X1 = X.components
        exp_X0 = sum(X0(atom) * P(atom) for atom in F.to_atoms)
        exp_X1 = sum(X1(atom) * P(atom) for atom in F.to_atoms)
        expected_data = RandomVector.from_constant(
            *prob_space, name="E(X)", constant=(exp_X0, exp_X1)
        ).data

        pd.testing.assert_frame_equal(expectation.data, expected_data)

    def test_unconditional_expectation_random_vector_with_2_dim_sig_alg(
        self, X2, F2, P2, prob_space2
    ):
        """Test the unconditional expectation of a random vector with sigma-algebra with 2-dimensional atom IDs."""
        expectation = Operators.expectation(rv=X2)
        X0, X1 = X2.components
        exp_X0 = sum(X0(atom) * P2(atom) for atom in F2.to_atoms)
        exp_X1 = sum(X1(atom) * P2(atom) for atom in F2.to_atoms)
        expected_data = RandomVector.from_constant(
            *prob_space2, name="E(X)", constant=(exp_X0, exp_X1)
        ).data

        pd.testing.assert_frame_equal(expectation.data, expected_data)

    def test_unconditional_expectation_random_variable(self, Y, F, P):
        """Test the unconditional expectation of a random variable."""
        expectation = Operators.expectation(rv=Y)
        exp_Y = sum(Y(atom) * P(atom) for atom in F.to_atoms)
        expected_data = pd.Series(
            [exp_Y] * len(Y.sample_space),
            index=Y.sample_space.data,
            name="E(Y)",
        )

        pd.testing.assert_series_equal(expectation.data, expected_data)

    def test_unconditional_expectation_random_variable_with_2_dim_sig_alg(
        self, Y2, F2, P2
    ):
        """Test the unconditional expectation of a random variable with sigma-algebra with 2-dimensional atom IDs."""
        expectation = Operators.expectation(rv=Y2)
        exp_Y2 = sum(Y2(atom) * P2(atom) for atom in F2.to_atoms)
        expected_data = pd.Series(
            [exp_Y2] * len(Y2.sample_space),
            index=Y2.sample_space.data,
            name="E(Y2)",
        )

        pd.testing.assert_series_equal(expectation.data, expected_data)

    def test_unconditional_expectation_preserves_prob_space(self, X, prob_space):
        """Test that the unconditional expectation preserves the probability space of the random vector."""
        expectation = Operators.expectation(rv=X)
        assert expectation.prob_space == prob_space

    def test_unconditional_expectation_with_prob_measure_parameter_preserves_prob_space(
        self, Omega, X, F, Q
    ):
        """Test that the unconditional expectation with a specified probability measure preserves the probability space of the random vector."""
        expectation = Operators.expectation(rv=X, prob_measure=Q)
        expected_prob_space = ProbabilitySpace(Omega, F, Q)
        assert expectation.prob_space == expected_prob_space

    def test_conditional_expectation_random_vector(self, X, G, P):
        """Test the conditional expectation of a random vector."""
        expectation = Operators.expectation(rv=X, sig_alg=G)
        int = Operators.integrate
        X0, X1 = X.components
        exp_X0 = sum((int(X0, atom) / P(atom)) * atom.indicator for atom in G.to_atoms)
        exp_X1 = sum((int(X1, atom) / P(atom)) * atom.indicator for atom in G.to_atoms)
        expected_data = pd.DataFrame(
            {
                0: exp_X0.data,
                1: exp_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(expectation.data, expected_data)

    def test_conditional_expectation_random_vector_with_2_dim_sig_alg(self, X2, G2, P2):
        """Test the conditional expectation of a random vector with sigma-algebra with 2-dimensional atom IDs."""
        expectation = Operators.expectation(rv=X2, sig_alg=G2)
        int = Operators.integrate
        X0, X1 = X2.components
        exp_X0 = sum(
            (int(X0, atom) / P2(atom)) * atom.indicator for atom in G2.to_atoms
        )
        exp_X1 = sum(
            (int(X1, atom) / P2(atom)) * atom.indicator for atom in G2.to_atoms
        )
        expected_data = pd.DataFrame(
            {
                0: exp_X0.data,
                1: exp_X1.data,
            },
            index=X2.sample_space.data,
            columns=X2.index.data,
        )

        pd.testing.assert_frame_equal(expectation.data, expected_data)

    def test_conditional_expectation_random_variable(self, Y, G, P):
        """Test the conditional expectation of a random variable."""
        expectation = Operators.expectation(rv=Y, sig_alg=G)
        int = Operators.integrate
        exp_Y = sum((int(Y, atom) / P(atom)) * atom.indicator for atom in G.to_atoms)
        expected_data = pd.Series(
            exp_Y.data,
            index=Y.sample_space.data,
            name="E(Y|G)",
        )

        pd.testing.assert_series_equal(expectation.data, expected_data)

    def test_conditional_expectation_random_variable_with_2_dim_sig_alg(
        self, Y2, G2, P2
    ):
        """Test the conditional expectation of a random variable with sigma-algebra with 2-dimensional atom IDs."""
        expectation = Operators.expectation(rv=Y2, sig_alg=G2)
        int = Operators.integrate
        exp_Y2 = sum(
            (int(Y2, atom) / P2(atom)) * atom.indicator for atom in G2.to_atoms
        )
        expected_data = pd.Series(
            exp_Y2.data,
            index=Y2.sample_space.data,
            name="E(Y2|G2)",
        )

        pd.testing.assert_series_equal(expectation.data, expected_data)

    def test_conditional_expectation_random_vector_with_prob_measure_parameter(
        self, X, G, Q
    ):
        """Test the conditional expectation of a random vector with a specified probability measure."""
        expectation = Operators.expectation(rv=X, sig_alg=G, prob_measure=Q)
        int = Operators.integrate
        X0, X1 = X.components
        atoms_nonzero_prob = [atom for atom in G.to_atoms if Q(atom) > 0]
        exp_X0 = sum(
            (int(X0, atom, prob_measure=Q) / Q(atom)) * atom.indicator
            for atom in atoms_nonzero_prob
        )
        exp_X1 = sum(
            (int(X1, atom, prob_measure=Q) / Q(atom)) * atom.indicator
            for atom in atoms_nonzero_prob
        )
        expected_data = pd.DataFrame(
            {
                0: exp_X0.data,
                1: exp_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(expectation.data, expected_data)

    def test_conditional_expectation_random_vector_with_prob_measure_parameter_and_2_dim_sig_alg(
        self, X2, G2, Q2
    ):
        """Test the conditional expectation of a random vector with a specified probability measure and sigma-algebra with 2-dimensional atom IDs."""
        expectation = Operators.expectation(rv=X2, sig_alg=G2, prob_measure=Q2)
        int = Operators.integrate
        X0, X1 = X2.components
        atoms_nonzero_prob = [atom for atom in G2.to_atoms if Q2(atom) > 0]
        exp_X0 = sum(
            (int(X0, atom, prob_measure=Q2) / Q2(atom)) * atom.indicator
            for atom in atoms_nonzero_prob
        )
        exp_X1 = sum(
            (int(X1, atom, prob_measure=Q2) / Q2(atom)) * atom.indicator
            for atom in atoms_nonzero_prob
        )
        expected_data = pd.DataFrame(
            {
                0: exp_X0.data,
                1: exp_X1.data,
            },
            index=X2.sample_space.data,
            columns=X2.index.data,
        )

        pd.testing.assert_frame_equal(expectation.data, expected_data)

    def test_conditional_expectation_random_variable_with_prob_measure_parameter(
        self, Y, G, Q
    ):
        """Test the conditional expectation of a random variable with a specified probability measure."""
        expectation = Operators.expectation(rv=Y, sig_alg=G, prob_measure=Q)
        int = Operators.integrate
        atoms_nonzero_prob = [atom for atom in G.to_atoms if Q(atom) > 0]
        exp_Y = sum(
            (int(Y, atom, prob_measure=Q) / Q(atom)) * atom.indicator
            for atom in atoms_nonzero_prob
        )
        expected_data = pd.Series(
            exp_Y.data,
            index=Y.sample_space.data,
            name="E(Y|G)",
        )

        pd.testing.assert_series_equal(expectation.data, expected_data)

    def test_conditional_expectation_random_variable_with_prob_measure_parameter_and_2_dim_sig_alg(
        self, Y2, G2, Q2
    ):
        """Test the conditional expectation of a random variable with a specified probability measure and sigma-algebra with 2-dimensional atom IDs."""
        expectation = Operators.expectation(rv=Y2, sig_alg=G2, prob_measure=Q2)
        int = Operators.integrate
        atoms_nonzero_prob = [atom for atom in G2.to_atoms if Q2(atom) > 0]
        exp_Y2 = sum(
            (int(Y2, atom, prob_measure=Q2) / Q2(atom)) * atom.indicator
            for atom in atoms_nonzero_prob
        )
        expected_data = pd.Series(
            exp_Y2.data,
            index=Y2.sample_space.data,
            name="E(Y2|G2)",
        )

        pd.testing.assert_series_equal(expectation.data, expected_data)

    def test_conditional_expectation_preserves_prob_space(self, X, G, prob_space):
        """Test that the conditional expectation preserves the probability space of the random vector."""
        expectation = Operators.expectation(rv=X, sig_alg=G)
        assert expectation.prob_space == prob_space

    def test_conditional_expectation_with_prob_measure_parameter_preserves_prob_space(
        self, Omega, X, F, G, Q
    ):
        """Test that the conditional expectation with a specified probability measure preserves the probability space of the random vector."""
        expectation = Operators.expectation(rv=X, sig_alg=G, prob_measure=Q)
        expected_prob_space = ProbabilitySpace(Omega, F, Q)
        assert expectation.prob_space == expected_prob_space

    def test_conditional_expectation_measurable_random_vector(self, X, F):
        """Test that the conditional expectation of a random vector that is measurable with respect to the sigma-algebra is equal to itself."""
        assert Operators.expectation(rv=X, sig_alg=F) == X

    def test_conditional_expectation_measurable_random_variable(self, Y, F):
        """Test that the conditional expectation of a random variable that is measurable with respect to the sigma-algebra is equal to itself."""
        assert Operators.expectation(rv=Y, sig_alg=F) == Y

    def test_linearity_of_expectation(self, X, Z, G):
        """Test the linearity of expectation."""
        a = 2
        b = -3
        exp = Operators.expectation

        assert exp(a * X + b * Z, G) == a * exp(X, G) + b * exp(Z, G)

    def test_factoring_out_measurable_functions(self, prob_space, Y, G):
        """Test that functions measurable with respect to the sigma-algebra can be factored out of the expectation."""
        C = RandomVariable(
            *prob_space,
            name="C",
            mapping={
                0: 2,
                1: 2,
                2: -1,
                3: -1,
                4: -1,
                5: -1,
            },
        )
        exp = Operators.expectation

        assert exp(C * Y, G) == C * exp(Y, G)

    def test_independence_and_expectation(self):
        """Test that if X is independent of F, then E(X|F) = E(X)."""
        exp = Operators.expectation
        Omega = SampleSpace.from_sequence(size=4)
        P = ProbabilityMeasure(
            sample_space=Omega,
            mapping={
                0: 0.75**2,  # HH
                1: 0.75 * 0.25,  # TH
                2: 0.75 * 0.25,  # HT
                3: 0.25**2,  # TT
            },
        )
        X = RandomVariable(
            Omega,
            prob_measure=P,
            mapping={
                0: 0,
                1: 1,
                2: 0,
                3: 1,
            },
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            },
        )

        assert exp(X) == exp(X, G)

    def test_iterated_expectation(self):
        """Test the law of iterated expectation."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            },
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 1,
            },
        )
        power_set = SigmaAlgebra.power_set(Omega)
        P = ProbabilityMeasure(
            sig_alg=power_set,
            mapping={
                0: 0.1,
                1: 0.15,
                2: 0.25,
                3: 0.5,
            },
        )
        X = RandomVariable(
            Omega,
            power_set,
            P,
            mapping={
                0: -1,
                1: 2,
                2: -3,
                3: 2,
            },
        )
        exp = Operators.expectation

        assert exp(exp(X, F), G) == exp(X, G)

    def test_expectation_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            Operators.expectation("not a random vector")

    def test_expectation_invalid_sigma_algebra_type_raises(self, X):
        """Test that invalid sigma algebra type raises TypeError."""
        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra instance"):
            Operators.expectation(X, sig_alg="not a sigma algebra")

    def test_expectation_invalid_probability_measure_type_raises(self, X):
        """Test that invalid probability measure type raises TypeError."""
        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure instance"
        ):
            Operators.expectation(X, prob_measure="not a probability measure")

    def test_non_sub_sigma_algebra_raises(self, X, Omega):
        """Test that passing a sigma-algebra that is not a sub-sigma-algebra of the sigma-algebra of the random variable raises ValueError."""
        H = SigmaAlgebra(
            sample_space=Omega,
            name="H",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
                5: 2,
            },
        )

        with pytest.raises(ValueError, match="must be a sub-sigma-algebra"):
            Operators.expectation(X, sig_alg=H)

    def test_invalid_prob_measure_raises(self, X, Omega):
        """Test that passing a probability measure that is not defined on the same sigma-algebra as the random variable raises ValueError."""
        power_set = SigmaAlgebra.power_set(Omega)
        P_invalid = ProbabilityMeasure(
            sig_alg=power_set,
            mapping={
                0: 0.05,
                1: 0.15,
                2: 0.25,
                3: 0.5,
                4: 0.05,
                5: 0.0,
            },
        )

        with pytest.raises(ValueError, match="must be defined on the sigma-algebra"):
            Operators.expectation(X, prob_measure=P_invalid)


class TestVariance:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
            },
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.3,
                1: 0.2,
                2: 0.5,
            },
        )

    @pytest.fixture
    def Q(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            name="Q",
            mapping={
                0: 0.0,
                1: 0.3,
                2: 0.7,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            },
        )

    @pytest.fixture
    def Z(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (0, -2),
                1: (0, -2),
                2: (-3, 1),
                3: (-3, 1),
                4: (2, 6),
                5: (2, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            },
        )

    def test_unconditional_variance_random_vector(self, X):
        """Test the unconditional variance of a random vector."""
        variance = Operators.variance(rv=X)
        X0, X1 = X.components
        E = Operators.expectation
        var_X0 = E(X0**2) - E(X0) ** 2
        var_X1 = E(X1**2) - E(X1) ** 2
        expected_data = pd.DataFrame(
            {
                0: var_X0.data,
                1: var_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(variance.data, expected_data)

    def test_unconditional_variance_random_variable(self, Y):
        """Test the unconditional variance of a random variable."""
        variance = Operators.variance(rv=Y)
        E = Operators.expectation
        var_Y = E(Y**2) - E(Y) ** 2
        expected_data = pd.Series(
            var_Y.data,
            index=Y.sample_space.data,
            name="V(Y)",
        )

        pd.testing.assert_series_equal(variance.data, expected_data)

    def test_conditional_variance_random_vector(self, X, G):
        """Test the conditional variance of a random vector."""
        variance = Operators.variance(rv=X, sig_alg=G)
        X0, X1 = X.components
        E = Operators.expectation
        var_X0 = E(X0**2, G) - E(X0, G) ** 2
        var_X1 = E(X1**2, G) - E(X1, G) ** 2
        expected_data = pd.DataFrame(
            {
                0: var_X0.data,
                1: var_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(variance.data, expected_data)

    def test_conditional_variance_random_variable(self, Y, G):
        """Test the conditional variance of a random variable."""
        variance = Operators.variance(rv=Y, sig_alg=G)
        E = Operators.expectation
        var_Y = E(Y**2, G) - E(Y, G) ** 2
        expected_data = pd.Series(
            var_Y.data,
            index=Y.sample_space.data,
            name="V(Y|G)",
        )

        pd.testing.assert_series_equal(variance.data, expected_data)

    def test_conditional_variance_random_vector_with_prob_measure_parameter(
        self, X, G, Q
    ):
        """Test the conditional variance of a random vector with a specified probability measure."""
        variance = Operators.variance(rv=X, sig_alg=G, prob_measure=Q)
        E = Operators.expectation
        X0, X1 = X.components
        var_X0 = E(X0**2, G, Q) - E(X0, G, Q) ** 2
        var_X1 = E(X1**2, G, Q) - E(X1, G, Q) ** 2
        expected_data = pd.DataFrame(
            {
                0: var_X0.data,
                1: var_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(variance.data, expected_data)

    def test_conditional_variance_random_variable_with_prob_measure_parameter(
        self, Y, G, Q
    ):
        """Test the conditional variance of a random variable with a specified probability measure."""
        variance = Operators.variance(rv=Y, sig_alg=G, prob_measure=Q)
        E = Operators.expectation
        var_Y = E(Y**2, G, Q) - E(Y, G, Q) ** 2
        expected_data = pd.Series(
            var_Y.data,
            index=Y.sample_space.data,
            name="V(Y|G)",
        )

        pd.testing.assert_series_equal(variance.data, expected_data)

    def test_total_variance(self, X, G):
        """Test the law of total variance."""
        E = Operators.expectation
        V = Operators.variance

        assert V(X) == E(V(X, G)) + V(E(X, G))

    def test_variance_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            Operators.variance("not a random vector")

    def test_variance_invalid_sigma_algebra_type_raises(self, X):
        """Test that invalid sigma algebra type raises TypeError."""
        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra instance"):
            Operators.variance(X, sig_alg="not a sigma algebra")

    def test_variance_invalid_probability_measure_type_raises(self, X):
        """Test that invalid probability measure type raises TypeError."""
        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure instance"
        ):
            Operators.variance(X, prob_measure="not a probability measure")

    def test_non_sub_sigma_algebra_raises(self, X, Omega):
        """Test that passing a sigma-algebra that is not a sub-sigma-algebra of the sigma-algebra of the random variable raises ValueError."""
        H = SigmaAlgebra(
            sample_space=Omega,
            name="H",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
                5: 2,
            },
        )

        with pytest.raises(ValueError, match="must be a sub-sigma-algebra"):
            Operators.variance(X, sig_alg=H)

    def test_invalid_prob_measure_raises(self, X, Omega):
        """Test that passing a probability measure that is not defined on the same sigma-algebra as the random variable raises ValueError."""
        power_set = SigmaAlgebra.power_set(Omega)
        P_invalid = ProbabilityMeasure(
            sig_alg=power_set,
            mapping={
                0: 0.05,
                1: 0.15,
                2: 0.25,
                3: 0.5,
                4: 0.05,
                5: 0.0,
            },
        )

        with pytest.raises(ValueError, match="must be defined on the sigma-algebra"):
            Operators.variance(X, prob_measure=P_invalid)


class TestStandardDeviation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
            },
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.3,
                1: 0.2,
                2: 0.5,
            },
        )

    @pytest.fixture
    def Q(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            name="Q",
            mapping={
                0: 0.0,
                1: 0.3,
                2: 0.7,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            },
        )

    @pytest.fixture
    def Z(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (0, -2),
                1: (0, -2),
                2: (-3, 1),
                3: (-3, 1),
                4: (2, 6),
                5: (2, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            },
        )

    def test_unconditional_standard_devation_random_vector(self, X):
        """Test the unconditional standard deviation of a random vector."""
        std = Operators.std(rv=X)
        X0, X1 = X.components
        E = Operators.expectation
        var_X0 = E(X0**2) - E(X0) ** 2
        var_X1 = E(X1**2) - E(X1) ** 2
        std_X0 = var_X0**0.5
        std_X1 = var_X1**0.5
        expected_data = pd.DataFrame(
            {
                0: std_X0.data,
                1: std_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(std.data, expected_data)

    def test_unconditional_standard_deviation_random_variable(self, Y):
        """Test the unconditional standard deviation of a random variable."""
        std = Operators.std(rv=Y)
        E = Operators.expectation
        var_Y = E(Y**2) - E(Y) ** 2
        std_Y = var_Y**0.5
        expected_data = pd.Series(
            std_Y.data,
            index=Y.sample_space.data,
            name="std(Y)",
        )

        pd.testing.assert_series_equal(std.data, expected_data)

    def test_conditional_standard_deviation_random_vector(self, X, G):
        """Test the conditional standard deviation of a random vector."""
        std = Operators.std(rv=X, sig_alg=G)
        X0, X1 = X.components
        E = Operators.expectation
        var_X0 = E(X0**2, G) - E(X0, G) ** 2
        var_X1 = E(X1**2, G) - E(X1, G) ** 2
        std_X0 = var_X0**0.5
        std_X1 = var_X1**0.5
        expected_data = pd.DataFrame(
            {
                0: std_X0.data,
                1: std_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(std.data, expected_data)

    def test_conditional_standard_deviation_random_variable(self, Y, G):
        """Test the conditional standard deviation of a random variable."""
        std = Operators.std(rv=Y, sig_alg=G)
        E = Operators.expectation
        var_Y = E(Y**2, G) - E(Y, G) ** 2
        std_Y = var_Y**0.5
        expected_data = pd.Series(
            std_Y.data,
            index=Y.sample_space.data,
            name="std(Y|G)",
        )

        pd.testing.assert_series_equal(std.data, expected_data)

    def test_conditional_standard_deviation_random_vector_with_prob_measure_parameter(
        self, X, G, Q
    ):
        """Test the conditional standard deviation of a random vector with a specified probability measure."""
        std = Operators.std(rv=X, sig_alg=G, prob_measure=Q)
        E = Operators.expectation
        X0, X1 = X.components
        var_X0 = E(X0**2, G, Q) - E(X0, G, Q) ** 2
        var_X1 = E(X1**2, G, Q) - E(X1, G, Q) ** 2
        std_X0 = var_X0**0.5
        std_X1 = var_X1**0.5
        expected_data = pd.DataFrame(
            {
                0: std_X0.data,
                1: std_X1.data,
            },
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(std.data, expected_data)

    def test_conditional_standard_deviation_random_variable_with_prob_measure_parameter(
        self, Y, G, Q
    ):
        """Test the conditional standard deviation of a random variable with a specified probability measure."""
        std = Operators.std(rv=Y, sig_alg=G, prob_measure=Q)
        E = Operators.expectation
        var_Y = E(Y**2, G, Q) - E(Y, G, Q) ** 2
        std_Y = var_Y**0.5
        expected_data = pd.Series(
            std_Y.data,
            index=Y.sample_space.data,
            name="std(Y|G)",
        )

        pd.testing.assert_series_equal(std.data, expected_data)

    def test_standard_deviation_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv must be a RandomVector"):
            Operators.std("not a random vector")

    def test_standard_deviation_invalid_sigma_algebra_type_raises(self, X):
        """Test that invalid sigma algebra type raises TypeError."""
        with pytest.raises(TypeError, match="sig_alg must be a SigmaAlgebra instance"):
            Operators.std(X, sig_alg="not a sigma algebra")

    def test_standard_deviation_invalid_probability_measure_type_raises(self, X):
        """Test that invalid probability measure type raises TypeError."""
        with pytest.raises(
            TypeError, match="prob_measure must be a ProbabilityMeasure instance"
        ):
            Operators.std(X, prob_measure="not a probability measure")

    def test_non_sub_sigma_algebra_raises(self, X, Omega):
        """Test that passing a sigma-algebra that is not a sub-sigma-algebra of the sigma-algebra of the random variable raises ValueError."""
        H = SigmaAlgebra(
            sample_space=Omega,
            name="H",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
                5: 2,
            },
        )

        with pytest.raises(ValueError, match="must be a sub-sigma-algebra"):
            Operators.std(X, sig_alg=H)

    def test_invalid_prob_measure_raises(self, X, Omega):
        """Test that passing a probability measure that is not defined on the same sigma-algebra as the random variable raises ValueError."""
        power_set = SigmaAlgebra.power_set(Omega)
        P_invalid = ProbabilityMeasure(
            sig_alg=power_set,
            mapping={
                0: 0.05,
                1: 0.15,
                2: 0.25,
                3: 0.5,
                4: 0.05,
                5: 0.0,
            },
        )

        with pytest.raises(ValueError, match="must be defined on the sigma-algebra"):
            Operators.std(X, prob_measure=P_invalid)


class TestCovariance:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=5)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        rng = np.random.default_rng(42)
        return ProbabilityMeasure.from_rand(sig_alg=F, random_state=rng)

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        rng = np.random.default_rng(42)
        return RandomVariable.from_randint(
            *prob_space, low=-20, high=21, random_state=rng
        )

    @pytest.fixture
    def Y(self, prob_space):
        rng = np.random.default_rng(42)
        return RandomVariable.from_randint(
            *prob_space, name="Y", low=-10, high=11, random_state=rng
        )

    @pytest.fixture
    def Z(self, prob_space):
        rng = np.random.default_rng(42)
        return RandomVariable.from_randint(
            *prob_space, name="Z", low=-20, high=21, random_state=rng
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
            },
        )

    def test_unconditional_covariance(self, X, Y):
        """Test unconditional covariance."""
        cov = Operators.cov
        exp = Operators.expectation
        covar = cov(X, Y)
        expected_covar = (exp(X * Y) - exp(X) * exp(Y)).with_name("cov(X, Y)")

        pd.testing.assert_series_equal(covar.data, expected_covar.data)
        assert covar.name == "cov(X, Y)"

    def test_conditional_covariance(self, X, Y, G):
        """Test conditional covariance."""
        cov = Operators.cov
        exp = Operators.expectation
        covar_cond = cov(X, Y, G)
        expected_covar_cond = (exp(X * Y, G) - exp(X, G) * exp(Y, G)).with_name(
            "cov(X, Y|G)"
        )

        pd.testing.assert_series_equal(covar_cond.data, expected_covar_cond.data)
        assert covar_cond.name == "cov(X, Y|G)"

    def test_sum_of_atom_covariances_formula(self, X, Y, G):
        """Test whether the conditional covariance is the linear combination of the indicator functions of the atoms with weights given by restricted covariances."""
        cov = Operators.cov
        I = RandomVariable.indicator_of
        covar_cond = cov(X, Y, G)

        covar_linear_combo = sum(
            [
                cov(X.restrict_to(atom), Y.restrict_to(atom)).item() * I(atom)
                for atom in G.to_atoms
            ]
        ).with_name("cov(X, Y|G)")

        pd.testing.assert_series_equal(covar_cond.data, covar_linear_combo.data)
        assert covar_cond.name == "cov(X, Y|G)"

    def test_alternate_formula_for_covariance(self, X, Y, G):
        """Test the alternate formula cov(X, Y|G) = E[(X - E(X|G))(Y - E(Y|G))|G]."""
        cov = Operators.cov
        exp = Operators.expectation
        covar = cov(X, Y, G)
        alternate = exp((X - exp(X, G)) * (Y - exp(Y, G)), G).with_name("cov(X, Y|G)")

        pd.testing.assert_series_equal(covar.data, alternate.data)

    def test_symmetry_of_covariance(self, X, Y, G):
        """Test that cov(X, Y|G) = cov(Y, X|G)."""
        cov = Operators.cov

        assert cov(X, Y, G) == cov(Y, X, G)

    def test_bilinearity_of_covariance(self, X, Y, Z, G):
        """Test the bilinearity property of covariance."""
        a = 3
        cov = Operators.cov

        assert cov(a * X + Y, Z, G) == a * cov(X, Z, G) + cov(Y, Z, G)

    def test_covariance_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv1 and rv2 must be RandomVariables"):
            Operators.cov("not a random variable", "also not")

    def test_covariance_different_domains_raises(self, Omega):
        """Test that random variables with different domains raise ValueError."""
        Omega2 = SampleSpace.from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        Y = RandomVariable(sample_space=Omega2, name="Y", mapping={0: 1, 1: 2, 2: 3})

        with pytest.raises(
            ValueError, match="rv1 and rv2 must be defined on the same sample space."
        ):
            Operators.cov(X, Y)

    def test_covariance_mismatched_probability_measures_raises(self, Omega):
        """Test that mismatched probability measures raise ValueError when not explicitly passed."""
        P1 = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(Omega),
            mapping={0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2},
        )
        P2 = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(Omega),
            mapping={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.2, 4: 0.2},
        )
        X = RandomVariable(sample_space=Omega, mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        Y = RandomVariable(
            sample_space=Omega, name="Y", mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
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
        return SampleSpace.from_sequence(size=5)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        rng = np.random.default_rng(42)
        return ProbabilityMeasure.from_rand(sig_alg=F, random_state=rng)

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        rng = np.random.default_rng(42)
        return RandomVariable.from_randint(
            *prob_space, low=-20, high=21, random_state=rng
        )

    @pytest.fixture
    def Y(self, prob_space):
        rng = np.random.default_rng(42)
        return RandomVariable.from_randint(
            *prob_space, name="Y", low=-10, high=11, random_state=rng
        )

    @pytest.fixture
    def Z(self, prob_space):
        rng = np.random.default_rng(42)
        return RandomVariable.from_randint(
            *prob_space, name="Z", low=-20, high=21, random_state=rng
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 1,
            },
        )

    def test_unconditional_correlation(self, X, Y):
        """Test unconditional correlation."""
        corr = Operators.corr(X, Y)
        var = Operators.variance
        cov = Operators.cov
        expected_corr = (cov(X, Y) / (var(X) * var(Y)) ** 0.5).with_name("corr(X, Y)")

        pd.testing.assert_series_equal(corr.data, expected_corr.data)
        assert corr.name == "corr(X, Y)"

    def test_conditional_correlation(self, X, Y, G):
        """Test conditional correlation."""
        corr = Operators.corr(X, Y, G)
        var = Operators.variance
        cov = Operators.cov
        expected_corr = (cov(X, Y, G) / (var(X, G) * var(Y, G)) ** 0.5).with_name(
            "corr(X, Y|G)"
        )

        pd.testing.assert_series_equal(corr.data, expected_corr.data)
        assert corr.name == "corr(X, Y|G)"

    def test_sum_of_atom_correlations_formula(self, X, Y, G):
        """Test whether the conditional correlation is the linear combination of the indicator functions of the atoms with weights given by restricted correlations."""
        I = RandomVariable.indicator_of
        corr = Operators.corr
        corr_linear_combo = sum(
            [corr(X | A, Y | A).item() * I(A) for A in G.to_atoms]
        ).with_name("corr(X, Y|G)")

        pd.testing.assert_series_equal(
            Operators.corr(X, Y, G).data, corr_linear_combo.data
        )
        assert Operators.corr(X, Y, G).name == "corr(X, Y|G)"

    def test_perfectly_correlated_random_variables(self):
        """Test that perfectly correlated random variables have correlation plus/minus 1."""
        rng = np.random.default_rng(42)
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra.power_set(Omega)
        P = ProbabilityMeasure.from_rand(sig_alg=F, random_state=rng)
        prob_space = ProbabilitySpace(Omega, F, P)
        X = RandomVariable(
            *prob_space,
            mapping={
                0: -1,  # on the line y = x
                1: 1,  # on the line y = x
                2: -1,  # on the line y = -x
                3: 1,  # on the line y = -x
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: -1,  # on the line y = x
                1: 1,  # on the line y = x
                2: 1,  # on the line y = -x
                3: -1,  # on the line y = -x
            },
        )

        G = SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            },
        )

        corr = Operators.corr(X, Y, G)

        for omega in Omega:
            assert np.abs(np.abs(corr(omega)) - 1.0) < 1e-9

    def test_independence_implies_uncorrelated(self):
        """Test that independent random variables are uncorrelated."""
        Omega = SampleSpace.from_sequence(size=2)
        P = ProbabilityMeasure(
            sample_space=Omega,
            mapping={0: 0.3, 1: 0.7},
        )
        Y = RandomVector.from_identity(sample_space=Omega, prob_measure=P, name="Y")
        X = (Y ^ 2).with_name("X")
        X_0, X_1 = X
        corr = Operators.corr(X_0, X_1)

        assert np.abs(corr.item()) < 1e-9

    def test_correlation_invalid_rv_type_raises(self):
        """Test that invalid rv type raises TypeError."""
        with pytest.raises(TypeError, match="rv1 and rv2 must be RandomVariables"):
            Operators.corr("not a random variable", "also not")

    def test_correlation_different_domains_raises(self, Omega):
        """Test that random variables with different domains raise ValueError."""
        Omega2 = SampleSpace.from_sequence(size=3)
        X = RandomVariable(sample_space=Omega, mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
        Y = RandomVariable(sample_space=Omega2, name="Y", mapping={0: 1, 1: 2, 2: 3})

        with pytest.raises(
            ValueError, match="rv1 and rv2 must be defined on the same sample space."
        ):
            Operators.corr(X, Y)

    def test_correlation_mismatched_probability_measures_raises(self, Omega):
        """Test that mismatched probability measures raise ValueError when not explicitly passed."""
        F = SigmaAlgebra.power_set(Omega)
        P1 = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.2,
                2: 0.2,
                3: 0.2,
                4: 0.2,
            },
        )
        P2 = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.1,
                1: 0.2,
                2: 0.3,
                3: 0.2,
                4: 0.2,
            },
        )
        X = RandomVariable(
            Omega,
            F,
            P1,
            mapping={
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: 5,
            },
        )
        Y = RandomVariable(
            Omega,
            F,
            P2,
            name="Y",
            mapping={
                0: 1,
                1: 2,
                2: 3,
                3: 4,
                4: 5,
            },
        )

        with pytest.raises(
            ValueError,
            match="If prob_measure is not passed, then the probability measures on the random variables will be used. But they are not equal.",
        ):
            Operators.corr(X, Y)


class TestPushforward:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def P(self, Omega):
        return ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(Omega),
            mapping={
                0: 0.15,
                1: 0.35,
                2: 0.1,
                3: 0.4,
            },
        )

    @pytest.fixture
    def X(self, Omega):
        return RandomVector(
            sample_space=Omega,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, -1),
                3: (0, 1),
            },
        )

    @pytest.fixture
    def Y(self, Omega):
        return RandomVariable(
            sample_space=Omega,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: -1,
                3: 2,
            },
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
        Omega2 = SampleSpace.from_sequence(size=3)
        Q = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(Omega2), mapping={0: 0.3, 1: 0.3, 2: 0.4}
        )

        with pytest.raises(
            ValueError,
            match="rv must be defined on the sample space of prob_measure",
        ):
            Operators.pushforward(X, prob_measure=Q)
