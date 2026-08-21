from itertools import product

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
    Set,
    SigmaAlgebra,
)
from sigalg.core.sigma_algebras.lattice import NonMeasurableError

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_with_no_parameters(self):
        """Test the constructor with no parameters."""
        P = ProbabilityMeasure()

        assert P.name == "P"
        assert P.domain is None
        assert P.sig_alg is None
        assert P.domain is None
        assert P.data is None

    def test_from_dict_with_valid_sig_alg(self):
        """Test from dict with a valid sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=6)
        F = SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 1,
                1: 1,
                2: 0,
                3: 2,
                4: 2,
                5: 2,
            },
        )
        mapping = {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }
        Q = ProbabilityMeasure(
            domain=F,
            mapping=mapping,
            name="Q",
        )

        assert Q.name == "Q"
        assert Q.sig_alg is F
        assert Q.domain == F.atom_space

    def test_from_pandas_with_valid_sig_alg(self):
        """Test from pandas with a valid sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=6)
        F = SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 1,
                1: 1,
                2: 0,
                3: 2,
                4: 2,
                5: 2,
            },
        )
        mapping = pd.Series([0.2, 0.2, 0.6])
        Q = ProbabilityMeasure(
            domain=F,
            mapping=mapping,
            name="Q",
        )
        expected_data = pd.Series(
            [0.2, 0.2, 0.6],
            index=pd.Index([1, 0, 2], name="F"),
            name="Q",
        )

        assert Q.name == "Q"
        assert Q.sig_alg is F
        assert Q.domain == F.atom_space
        pd.testing.assert_series_equal(Q.data, expected_data)


class TestUniform:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(["a", "b", "c", "d"])

    def test_on_power_set(self, Omega):
        """Test the uniform probability measure constructor on a power set."""
        U = ProbabilityMeasure.uniform(domain=Omega)
        expected_data = pd.Series(
            [0.25, 0.25, 0.25, 0.25],
            index=pd.Index(["a", "b", "c", "d"], name="s"),
            name="U",
        )

        pd.testing.assert_series_equal(U.data, expected_data)
        assert U.name == "U"

    def test_on_coarser_sigma_algebra(self, Omega):
        """Test the uniform probability measure constructor on a coarser sigma-algebra."""
        F = SigmaAlgebra(domain=Omega, mapping={"a": 0, "b": 0, "c": 1, "d": 1})
        K = ProbabilityMeasure.uniform(domain=F, name="K")
        expected_data = pd.Series(
            [0.5, 0.5], index=pd.Index([0, 1], name="F"), name="K"
        )

        pd.testing.assert_series_equal(K.data, expected_data)
        assert K.name == "K"

    def test_on_trivial_sigma_algebra(self, Omega):
        """Test the uniform probability measure constructor on a trivial sigma-algebra."""
        F = SigmaAlgebra.trivial(domain=Omega, name="F")
        U = ProbabilityMeasure.uniform(domain=F)
        expected_data = pd.Series([1.0], index=pd.Index([0], name="F"), name="U")

        pd.testing.assert_series_equal(U.data, expected_data)
        assert U.name == "U"


class TestTensorProduct:
    def test_product_prob_measure(self):
        """Test the tensor_product method on two probability measures."""
        S = SampleSpace.from_sequence(size=3, variable_name="s", name="S")
        T = SampleSpace.from_sequence(size=3, variable_name="t", name="T")
        F = SigmaAlgebra(
            domain=S,
            mapping={
                0: 0,
                1: 1,
                2: 1,
            },
            variable_names=["u"],
        )
        G = SigmaAlgebra(
            domain=T,
            mapping={
                0: ("a", "b"),
                1: ("a", "b"),
                2: ("c", "d"),
            },
            variable_names=["v", "w"],
            name="G",
        )
        P = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.1,
                1: 0.9,
            },
        )
        Q = ProbabilityMeasure(
            domain=G,
            mapping={
                ("a", "b"): 0.3,
                ("c", "d"): 0.7,
            },
            name="Q",
        )
        P_times_Q = ProbabilityMeasure.tensor_product([P, Q])
        expected_data = pd.Series(
            [
                0.1 * 0.3,
                0.1 * 0.7,
                0.9 * 0.3,
                0.9 * 0.7,
            ],
            index=pd.MultiIndex.from_tuples(
                [
                    (0, "a", "b"),
                    (0, "c", "d"),
                    (1, "a", "b"),
                    (1, "c", "d"),
                ],
                names=["u", "v", "w"],
            ),
            name="P x Q",
        )

        pd.testing.assert_series_equal(P_times_Q.data, expected_data)


# --------------------- test properties --------------------- #


class TestSigAlg:
    def test_sig_alg_is_the_original(self):
        """Test whether the sig_alg attribute is the same as the original sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=3)
        F = SigmaAlgebra(domain=Omega, mapping=dict(zip(Omega, [0, 0, 1])))
        P = ProbabilityMeasure(domain=F, mapping=dict(zip([0, 1], [0.2, 0.8])))

        assert P.sig_alg is F


# --------------------- test data access methods --------------------- #


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 1,
                1: 1,
                2: 0,
                3: 2,
                4: 2,
                5: 2,
            },
            variable_names=["F"],
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            name="G",
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (0, 2),
                3: (2, 4),
                4: (2, 4),
                5: (2, 4),
            },
            variable_names=["G_0", "G_1"],
        )

    @pytest.fixture
    def mapping_F(self):
        return {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }

    @pytest.fixture
    def mapping_G(self):
        return {
            (1, 2): 0.2,
            (0, 2): 0.2,
            (2, 4): 0.6,
        }

    @pytest.fixture
    def P(self, F, mapping_F):
        return ProbabilityMeasure(domain=F, mapping=mapping_F)

    @pytest.fixture
    def Q(self, G, mapping_G):
        return ProbabilityMeasure(domain=G, mapping=mapping_G, name="G")

    def test_on_event(self, F, G, P, Q):
        """Test call method on event instances."""
        A = F.get_set([0, 1])
        B = G.get_set([0, 1], name="B")
        C = F.get_set([2, 3, 4, 5], name="C")
        D = G.get_set([2, 3, 4, 5], name="D")

        assert P(A) == 0.2
        assert Q(B) == 0.2
        assert P(C) == 0.8
        assert Q(D) == 0.8

    def test_on_list(self, P, Q):
        """Test call method on list of sample points."""
        assert P([0, 1]) == 0.2
        assert Q([0, 1]) == 0.2

    def test_on_atom_id(self, P, Q):
        """Test call method on atom ID."""
        assert P(F=2) == 0.6
        assert Q(G_0=2, G_1=4) == 0.6

    def test_non_measurable_event_raises(self, Omega, P):
        power_set = SigmaAlgebra.power_set(Omega)
        power_set.get_set([2, 3])

        with pytest.raises(
            NonMeasurableError, match="The candidate set is not measurable"
        ):
            P([2, 3])


# --------------------- test equality --------------------- #


class TestEquality:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )

    def test_non_equality_different_probabilities(self, F):
        """Test the __eq__ method for inequality with different probabilities."""
        P1 = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.6,
                1: 0.4,
            },
        )
        P2 = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.5,
                1: 0.5,
            },
        )

        assert P1 != P2

    def test_equality_same_probabilities_and_sigma_algebra(self, F):
        """Test the __eq__ method for equality with same probabilities and sigma algebra."""
        P1 = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.7,
                1: 0.3,
            },
        )
        P2 = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.7,
                1: 0.3,
            },
        )

        assert P1 == P2

    def test_equality_same_components_different_names(self):
        """Test the __eq__ method for equality with same components but different names."""
        Omega1 = SampleSpace.from_sequence(size=3, name="Omega1")
        Omega2 = SampleSpace.from_sequence(size=3, name="Omega2")
        F1 = SigmaAlgebra(
            domain=Omega1,
            name="F1",
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )
        F2 = SigmaAlgebra(
            domain=Omega2,
            name="F2",
            mapping={
                0: 4,
                1: 4,
                2: 1,
            },
        )
        P1 = ProbabilityMeasure(
            domain=F1,
            name="P1",
            mapping={
                0: 0.25,
                1: 0.75,
            },
        )
        P2 = ProbabilityMeasure(
            domain=F2,
            name="P2",
            mapping={
                4: 0.25,
                1: 0.75,
            },
        )

        assert P1 == P2


# --------------------- test probability methods --------------------- #


class TestConditional:
    def test_orthogonality_property_of_conditional(self):
        """Test that conditional probability (as a random variable) has the defining orthogonality property."""
        prob_space = ProbabilitySpace.from_rand(
            domain_size=10,
            num_atoms=4,
            sig_alg_variable_names=["x"],
            random_state=42,
        )
        P = prob_space.measure
        F = prob_space.sig_alg
        G = SigmaAlgebra.from_rand(
            super=F,
            num_atoms=3,
            random_state=42,
            variable_names=["y"],
        )

        for A, B in product(F.atoms, G.atoms):
            assert np.allclose(
                P(A & B),
                P.conditional(subset=A, given=G).integrate(subset=B),
            )

    def test_cond_exp_is_integral(self):
        """Test that a conditional expectation is equal to an integral against a conditional probability meassure."""
        prob_space = ProbabilitySpace.from_rand(
            domain_size=10,
            num_atoms=4,
            random_state=42,
        )
        X = RandomVariable.from_rand(
            *prob_space,
            random_state=42,
        )
        P = prob_space.measure
        F = prob_space.sig_alg
        G = SigmaAlgebra.from_rand(super=F, num_atoms=3, random_state=42, name="G")
        expectation = Operators.expectation(X, G)
        integral = Operators.integrate(
            X, measure=P.conditional(G, ascend=True)
        ).to_measurable_vector(sig_alg=G)

        assert P.equal_almost_surely(expectation, integral)

    def test_with_random_variables_and_pushforward(self):
        """Integration test of the given method with random variables and pushforwards."""
        Omega = SampleSpace.from_sequence(size=7)
        F = SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 2,
                4: 3,
                5: 3,
                6: 4,
            },
        )
        P = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.5,
                2: 0.2,
                3: 0.05,
                4: 0.05,
            },
        )
        X = RandomVariable(
            domain=Omega,
            sig_alg=F,
            measure=P,
            mapping={
                0: 2,
                1: -1,
                2: -1,
                3: 4,
                4: 1,
                5: 1,
                6: 5,
            },
        )
        Y = RandomVector(
            domain=Omega,
            sig_alg=F,
            measure=P,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (1, 2),
                3: (1, 2),
                4: (3, 4),
                5: (3, 4),
                6: (5, 6),
            },
            name="Y",
        )
        conditional = P.conditional(Y) >> X
        quotient = (P >> (X | Y)) / (P >> Y)

        assert all(
            conditional(**kwargs) == quotient(**kwargs)
            for kwargs in (X | Y).range.to_kwargs()
        )


class TestAreIndependent:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.25**2,
                1: 0.25 * 0.75,
                2: 0.75 * 0.25,
                3: 0.75**2,
            },
        )

    def test_are_independent_events_independent(self, F, P, Omega):
        """Test the are_independent method with independent events."""
        A = Set([0, 1], domain=Omega)
        B = Set([0, 2], domain=Omega)

        assert P.are_independent(given1=A, given2=B)

    def test_are_independent_events_dependent(self, F, P, Omega):
        """Test the are_independent method with dependent events."""
        A = Set([0, 1], domain=Omega)
        B = Set([2, 3], domain=Omega)

        assert not P.are_independent(given1=A, given2=B)

    def test_are_independent_sigma_algebras_independent(self, Omega, P):
        """Test the are_independent method for independent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 0, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 1, 2: 0, 3: 1}
        F1 = SigmaAlgebra(domain=Omega, name="F1", mapping=atom_ids_1)
        F2 = SigmaAlgebra(domain=Omega, name="F2", mapping=atom_ids_2)

        assert P.are_independent(given1=F1, given2=F2)

    def test_are_independent_sigma_algebras_dependent(self, Omega, P):
        """Test the are_independent method for dependent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 1, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 0, 2: 1, 3: 1}
        F1 = SigmaAlgebra(domain=Omega, name="F1", mapping=atom_ids_1)
        F2 = SigmaAlgebra(domain=Omega, name="F2", mapping=atom_ids_2)

        assert not P.are_independent(given1=F1, given2=F2)


class TestEqualAlmostSurely:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 1,
                1: 1,
                2: 0,
                3: 0,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 1.0,
                1: 0.0,
            },
        )

    def test_equal_almost_surely_true_on_random_variables(self, F, P):
        """Test the equal_almost_surely method returns True for random variables that are equal almost surely."""
        X = RandomVariable.with_uniform(
            sig_alg=F,
            mapping={
                0: 2,
                1: 2,
                2: 4,
                3: 4,
            },
        )
        Y = RandomVariable.with_uniform(
            sig_alg=F,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 4,
                3: 4,
            },
        )

        assert P.equal_almost_surely(X, Y)

    def test_equal_almost_surely_false_on_random_variables(self, F, P):
        """Test the equal_almost_surely method returns False for random variables that are not equal almost surely."""
        X = RandomVariable.with_uniform(
            sig_alg=F,
            mapping={
                0: 2,
                1: 2,
                2: 4,
                3: 4,
            },
        )
        Z = RandomVariable.with_uniform(
            sig_alg=F,
            name="Z",
            mapping={
                0: 2,
                1: 2,
                2: 1,
                3: 1,
            },
        )

        assert not P.equal_almost_surely(X, Z)

    def test_equal_almost_surely_true_on_random_vectors(self, F, P):
        """Test the equal_almost_surely method returns True for random vectors that are equal almost surely."""
        U = RandomVector.with_uniform(
            sig_alg=F,
            name="U",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            },
        )
        V = RandomVector.with_uniform(
            sig_alg=F,
            name="V",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            },
        )

        assert P.equal_almost_surely(U, V)

    def test_equal_almost_surely_false_on_random_vectors(self, F, P):
        """Test the equal_almost_surely method returns False for random vectors that are not equal almost surely."""
        U = RandomVector.with_uniform(
            sig_alg=F,
            name="U",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            },
        )
        W = RandomVector.with_uniform(
            sig_alg=F,
            name="W",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 1),
                3: (1, 1),
            },
        )

        assert not P.equal_almost_surely(U, W)


class TestDerivative:
    def test_change_of_variables(self):
        """Test the change-of-variables formula for Radon-Nikodym derivatives."""
        Omega = SampleSpace.from_sequence(size=10)
        F = SigmaAlgebra.from_rand(
            num_atoms=3,
            domain=Omega,
            random_state=42,
        )
        P = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.8,
                2: 0.0,
            },
        )
        Q = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.9,
                1: 0.1,
                2: 0.0,
            },
            name="Q",
        )
        dQ_dP = Q.derivative(P)
        X = RandomVariable.from_rand(
            domain=Omega,
            sig_alg=F,
            measure=P,
            random_state=42,
        )

        assert np.allclose(X.integrate(measure=Q), (X * dQ_dP).integrate(measure=P))

    def test_radon_nikodym_derivatives_and_conditional_measures(self):
        """Test the relationship between Radon-Nikodym derivatives and conditional measures."""
        Omega = SampleSpace.from_sequence(size=10)
        F = SigmaAlgebra.from_rand(
            num_atoms=3,
            domain=Omega,
            random_state=42,
        )
        P = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.8,
                2: 0.0,
            },
        )
        Q = ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.9,
                1: 0.1,
                2: 0.0,
            },
            name="Q",
        )
        dQ_dP = Q.derivative(P)
        for A in F.atoms:
            assert dQ_dP.integrate(subset=A) == Q(A)

    def test_conditional_distribution_radon_nikodym_formula(self):
        """Test the formula for the Radon-Nikodym derivative of a conditional distribution."""
        Omega = SampleSpace.from_sequence(size=50)
        F = SigmaAlgebra.from_rand(
            num_atoms=23,
            domain=Omega,
            random_state=42,
            variable_names=["A_i"],
        )
        G = SigmaAlgebra.from_rand(
            num_atoms=12,
            super=F,
            random_state=42,
            name="G",
            variable_names=["B_i"],
        )
        P = ProbabilityMeasure.from_rand(
            domain=F,
            num_null_atoms=4,
            random_state=42,
        )

        for i, B in G.atom_id_to_atom.items():
            if P(B) > 1e-8:
                Q = P.conditional(G, name="Q")(B_i=i)
                dQ_dP = Q.derivative(P)
                assert P.equal_almost_surely(dQ_dP, B.indicator / P(B))
