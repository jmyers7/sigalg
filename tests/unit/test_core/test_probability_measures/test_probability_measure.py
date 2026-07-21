from itertools import product

import numpy as np
import pandas as pd
import pytest

from sigalg.core import (
    Event,
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
)
from sigalg.core.random_objects.random_variable import RandomVariable
from sigalg.core.random_objects.random_vector import RandomVector

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_with_no_parameters(self):
        """Test the constructor with no parameters."""
        P = ProbabilityMeasure()

        assert P.name == "P"
        assert P.sample_space is None
        assert P.sig_alg is None
        assert P.domain is None
        assert P.data is None

    def test_from_dict_with_valid_sig_alg(self):
        """Test from dict with a valid sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=6)
        F = SigmaAlgebra(
            sample_space=Omega,
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
            sig_alg=F,
            mapping=mapping,
            name="Q",
        )

        assert Q.name == "Q"
        assert Q.sample_space is Omega
        assert Q.sig_alg is F
        assert Q.domain is F.atom_space
        assert Q.data is not None

    def test_from_pandas_with_valid_sig_alg(self):
        """Test from pandas with a valid sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=6)
        F = SigmaAlgebra(
            sample_space=Omega,
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
            sig_alg=F,
            mapping=mapping,
            name="Q",
        )
        expected_data = pd.Series(
            [0.2, 0.2, 0.6],
            index=pd.Index([1, 0, 2], name="atom_ID"),
            name="probability",
        )

        assert Q.name == "Q"
        assert Q.sample_space is Omega
        assert Q.sig_alg is F
        assert Q.domain is F.atom_space
        pd.testing.assert_series_equal(Q.data, expected_data)


class TestUniform:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(["a", "b", "c", "d"])

    def test_on_power_set(self, Omega):
        """Test the uniform probability measure constructor on a power set."""
        U = ProbabilityMeasure.uniform(sample_space=Omega)
        expected_probs = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        expected_data = pd.Series(
            [0.25, 0.25, 0.25, 0.25],
            index=pd.Index(["a", "b", "c", "d"], name="sample"),
            name="probability",
        )

        assert U.probs == expected_probs
        pd.testing.assert_series_equal(U.data, expected_data)
        assert U.name == "U"

    def test_on_coarser_sigma_algebra(self, Omega):
        """Test the uniform probability measure constructor on a coarser sigma-algebra."""
        F = SigmaAlgebra(sample_space=Omega, mapping={"a": 0, "b": 0, "c": 1, "d": 1})
        K = ProbabilityMeasure.uniform(sig_alg=F, name="K")
        expected_probs = {0: 0.5, 1: 0.5}
        expected_data = pd.Series(
            [0.5, 0.5], index=pd.Index([0, 1], name="atom_ID"), name="probability"
        )

        assert K.probs == expected_probs
        pd.testing.assert_series_equal(K.data, expected_data)
        assert K.name == "K"

    def test_on_trivial_sigma_algebra(self, Omega):
        """Test the uniform probability measure constructor on a trivial sigma-algebra."""
        F = SigmaAlgebra.trivial(sample_space=Omega, name="F")
        U = ProbabilityMeasure.uniform(sig_alg=F)
        expected_probs = {0: 1.0}
        expected_data = pd.Series(
            [1.0], index=pd.Index([0], name="atom_ID"), name="probability"
        )

        assert U.probs == expected_probs
        pd.testing.assert_series_equal(U.data, expected_data)
        assert U.name == "U"


# --------------------- test properties --------------------- #


class TestSigAlg:
    def test_sig_alg_setter_with_1_dim_sig_alg(self):
        """Test the sig_alg setter with a 1-dimensional sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 1,
                1: 0,
                2: 1,
                3: 2,
            },
            variable_names=["x"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 1,
                1: 1,
                2: 1,
                3: 2,
            },
            variable_names=["y"],
            name="G",
        )

        def mapping(*, x):  # noqa: D103
            if x == 0:
                return 0.1
            elif x == 1:
                return 0.4
            elif x == 2:
                return 0.5

        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)
        P.sig_alg = G
        expected_domain = pd.Index([1, 2], name="y")
        expected_data = pd.Series([0.5, 0.5], index=expected_domain, name="probability")

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_from_2_dim_to_1_dim(self):
        """Test the sig_alg setter from 2D to 1D sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "a"),
                1: ("a", "b"),
                2: ("b", "c"),
                3: ("b", "d"),
            },
            variable_names=["F_0", "F_1"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: "x",
                1: "x",
                2: "y",
                3: "y",
            },
            variable_names=["z"],
            name="G",
        )

        def mapping(*, F_0, F_1):  # noqa: D103
            if (F_0, F_1) == ("a", "a"):
                return 0.1
            elif (F_0, F_1) == ("a", "b"):
                return 0.2
            elif (F_0, F_1) == ("b", "c"):
                return 0.3
            elif (F_0, F_1) == ("b", "d"):
                return 0.4

        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)
        P.sig_alg = G
        expected_domain = pd.Index(["x", "y"], name="z")
        expected_data = pd.Series([0.3, 0.7], index=expected_domain, name="probability")

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_with_2_dim_sig_alg(self):
        """Test the sig_alg setter with a 2-dimensional sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=6)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("x", "a"),
                1: ("x", "b"),
                2: ("y", "c"),
                3: ("y", "d"),
                4: ("z", "e"),
                5: ("z", "f"),
            },
            variable_names=["F_0", "F_1"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("x", "a"),
                1: ("x", "a"),
                2: ("y", "b"),
                3: ("y", "b"),
                4: ("z", "c"),
                5: ("z", "c"),
            },
            variable_names=["G_0", "G_1"],
            name="G",
        )

        def mapping(*, F_0, F_1):  # noqa: D103
            if (F_0, F_1) == ("x", "a"):
                return 0.1
            elif (F_0, F_1) == ("x", "b"):
                return 0.15
            elif (F_0, F_1) == ("y", "c"):
                return 0.2
            elif (F_0, F_1) == ("y", "d"):
                return 0.25
            elif (F_0, F_1) == ("z", "e"):
                return 0.15
            elif (F_0, F_1) == ("z", "f"):
                return 0.15

        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)
        P.sig_alg = G
        expected_domain = pd.MultiIndex.from_tuples(
            [("x", "a"), ("y", "b"), ("z", "c")],
            names=["G_0", "G_1"],
        )
        expected_data = pd.Series(
            [0.25, 0.45, 0.3], index=expected_domain, name="probability"
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_from_3_dim_to_1_dim(self):
        """Test the sig_alg setter from 3D to 1D sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=4)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "x", 1),
                1: ("a", "y", 2),
                2: ("b", "x", 1),
                3: ("b", "y", 3),
            },
            variable_names=["F_0", "F_1", "F_2"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: "A",
                1: "A",
                2: "B",
                3: "B",
            },
            variable_names=["w"],
            name="G",
        )

        def mapping(*, F_0, F_1, F_2):  # noqa: D103
            if (F_0, F_1, F_2) == ("a", "x", 1):
                return 0.3
            elif (F_0, F_1, F_2) == ("a", "y", 2):
                return 0.4
            elif (F_0, F_1, F_2) == ("b", "x", 1):
                return 0.2
            elif (F_0, F_1, F_2) == ("b", "y", 3):
                return 0.1

        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)
        P.sig_alg = G
        expected_domain = pd.Index(["A", "B"], name="w")
        expected_data = pd.Series([0.7, 0.3], index=expected_domain, name="probability")

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_sig_alg_setter_with_3_dim_sig_alg(self):
        """Test the sig_alg setter with a 3-dimensional sigma-algebra."""
        Omega = SampleSpace.from_sequence(size=6)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "x", 1),
                1: ("a", "x", 2),
                2: ("a", "y", 3),
                3: ("b", "x", 4),
                4: ("b", "y", 5),
                5: ("b", "y", 6),
            },
            variable_names=["F_0", "F_1", "F_2"],
        )
        G = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: ("a", "x", 1),
                1: ("a", "x", 1),
                2: ("a", "y", 2),
                3: ("b", "x", 3),
                4: ("b", "y", 4),
                5: ("b", "y", 4),
            },
            variable_names=["G_0", "G_1", "G_2"],
            name="G",
        )

        def mapping(*, F_0, F_1, F_2):  # noqa: D103
            if (F_0, F_1, F_2) == ("a", "x", 1):
                return 0.05
            elif (F_0, F_1, F_2) == ("a", "x", 2):
                return 0.15
            elif (F_0, F_1, F_2) == ("a", "y", 3):
                return 0.3
            elif (F_0, F_1, F_2) == ("b", "x", 4):
                return 0.2
            elif (F_0, F_1, F_2) == ("b", "y", 5):
                return 0.1
            elif (F_0, F_1, F_2) == ("b", "y", 6):
                return 0.2

        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)
        P.sig_alg = G
        expected_domain = pd.MultiIndex.from_tuples(
            [("a", "x", 1), ("a", "y", 2), ("b", "x", 3), ("b", "y", 4)],
            names=["G_0", "G_1", "G_2"],
        )
        expected_data = pd.Series(
            [0.2, 0.3, 0.2, 0.3], index=expected_domain, name="probability"
        )

        pd.testing.assert_series_equal(P.data, expected_data)


class TestSampleSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            },
        )

    @pytest.fixture
    def mapping(self):
        return {
            0: 0.2,
            1: 0.3,
            2: 0.5,
        }

    def test_getter_on_prob_measure_with_data(self, Omega, F, mapping):
        """Test the sample_space getter on a ProbabilityMeasure instance with data."""
        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)

        assert P.sample_space == Omega

    def test_setter(self, F, mapping):
        """Test the sample_space setter on a ProbabilityMeasure instance from probabilities."""
        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)
        S = SampleSpace(["a", "b", "c", "d"], name="S")
        P.sample_space = S
        expected_data = pd.Series(
            [0.2, 0.3, 0.5], index=F.atom_space.data, name="probability"
        )

        assert P.sample_space == S
        pd.testing.assert_series_equal(P.data, expected_data)


# --------------------- test data access methods --------------------- #


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
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
            sample_space=Omega,
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
        return ProbabilityMeasure(sig_alg=F, mapping=mapping_F)

    @pytest.fixture
    def Q(self, G, mapping_G):
        return ProbabilityMeasure(sig_alg=G, mapping=mapping_G, name="G")

    def test_on_event(self, F, G, P, Q):
        """Test call method on event instances."""
        A = F.get_event([0, 1])
        B = G.get_event([0, 1], name="B")
        C = F.get_event([2, 3, 4, 5], name="C")
        D = G.get_event([2, 3, 4, 5], name="D")

        assert P(event=A) == 0.2
        assert P(A) == 0.2
        assert Q(event=B) == 0.2
        assert Q(B) == 0.2
        assert P(event=C) == 0.8
        assert P(C) == 0.8
        assert Q(event=D) == 0.8
        assert Q(D) == 0.8

    def test_on_list(self, P, Q):
        """Test call method on list of sample points."""
        assert P(event=[0, 1]) == 0.2
        assert P([0, 1]) == 0.2
        assert Q(event=[0, 1]) == 0.2
        assert Q([0, 1]) == 0.2

    def test_on_sample_point(self, P, Q):
        """Test call method on single sample point."""
        assert P(sample_point=2) == 0.2
        assert P(2) == 0.2
        assert Q(sample_point=2) == 0.2
        assert Q(2) == 0.2

    def test_on_atom_id(self, P, Q):
        """Test call method on atom ID."""
        assert P(F=2) == 0.6
        assert Q(G_0=2, G_1=4) == 0.6

    def test_curry(self, Q):
        """Test the curried call method on atom ID."""
        assert Q(G_0=2)(G_1=4) == 0.6
        assert Q(G_1=4)(G_0=2) == 0.6

    def test_non_measurable_event_raises(self, Omega, P):
        power_set = SigmaAlgebra.power_set(Omega)
        A = power_set.get_event([2, 3])

        with pytest.raises(ValueError, match="event is not measurable"):
            P([2, 3])

        with pytest.raises(
            ValueError, match="Event is not in the domain of the probability measure"
        ):
            P(event=A)


# --------------------- test equality --------------------- #


class TestEquality:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )

    def test_non_equality_different_sigma_algebras(self, Omega, F):
        """Test the __eq__ method for inequality with different sigma-algebras."""
        G = SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 1,
                2: 1,
            },
        )
        P1 = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.5,
                1: 0.5,
            },
        )
        P2 = ProbabilityMeasure(
            sig_alg=G,
            mapping={
                0: 0.5,
                1: 0.5,
            },
        )

        assert P1 != P2

    def test_non_equality_different_probabilities(self, F):
        """Test the __eq__ method for inequality with different probabilities."""
        P1 = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.6,
                1: 0.4,
            },
        )
        P2 = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.5,
                1: 0.5,
            },
        )

        assert P1 != P2

    def test_equality_same_probabilities_and_sigma_algebra(self, F):
        """Test the __eq__ method for equality with same probabilities and sigma algebra."""
        P1 = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.7,
                1: 0.3,
            },
        )
        P2 = ProbabilityMeasure(
            sig_alg=F,
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
            sample_space=Omega1,
            name="F1",
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )
        F2 = SigmaAlgebra(
            sample_space=Omega2,
            name="F2",
            mapping={
                0: 4,
                1: 4,
                2: 1,
            },
        )
        P1 = ProbabilityMeasure(
            sig_alg=F1,
            name="P1",
            mapping={
                0: 0.25,
                1: 0.75,
            },
        )
        P2 = ProbabilityMeasure(
            sig_alg=F2,
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
            sample_space_size=10,
            num_atoms=4,
            sig_alg_variable_names=["x"],
            random_state=42,
        )
        P = prob_space.prob_measure
        F = prob_space.sig_alg
        G = SigmaAlgebra.from_rand(
            super=F,
            num_atoms=3,
            random_state=42,
            variable_names=["y"],
        )

        for A, B in product(F.to_atoms, G.to_atoms):
            assert np.allclose(
                P(A & B), P.conditional(event=A, given=G).integrate(event=B)
            )


class TestGiven:
    def test_cond_exp_is_integral(self):
        """Test that a conditional expectation is equal to an integral against a conditional probability meassure."""
        prob_space = ProbabilitySpace.from_rand(
            sample_space_size=10,
            num_atoms=4,
            sig_alg_variable_names=["x"],
            random_state=42,
        )
        X = RandomVariable.from_randnorm(
            *prob_space,
            random_state=42,
        )
        P = prob_space.prob_measure
        F = prob_space.sig_alg
        G = SigmaAlgebra.from_rand(
            super=F,
            num_atoms=3,
            random_state=42,
            variable_names=["y"],
        )

        for id in G.atom_ids:
            B = G.atom_id_to_event[id]
            assert X.integrate(prob_measure=P.given(G)(y=id)) == X.expectation(
                sig_alg=G
            )(B)


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
            sig_alg=F,
            mapping={
                0: 0.25**2,
                1: 0.25 * 0.75,
                2: 0.75 * 0.25,
                3: 0.75**2,
            },
        )

    def test_are_independent_events_independent(self, F, P):
        """Test the are_independent method with independent events."""
        A = Event.from_list([0, 1], sig_alg=F)
        B = Event.from_list([0, 2], sig_alg=F)

        assert P.are_independent(event1=A, event2=B)

    def test_are_independent_events_dependent(self, F, P):
        """Test the are_independent method with dependent events."""
        A = Event.from_list([0, 1], sig_alg=F)
        B = Event.from_list([2, 3], sig_alg=F)

        assert not P.are_independent(event1=A, event2=B)

    def test_are_independent_sigma_algebras_independent(self, Omega, P):
        """Test the are_independent method for independent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 0, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 1, 2: 0, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega, name="F1", mapping=atom_ids_1)
        F2 = SigmaAlgebra(sample_space=Omega, name="F2", mapping=atom_ids_2)

        assert P.are_independent(algebra1=F1, algebra2=F2)

    def test_are_independent_sigma_algebras_dependent(self, Omega, P):
        """Test the are_independent method for dependent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 1, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 0, 2: 1, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega, name="F1", mapping=atom_ids_1)
        F2 = SigmaAlgebra(sample_space=Omega, name="F2", mapping=atom_ids_2)

        assert not P.are_independent(algebra1=F1, algebra2=F2)

    def test_are_independent_raises_for_both_events_and_algebras(self):
        """Test that are_independent raises ValueError when both events and algebras are provided."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.5,
                1: 0.5,
            },
        )
        A = Event.from_list([0], sig_alg=F)
        B = Event.from_list([1], sig_alg=F)
        F1 = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
            },
        )
        F2 = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
            },
        )

        with pytest.raises(ValueError, match="Must provide exactly one"):
            P.are_independent(event1=A, event2=B, algebra1=F1, algebra2=F2)

    def test_are_independent_raises_for_neither_events_nor_algebras(self):
        """Test that are_independent raises ValueError when neither events nor algebras are provided."""
        Omega = SampleSpace.from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.5,
                1: 0.5,
            },
        )

        with pytest.raises(ValueError, match="Must provide exactly one"):
            P.are_independent()


class TestAlmostSurelyEqual:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
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
            sig_alg=F,
            mapping={
                0: 1.0,
                1: 0.0,
            },
        )

    def test_almost_surely_equal_true_on_random_variables(self, F, P):
        """Test the almost_surely_equal method returns True for random variables that are equal almost surely."""
        X = RandomVariable(
            sig_alg=F,
            mapping={
                0: 2,
                1: 2,
                2: 4,
                3: 4,
            },
        )
        Y = RandomVariable(
            sig_alg=F,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 4,
                3: 4,
            },
        )

        assert P.almost_surely_equal(X, Y)

    def test_almost_surely_equal_false_on_random_variables(self, F, P):
        """Test the almost_surely_equal method returns False for random variables that are not equal almost surely."""
        X = RandomVariable(
            sig_alg=F,
            mapping={
                0: 2,
                1: 2,
                2: 4,
                3: 4,
            },
        )
        Z = RandomVariable(
            sig_alg=F,
            name="Z",
            mapping={
                0: 2,
                1: 2,
                2: 1,
                3: 1,
            },
        )

        assert not P.almost_surely_equal(X, Z)

    def test_almost_surely_equal_true_on_random_vectors(self, F, P):
        """Test the almost_surely_equal method returns True for random vectors that are equal almost surely."""
        U = RandomVector(
            sig_alg=F,
            name="U",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            },
        )
        V = RandomVector(
            sig_alg=F,
            name="V",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            },
        )

        assert P.almost_surely_equal(U, V)

    def test_almost_surely_equal_false_on_random_vectors(self, F, P):
        """Test the almost_surely_equal method returns False for random vectors that are not equal almost surely."""
        U = RandomVector(
            sig_alg=F,
            name="U",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            },
        )
        W = RandomVector(
            sig_alg=F,
            name="W",
            mapping={
                0: (2, 1),
                1: (2, 1),
                2: (1, 1),
                3: (1, 1),
            },
        )

        assert not P.almost_surely_equal(U, W)
