import pandas as pd
import pytest

from sigalg.core import Event, ProbabilityMeasure, SampleSpace, SigmaAlgebra
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
    def G(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            name="G",
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 1,
            },
        )

    @pytest.fixture
    def mapping(self):
        return {
            0: 0.2,
            1: 0.3,
            2: 0.5,
        }

    def test_getter_on_prob_measure_with_data(self, F, mapping):
        """Test the sig_alg getter on a ProbabilityMeasure instance with data."""
        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)

        assert P.sig_alg == F

    def test_setter(self, F, G, mapping):
        """Test the sig_alg setter on a ProbabilityMeasure instance with data."""
        P = ProbabilityMeasure(sig_alg=F, mapping=mapping)
        data_new = pd.Series(
            [0.2, 0.8], index=pd.Index([0, 1], name="atom_ID"), name="probability"
        )
        P.sig_alg = G

        assert P.sig_alg == G
        pd.testing.assert_series_equal(P.data, data_new)


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


class TestConditionalProbability:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=7)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 1,
                3: 2,
                4: 2,
                5: 3,
                6: 3,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.0,
                1: 0.35,
                2: 0.25,
                3: 0.4,
            },
        )

    def test_conditional_probability_subset_of_conditioning_event(self, F, P):
        """Test conditional_probability method when event A is subset of B."""
        A = Event.from_list([1, 2], sig_alg=F)
        B = Event.from_list([1, 2, 3, 4], sig_alg=F, name="B")
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9

    def test_conditional_probability_non_trivial_overlap(self, F, P):
        """Test conditional_probability method with non-trivial overlap."""
        A = Event.from_list([1, 2, 3, 4], sig_alg=F)
        B = Event.from_list([3, 4, 5, 6], sig_alg=F, name="B")
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9

    def test_conditional_probability_no_overlap(self, F, P):
        """Test conditional_probability method with no overlap."""
        A = Event.from_list([1, 2], sig_alg=F)
        B = Event.from_list([3, 4], sig_alg=F, name="B")
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9
        assert abs(expected) < 1e-9

    def test_conditioning_on_impossible_event(self, F, P):
        """Test that conditional_probability raises ValueError when P(B) = 0."""
        A = F.get_event([1, 2])
        B = F.get_event([0])

        with pytest.raises(ValueError, match="given event with probability 0"):
            P.conditional_probability(A, B)


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
