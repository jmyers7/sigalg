import pandas as pd
import pydantic
import pytest

from sigalg.core import Event, ProbabilityMeasure, SampleSpace, SigmaAlgebra
from sigalg.core.random_objects.random_variable import RandomVariable
from sigalg.core.random_objects.random_vector import RandomVector

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters."""
        P = ProbabilityMeasure()

        assert P.name == "P"
        assert P.sample_space is None
        assert P.sig_alg is None
        assert P.data is None
        assert P.point_probs is None
        assert P.atom_probs is None

    def test_constructor_with_custom_parameters(self):
        """Test the constructor with a custom parameters."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        Q = ProbabilityMeasure(sig_alg=F, name="Q")

        assert Q.name == "Q"
        assert Q.sample_space == Omega
        assert Q.sig_alg == F
        assert Q.data is None
        assert Q.point_probs is None
        assert Q.atom_probs is None


class TestFromDict:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 1,
                2: 0,
                3: 2,
                4: 2,
                5: 2,
            }
        )

    def test_from_dict_with_sig_alg(self, F):
        """Test from_dict with pre-existing sigma-algebra."""
        point_probs = {
            0: 0.1,
            1: 0.1,
            2: 0.2,
            3: 0.05,
            4: 0.4,
            5: 0.15,
        }
        P = ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)

        assert P.sig_alg == F
        assert P.point_probs == point_probs

    def test_from_dict_with_no_sig_alg(self):
        """Test from_dict automatically creates power-set sigma-algebra."""
        point_probs = {"a": 0.5, "b": 0.25, "c": 0.25}
        P = ProbabilityMeasure().from_dict(point_probs=point_probs)
        expected_sample_space = SampleSpace().from_list(["a", "b", "c"])
        expected_sig_alg = SigmaAlgebra.power_set(sample_space=expected_sample_space)

        assert P.sig_alg == expected_sig_alg

    def test_from_atoms(self, F):
        """Test from_atoms method."""
        atom_probs = {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }
        P = ProbabilityMeasure(sig_alg=F).from_atoms(atom_probs=atom_probs)

        assert P.sig_alg == F
        assert P.atom_probs == atom_probs
        assert P.point_probs is None

    def test_from_dict_and_from_atoms_compatibility(self, F):
        """Test compatibility between from_dict and from_atoms methods."""
        point_probs = {
            0: 0.1,
            1: 0.1,
            2: 0.2,
            3: 0.05,
            4: 0.4,
            5: 0.15,
        }
        P = ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)
        expected_atom_probs = {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }

        assert P.atom_probs == expected_atom_probs

    def test_invalid_input_probabilities_not_summing_to_1(self, F):
        """Test that probabilities not summing to 1 raises ValueError."""
        point_probs = {
            0: 0.5,
            1: 0.5,
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 0.5,
        }

        with pytest.raises(ValueError):
            ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)

    def test_invalid_input_negative_and_greater_than_one_probability(self, F):
        """Test that negative and greater than one probabilities raise ValueError."""
        point_probs = {
            0: -0.5,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 1.5,
        }

        with pytest.raises(ValueError):
            ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)

    def test_invalid_input_non_numeric_probability(self, F):
        """Test that non-numeric probabilities raise TypeError."""
        point_probs = {
            0: 1,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: "a",
        }

        with pytest.raises(TypeError):
            ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)


class TestFromAtoms:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 1,
                2: 0,
                3: 2,
                4: 2,
                5: 2,
            }
        )

    def test_from_atoms(self, F):
        """Test from_atoms method."""
        atom_probs = {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }
        P = ProbabilityMeasure(sig_alg=F).from_atoms(atom_probs=atom_probs)

        assert P.sig_alg == F
        assert P.atom_probs == atom_probs
        assert P.point_probs is None

    def test_from_dict_and_from_atoms_compatibility(self, F):
        """Test compatibility between from_dict and from_atoms methods."""
        point_probs = {
            0: 0.1,
            1: 0.1,
            2: 0.2,
            3: 0.05,
            4: 0.4,
            5: 0.15,
        }
        P = ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)
        expected_atom_probs = {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }

        assert P.atom_probs == expected_atom_probs


class TestFromPandas:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(data_name="letter").from_list(["a", "b", "c"])

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({"a": 0, "b": 0, "c": 1})

    @pytest.fixture
    def data(self):
        return pd.Series([0.4, 0.6])

    def test_from_pandas(self, F, data):
        """Test from_pandas method."""
        P = ProbabilityMeasure(sig_alg=F).from_pandas(data=data)
        expected_data = pd.Series(
            [0.4, 0.6], index=pd.Index([0, 1], name="atom ID"), name="probability"
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_from_pandas_with_no_sig_alg_raises(self, data):
        """Test from_pandas method with no sigma-algebra will create the power-set sigma-algebra."""

        with pytest.raises(
            ValueError, match="must be initialized with a sigma-algebra"
        ):
            ProbabilityMeasure().from_pandas(data=data)

    def test_from_pandas_with_misaligned_indices_raises(self, F, data):
        """Test the from_pandas method with misaligned indices raises."""
        data.index = pd.Index([1, 2])

        with pytest.raises(pydantic.ValidationError):
            ProbabilityMeasure(sig_alg=F).from_pandas(data=data)


class TestUniform:
    def test_uniform_on_power_set(self):
        """Test the uniform probability measure constructor on a power set."""
        Omega = SampleSpace().from_list(["a", "b", "c", "d"])
        F = SigmaAlgebra.power_set(Omega)
        U = ProbabilityMeasure.uniform(sig_alg=F, name="U")
        expected_probabilities = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}

        assert U.point_probs == expected_probabilities
        assert U.name == "U"

    def test_uniform_on_coarser_sigma_algebra(self):
        """Test the uniform probability measure constructor on a coarser sigma-algebra."""
        Omega = SampleSpace().from_list(["a", "b", "c", "d"])
        F = SigmaAlgebra(sample_space=Omega).from_dict({"a": 0, "b": 0, "c": 1, "d": 1})
        U = ProbabilityMeasure.uniform(sig_alg=F, name="U")
        expected_atom_probs = {0: 0.5, 1: 0.5}
        expected_point_probs = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}

        assert U.atom_probs == expected_atom_probs
        assert U.point_probs == expected_point_probs
        assert U.name == "U"

    def test_uniform_on_trivial_sigma_algebra(self):
        """Test the uniform probability measure constructor on a trivial sigma-algebra."""
        Omega = SampleSpace().from_list(["a", "b", "c", "d"])
        F = SigmaAlgebra.trivial(sample_space=Omega)
        U = ProbabilityMeasure.uniform(sig_alg=F, name="U")
        expected_atom_probs = {0: 1.0}
        expected_point_probs = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}

        assert U.atom_probs == expected_atom_probs
        assert U.point_probs == expected_point_probs
        assert U.name == "U"


# --------------------- test properties --------------------- #


class TestData:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 1,
                2: 0,
                3: 2,
                4: 2,
                5: 2,
            }
        )

    @pytest.fixture
    def point_probs(self):
        return {
            0: 0.1,
            1: 0.1,
            2: 0.2,
            3: 0.05,
            4: 0.4,
            5: 0.15,
        }

    @pytest.fixture
    def atom_probs(self):
        return {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }

    @pytest.fixture
    def P(self, F, point_probs):
        return ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)

    @pytest.fixture
    def Q(self, F, atom_probs):
        return ProbabilityMeasure(sig_alg=F, name="Q").from_atoms(atom_probs=atom_probs)

    def test_data_and_from_dict(self, P):
        """Test data property and from_dict."""
        expected_data = pd.Series(
            [0.2, 0.2, 0.6],
            index=pd.Index([1, 0, 2], name="atom ID"),
            name="probability",
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_data_and_from_atoms(self, Q):
        """Test data property and from_atoms."""
        expected_data = pd.Series(
            [0.2, 0.2, 0.6],
            index=pd.Index([1, 0, 2], name="atom ID"),
            name="probability",
        )

        pd.testing.assert_series_equal(Q.data, expected_data)

    def test_data_from_pandas(self, F):
        """Test data property and from_pandas."""
        data = pd.Series(
            [0.2, 0.2, 0.6],
            index=pd.Index([1, 0, 2], name="atom ID"),
            name="probability",
        )
        P = ProbabilityMeasure(sig_alg=F).from_pandas(data=data)

        pd.testing.assert_series_equal(P.data, data)


# --------------------- test data access methods --------------------- #


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 1,
                2: 0,
                3: 2,
                4: 2,
                5: 2,
            }
        )

    @pytest.fixture
    def point_probs(self):
        return {
            0: 0.1,
            1: 0.1,
            2: 0.2,
            3: 0.05,
            4: 0.4,
            5: 0.15,
        }

    @pytest.fixture
    def atom_probs(self):
        return {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }

    @pytest.fixture
    def P(self, F, point_probs):
        return ProbabilityMeasure(sig_alg=F).from_dict(point_probs=point_probs)

    @pytest.fixture
    def Q(self, F, atom_probs):
        return ProbabilityMeasure(sig_alg=F, name="Q").from_atoms(atom_probs=atom_probs)

    def test_call_on_event_instance(self, F, P, Q):
        """Test call method on event instances."""
        A = F.get_event([0, 1])

        assert P(A) == 0.2
        assert Q(A) == 0.2

    def test_call_on_list(self, P, Q):
        """Test call method on list of sample points."""
        assert P([3, 4, 5]) == 0.6
        assert Q([3, 4, 5]) == 0.6

    def test_call_on_sample_point(self, P, Q):
        """Test call method on a sample point."""
        assert P(2) == 0.2
        assert Q(2) == 0.2

    def test_call_on_non_measurable_set_raises(self, P):
        """Test call method on non-measurable set raises."""
        with pytest.raises(ValueError, match="do not form a measurable event"):
            P([0, 2])


# --------------------- test equality --------------------- #


class TestEquality:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})

    def test_non_equality_different_sigma_algebras(self, Omega, F):
        """Test the __eq__ method for inequality with different sigma-algebras."""
        G = SigmaAlgebra(sample_space=Omega, name="G").from_dict({0: 0, 1: 1, 2: 1})
        P1 = ProbabilityMeasure(sig_alg=F).from_dict(
            point_probs={0: 0.5, 1: 0.25, 2: 0.25}
        )
        P2 = ProbabilityMeasure(sig_alg=G).from_dict(
            point_probs={0: 0.5, 1: 0.25, 2: 0.25}
        )

        assert P1 != P2

    def test_non_equality_different_probabilities(self, F):
        """Test the __eq__ method for inequality with different probabilities."""
        P1 = ProbabilityMeasure(sig_alg=F).from_dict(
            point_probs={0: 0.6, 1: 0.3, 2: 0.1}
        )
        P2 = ProbabilityMeasure(sig_alg=F).from_dict(
            point_probs={0: 0.5, 1: 0.5, 2: 0.0}
        )

        assert P1 != P2

    def test_equality_same_probabilities_and_sigma_algebra(self, F):
        """Test the __eq__ method for equality with same probabilities and sigma algebra."""
        P1 = ProbabilityMeasure(sig_alg=F).from_dict(
            point_probs={0: 0.5, 1: 0.3, 2: 0.2}
        )
        P2 = ProbabilityMeasure(sig_alg=F).from_dict(
            point_probs={0: 0.5, 1: 0.3, 2: 0.2}
        )

        assert P1 == P2

    def test_equality_same_components_different_names(self):
        """Test the __eq__ method for equality with same components but different names."""
        Omega1 = SampleSpace(name="Omega1").from_sequence(size=3)
        Omega2 = SampleSpace(name="Omega2").from_sequence(size=3)
        F1 = SigmaAlgebra(sample_space=Omega1, name="F1").from_dict({0: 0, 1: 0, 2: 1})
        F2 = SigmaAlgebra(sample_space=Omega2, name="F2").from_dict({0: 4, 1: 4, 2: 1})
        P1 = ProbabilityMeasure(sig_alg=F1, name="P1").from_dict(
            point_probs={0: 0.5, 1: 0.25, 2: 0.25}
        )
        P2 = ProbabilityMeasure(sig_alg=F2, name="P2").from_dict(
            point_probs={0: 0.5, 1: 0.25, 2: 0.25}
        )

        assert P1 == P2


# --------------------- test probability methods --------------------- #


class TestConditionalProbability:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=7)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
                4: 2,
                5: 3,
                6: 3,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            point_probs={
                0: 0.0,
                1: 0.3,
                2: 0.05,
                3: 0.1,
                4: 0.15,
                5: 0.25,
                6: 0.15,
            }
        )

    def test_conditional_probability_subset_of_conditioning_event(self, F, P):
        """Test conditional_probability method when event A is subset of B."""
        A = Event(sig_alg=F).from_list([1, 2])
        B = Event(sig_alg=F, name="B").from_list([1, 2, 3, 4])
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9

    def test_conditional_probability_non_trivial_overlap(self, F, P):
        """Test conditional_probability method with non-trivial overlap."""
        A = Event(sig_alg=F).from_list([1, 2, 3, 4])
        B = Event(sig_alg=F, name="B").from_list([3, 4, 5, 6])
        result = P.conditional_probability(A, B)
        expected = P(A & B) / P(B)

        assert abs(result - expected) < 1e-9

    def test_conditional_probability_no_overlap(self, F, P):
        """Test conditional_probability method with no overlap."""
        A = Event(sig_alg=F).from_list([1, 2])
        B = Event(sig_alg=F, name="B").from_list([3, 4])
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
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        probabilities = {0: 0.25**2, 1: 0.25 * 0.75, 2: 0.75 * 0.25, 3: 0.75**2}
        return ProbabilityMeasure(sig_alg=F).from_dict(point_probs=probabilities)

    def test_are_independent_events_independent(self, F, P):
        """Test the are_independent method with independent events."""
        A = Event(sig_alg=F).from_list([0, 1])
        B = Event(sig_alg=F).from_list([0, 2])
        result = P.are_independent(event1=A, event2=B)

        assert result

    def test_are_independent_events_dependent(self, F, P):
        """Test the are_independent method with dependent events."""
        A = Event(sig_alg=F).from_list([0, 1])
        B = Event(sig_alg=F).from_list([2, 3])
        result = P.are_independent(event1=A, event2=B)

        assert not result

    def test_are_independent_sigma_algebras_independent(self, Omega, P):
        """Test the are_independent method for independent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 0, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 1, 2: 0, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega, name="F1").from_dict(
            sample_id_to_atom_id=atom_ids_1
        )
        F2 = SigmaAlgebra(sample_space=Omega, name="F2").from_dict(
            sample_id_to_atom_id=atom_ids_2
        )
        result = P.are_independent(algebra1=F1, algebra2=F2)

        assert result

    def test_are_independent_sigma_algebras_dependent(self, Omega, P):
        """Test the are_independent method for dependent sigma algebras."""
        atom_ids_1 = {0: 0, 1: 1, 2: 1, 3: 1}
        atom_ids_2 = {0: 0, 1: 0, 2: 1, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega, name="F1").from_dict(
            sample_id_to_atom_id=atom_ids_1
        )
        F2 = SigmaAlgebra(sample_space=Omega, name="F2").from_dict(
            sample_id_to_atom_id=atom_ids_2
        )
        result = P.are_independent(algebra1=F1, algebra2=F2)

        assert not result

    def test_are_independent_raises_for_both_events_and_algebras(self):
        """Test that are_independent raises ValueError when both events and algebras are provided."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        probabilities = {0: 0.5, 1: 0.5}
        P = ProbabilityMeasure(sig_alg=F).from_dict(point_probs=probabilities)
        A = Event(sig_alg=F).from_list([0])
        B = Event(sig_alg=F).from_list([1])
        F1 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 1}
        )
        F2 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 1}
        )

        with pytest.raises(ValueError, match="Must provide exactly one"):
            P.are_independent(event1=A, event2=B, algebra1=F1, algebra2=F2)

    def test_are_independent_raises_for_neither_events_nor_algebras(self):
        """Test that are_independent raises ValueError when neither events nor algebras are provided."""
        Omega = SampleSpace().from_sequence(size=2)
        F = SigmaAlgebra.power_set(Omega)
        probabilities = {0: 0.5, 1: 0.5}
        P = ProbabilityMeasure(sig_alg=F).from_dict(point_probs=probabilities)

        with pytest.raises(ValueError, match="Must provide exactly one"):
            P.are_independent()


class TestAlmostSurelyEqual:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 1,
                2: 0,
                3: 0,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            point_probs={
                0: 0.0,
                1: 0.0,
                2: 0.9,
                3: 0.1,
            }
        )

    def test_almost_surely_equal_true_on_random_variables(self, F, P):
        """Test the almost_surely_equal method returns True for random variables that are equal almost surely."""
        X = RandomVariable(sig_alg=F).from_dict(
            {
                0: 2,
                1: 2,
                2: 4,
                3: 4,
            }
        )
        Y = RandomVariable(sig_alg=F, name="Y").from_dict(
            {
                0: 1,
                1: 1,
                2: 4,
                3: 4,
            }
        )

        assert P.almost_surely_equal(X, Y)

    def test_almost_surely_equal_false_on_random_variables(self, F, P):
        """Test the almost_surely_equal method returns False for random variables that are not equal almost surely."""
        X = RandomVariable(sig_alg=F).from_dict(
            {
                0: 2,
                1: 2,
                2: 4,
                3: 4,
            }
        )
        Z = RandomVariable(sig_alg=F, name="Z").from_dict(
            {
                0: 2,
                1: 2,
                2: 1,
                3: 1,
            }
        )

        assert not P.almost_surely_equal(X, Z)

    def test_almost_surely_equal_true_on_random_vectors(self, F, P):
        """Test the almost_surely_equal method returns True for random vectors that are equal almost surely."""
        U = RandomVector(sig_alg=F, name="U").from_dict(
            {
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            }
        )
        V = RandomVector(sig_alg=F, name="V").from_dict(
            {
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            }
        )

        assert P.almost_surely_equal(U, V)

    def test_almost_surely_equal_false_on_random_vectors(self, F, P):
        """Test the almost_surely_equal method returns False for random vectors that are not equal almost surely."""
        U = RandomVector(sig_alg=F, name="U").from_dict(
            {
                0: (2, 1),
                1: (2, 1),
                2: (1, 4),
                3: (1, 4),
            }
        )
        W = RandomVector(sig_alg=F, name="W").from_dict(
            {
                0: (2, 1),
                1: (2, 1),
                2: (1, 1),
                3: (1, 1),
            }
        )

        assert not P.almost_surely_equal(U, W)
