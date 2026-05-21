import pandas as pd
import pytest

from sigalg.core import Event, ProbabilityMeasure, SampleSpace, SigmaAlgebra
from sigalg.core.random_objects.random_variable import RandomVariable
from sigalg.core.random_objects.random_vector import RandomVector

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_with_valid_parameters(self):
        """Test the constructor with valid parameters."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0, 2: 1})
        Q = ProbabilityMeasure(sig_alg=F, name="Q")

        assert Q.name == "Q"
        assert Q.sample_space == Omega
        assert Q.sig_alg == F
        assert Q.domain == F.atom_space
        assert Q.data is None


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

    @pytest.fixture
    def probs(self):
        return {
            0: 0.2,
            1: 0.2,
            2: 0.6,
        }

    def test_with_valid_sig_alg(self, F, probs):
        """Test from_dict with a valid sigma-algebra."""
        P = ProbabilityMeasure(sig_alg=F).from_dict(probs=probs)

        assert P.sig_alg == F
        assert P.probs == probs


class TestFromPandas:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_list(["a", "b", "c"], data_name=["letter"])

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({"a": 0, "b": 0, "c": 1})

    @pytest.fixture
    def data(self):
        return pd.Series([0.4, 0.6])

    def test_with_valid_sig_alg(self, F, data):
        """Test from_pandas with a valid sigma-algebra."""
        P = ProbabilityMeasure(sig_alg=F).from_pandas(data=data)
        expected_data = pd.Series(
            [0.4, 0.6], index=pd.Index([0, 1], name="F"), name="probability"
        )

        assert P.sig_alg == F
        pd.testing.assert_series_equal(P.data, expected_data)


class TestUniform:
    def test_on_power_set(self):
        """Test the uniform probability measure constructor on a power set."""
        Omega = SampleSpace().from_list(["a", "b", "c", "d"])
        F = SigmaAlgebra.power_set(Omega, name="F")
        U = ProbabilityMeasure.uniform(sig_alg=F)
        expected_probs = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        expected_data = pd.Series(
            [0.25, 0.25, 0.25, 0.25],
            index=pd.Index(["a", "b", "c", "d"], name="F"),
            name="probability",
        )

        assert U.probs == expected_probs
        pd.testing.assert_series_equal(U.data, expected_data)
        assert U.name == "U"

    def test_on_coarser_sigma_algebra(self):
        """Test the uniform probability measure constructor on a coarser sigma-algebra."""
        Omega = SampleSpace().from_list(["a", "b", "c", "d"])
        F = SigmaAlgebra(sample_space=Omega).from_dict({"a": 0, "b": 0, "c": 1, "d": 1})
        K = ProbabilityMeasure.uniform(sig_alg=F, name="K")
        expected_probs = {0: 0.5, 1: 0.5}
        expected_data = pd.Series(
            [0.5, 0.5], index=pd.Index([0, 1], name="F"), name="probability"
        )

        assert K.probs == expected_probs
        pd.testing.assert_series_equal(K.data, expected_data)
        assert K.name == "K"

    def test_on_trivial_sigma_algebra(self):
        """Test the uniform probability measure constructor on a trivial sigma-algebra."""
        Omega = SampleSpace().from_list(["a", "b", "c", "d"])
        F = SigmaAlgebra.trivial(sample_space=Omega, name="F")
        U = ProbabilityMeasure.uniform(sig_alg=F)
        expected_probs = {0: 1.0}
        expected_data = pd.Series(
            [1.0], index=pd.Index([0], name="F"), name="probability"
        )

        assert U.probs == expected_probs
        pd.testing.assert_series_equal(U.data, expected_data)
        assert U.name == "U"


# --------------------- test properties --------------------- #


class TestSigAlg:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            }
        )

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 1,
            }
        )

    @pytest.fixture
    def atom_probs(self):
        return {
            0: 0.2,
            1: 0.3,
            2: 0.5,
        }

    def test_sig_alg_getter_on_prob_measure_with_data(self, F, atom_probs):
        """Test the sig_alg getter on a ProbabilityMeasure instance with data."""
        P = ProbabilityMeasure(sig_alg=F).from_dict(atom_probs)

        assert P.sig_alg == F

    def test_sig_alg_getter_on_prob_measure_without_data(self, F):
        """Test sig_alg getter on a ProbabilityMeasure instance without data."""
        P = ProbabilityMeasure(sig_alg=F)

        assert P.sig_alg == F

    def test_sig_alg_setter_on_empty_prob_measure(self, F):
        """Test the sig_alg setter on an empty ProbabilityMeasure instance."""
        P = ProbabilityMeasure()
        P.sig_alg = F

        assert P.sig_alg == F

    def test_sig_alg_setter_on_prob_measure_with_data(self, F, G, atom_probs):
        """Test the sig_alg setter on a ProbabilityMeasure instance with data."""
        P = ProbabilityMeasure(sig_alg=F).from_dict(atom_probs)
        data_new = pd.Series(
            [0.2, 0.8], index=pd.Index([0, 1], name="atom ID"), name="probability"
        )
        atom_probs_new = {0: 0.2, 1: 0.8}
        P.sig_alg = G

        assert P.sig_alg == G
        pd.testing.assert_series_equal(P.data, data_new)
        assert P.atom_probs == atom_probs_new


class TestSampleSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            }
        )

    @pytest.fixture
    def atom_probs(self):
        return {
            0: 0.2,
            1: 0.3,
            2: 0.5,
        }

    @pytest.fixture
    def point_probs(self):
        return {
            0: 0.2,
            1: 0.3,
            2: 0.2,
            3: 0.3,
        }

    @pytest.fixture
    def data(self):
        return pd.Series(
            [0.2, 0.3, 0.5],
            index=pd.Index([0, 1, 2], name="atom ID"),
            name="probability",
        )

    def test_sample_space_getter_on_prob_measure_with_data(self, Omega, F, atom_probs):
        """Test the sample_space getter on a ProbabilityMeasure instance with data."""
        P = ProbabilityMeasure(sig_alg=F).from_dict(atom_probs)

        assert P.sample_space == Omega

    def test_sample_space_getter_on_prob_measure_without_data(self, Omega, F):
        """Test sample_space getter on a ProbabilityMeasure instance without data."""
        P = ProbabilityMeasure(sig_alg=F)

        assert P.sample_space == Omega

    def test_sample_space_setter_on_empty_prob_measure_raises(self, Omega):
        """Test the sample_space setter on an empty ProbabilityMeasure instance."""
        P = ProbabilityMeasure()

        with pytest.raises(
            ValueError,
            match="Cannot set sample space when sig_alg is not set.",
        ):
            P.sample_space = Omega

    def test_sample_space_setter_on_prob_measure_from_atom_probs(
        self, F, atom_probs, data
    ):
        """Test the sample_space setter on a ProbabilityMeasure instance from atom probabilities."""
        P = ProbabilityMeasure(sig_alg=F).from_dict(atom_probs)
        Omega_new = SampleSpace().from_list(["a", "b", "c", "d"])
        P.sample_space = Omega_new

        assert P.sample_space == Omega_new
        assert P.atom_probs == atom_probs
        assert P.point_probs is None
        pd.testing.assert_series_equal(P.data, data)

    def test_sample_space_setter_on_prob_measure_from_point_probs(
        self, F, atom_probs, point_probs, data
    ):
        """Test the sample_space setter on a ProbabilityMeasure instance from point probabilities."""
        P = ProbabilityMeasure(sig_alg=F).from_dict(point_probs, type="point")
        Omega_new = SampleSpace().from_list(["a", "b", "c", "d"])
        point_probs_new = dict(zip(Omega_new.data, point_probs.values()))
        P.sample_space = Omega_new

        assert P.sample_space == Omega_new
        assert P.atom_probs == atom_probs
        assert P.point_probs == point_probs_new
        pd.testing.assert_series_equal(P.data, data)


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
        return ProbabilityMeasure(sig_alg=F).from_dict(probs=point_probs, type="point")

    @pytest.fixture
    def Q(self, F, atom_probs):
        return ProbabilityMeasure(sig_alg=F, name="Q").from_dict(probs=atom_probs)

    def test_data_and_from_dict_with_point_type(self, P):
        """Test data property and from_dict with type='point'."""
        expected_data = pd.Series(
            [0.2, 0.2, 0.6],
            index=pd.Index([1, 0, 2], name="atom ID"),
            name="probability",
        )

        pd.testing.assert_series_equal(P.data, expected_data)

    def test_data_and_from_dict_with_atom_type(self, Q):
        """Test data property and from_dict with type='atom'."""
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


class TestPointData:
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
        return ProbabilityMeasure(sig_alg=F).from_dict(probs=point_probs, type="point")

    @pytest.fixture
    def Q(self, F, atom_probs):
        return ProbabilityMeasure(sig_alg=F, name="Q").from_dict(probs=atom_probs)

    def test_point_data_and_from_dict_with_point_type(self, P):
        """Test point_data property and from_dict with type='point'."""
        expected_point_data = pd.Series(
            [0.1, 0.1, 0.2, 0.05, 0.4, 0.15],
            index=pd.Index([0, 1, 2, 3, 4, 5], name="Omega"),
            name="probability",
        )

        pd.testing.assert_series_equal(P.point_data, expected_point_data)

    def test_point_data_and_from_dict_with_atom_type(self, Q):
        """Test point_data property and from_dict with type='atom'."""
        assert Q.point_data is None

    def test_point_data_from_pandas(self, F):
        """Test point_data property and from_pandas."""
        data = pd.Series(
            [0.1, 0.1, 0.2, 0.05, 0.4, 0.15],
            index=pd.Index([0, 1, 2, 3, 4, 5], name="Omega"),
            name="probability",
        )
        P = ProbabilityMeasure(sig_alg=F).from_pandas(data=data, type="point")

        pd.testing.assert_series_equal(P.point_data, data)


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
        return ProbabilityMeasure(sig_alg=F).from_dict(probs=point_probs, type="point")

    @pytest.fixture
    def Q(self, F, atom_probs):
        return ProbabilityMeasure(sig_alg=F, name="Q").from_dict(probs=atom_probs)

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
        with pytest.raises(ValueError, match="The event is not measurable"):
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
            probs={0: 0.5, 1: 0.25, 2: 0.25}, type="point"
        )
        P2 = ProbabilityMeasure(sig_alg=G).from_dict(
            probs={0: 0.5, 1: 0.25, 2: 0.25}, type="point"
        )

        assert P1 != P2

    def test_non_equality_different_probabilities(self, F):
        """Test the __eq__ method for inequality with different probabilities."""
        P1 = ProbabilityMeasure(sig_alg=F).from_dict(
            probs={0: 0.6, 1: 0.3, 2: 0.1}, type="point"
        )
        P2 = ProbabilityMeasure(sig_alg=F).from_dict(
            probs={0: 0.5, 1: 0.5, 2: 0.0}, type="point"
        )

        assert P1 != P2

    def test_equality_same_probabilities_and_sigma_algebra(self, F):
        """Test the __eq__ method for equality with same probabilities and sigma algebra."""
        P1 = ProbabilityMeasure(sig_alg=F).from_dict(
            probs={0: 0.5, 1: 0.3, 2: 0.2}, type="point"
        )
        P2 = ProbabilityMeasure(sig_alg=F).from_dict(
            probs={0: 0.5, 1: 0.3, 2: 0.2}, type="point"
        )

        assert P1 == P2

    def test_equality_same_components_different_names(self):
        """Test the __eq__ method for equality with same components but different names."""
        Omega1 = SampleSpace(name="Omega1").from_sequence(size=3)
        Omega2 = SampleSpace(name="Omega2").from_sequence(size=3)
        F1 = SigmaAlgebra(sample_space=Omega1, name="F1").from_dict({0: 0, 1: 0, 2: 1})
        F2 = SigmaAlgebra(sample_space=Omega2, name="F2").from_dict({0: 4, 1: 4, 2: 1})
        P1 = ProbabilityMeasure(sig_alg=F1, name="P1").from_dict(
            probs={0: 0.5, 1: 0.25, 2: 0.25}, type="point"
        )
        P2 = ProbabilityMeasure(sig_alg=F2, name="P2").from_dict(
            probs={0: 0.5, 1: 0.25, 2: 0.25}, type="point"
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
            probs={
                0: 0.0,
                1: 0.3,
                2: 0.05,
                3: 0.1,
                4: 0.15,
                5: 0.25,
                6: 0.15,
            },
            type="point",
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
        return ProbabilityMeasure(sig_alg=F).from_dict(
            probs=probabilities, type="point"
        )

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
        P = ProbabilityMeasure(sig_alg=F).from_dict(probs=probabilities, type="point")
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
        P = ProbabilityMeasure(sig_alg=F).from_dict(probs=probabilities, type="point")

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
            probs={
                0: 0.0,
                1: 0.0,
                2: 0.9,
                3: 0.1,
            },
            type="point",
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
