import pandas as pd
import pytest

from sigalg.core import (
    ProbabilityMeasure,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test constructor with no parameters."""
        Omega = SampleSpace()

        assert Omega.name == "Omega"
        assert Omega.data_name is None
        assert Omega.indices is None
        assert Omega.data is None

    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        Omega_0 = SampleSpace(name="Omega_0")

        assert Omega_0.name == "Omega_0"
        assert Omega_0.data_name is None
        assert Omega_0.indices is None
        assert Omega_0.data is None


class TestFromList:
    def test_from_list_with_default_parameters(self):
        """Test from_list method with default parameters."""
        Omega = SampleSpace().from_list([0, 1, 2])
        expected_data = pd.Index([0, 1, 2], name="sample")

        assert Omega.name == "Omega"
        assert Omega.data_name == "sample"
        assert Omega.indices == [0, 1, 2]
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_from_list_with_custom_parameters(self):
        """Test from_list method with custom parameters."""
        Omega = SampleSpace(name="Omega_1").from_list([10, 20, 30], data_name="outcome")
        expected_data = pd.Index([10, 20, 30], name="outcome")

        assert Omega.name == "Omega_1"
        assert Omega.data_name == "outcome"
        assert Omega.indices == [10, 20, 30]
        pd.testing.assert_index_equal(Omega.data, expected_data)


class TestFromPandas:
    def test_from_pandas_with_default_parameters(self):
        """Test from_pandas method with default parameters."""
        data = pd.Index([0, 1, 2])
        Omega = SampleSpace().from_pandas(data)

        assert Omega.name == "Omega"
        assert Omega.data_name is None
        assert Omega.indices == [0, 1, 2]
        pd.testing.assert_index_equal(Omega.data, data)

    def test_from_pandas_with_custom_parameters(self):
        """Test from_pandas method with custom parameters."""
        data = pd.Index([10, 20, 30], name="outcome")
        Omega = SampleSpace(name="Omega_1").from_pandas(data)

        assert Omega.name == "Omega_1"
        assert Omega.data_name == "outcome"
        assert Omega.indices == [10, 20, 30]
        pd.testing.assert_index_equal(Omega.data, data)


class TestFromSequence:
    def test_from_sequence_with_default_parameters(self):
        """Test from_sequence method with default parameters."""
        Omega = SampleSpace().from_sequence(size=3)
        expected_data = pd.Index([0, 1, 2], name="sample")

        assert Omega.name == "Omega"
        assert Omega.data_name == "sample"
        assert Omega.indices == [0, 1, 2]
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_from_sequence_with_custom_parameters(self):
        """Test from_sequence method with custom parameters."""
        Omega = SampleSpace(name="Omega_1").from_sequence(
            size=3, prefix="outcome", initial_index=1, data_name="result"
        )
        expected_data = pd.Index(["outcome_1", "outcome_2", "outcome_3"], name="result")

        assert Omega.name == "Omega_1"
        assert Omega.data_name == "result"
        assert Omega.indices == ["outcome_1", "outcome_2", "outcome_3"]
        pd.testing.assert_index_equal(Omega.data, expected_data)


# --------------------- test conversion methods --------------------- #


class TestMakeProbabilitySpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.3,
                1: 0.7,
            }
        )

    def test_make_probability_space_with_all_parameters(self, Omega, F, P):
        """Test making a ProbabilitySpace with all parameters."""
        prob_space = Omega.make_probability_space(sig_alg=F, prob_measure=P)

        assert prob_space.sample_space is Omega
        assert prob_space.prob_measure is P
        assert prob_space.sig_alg is F

    def test_make_probability_space_with_custom_sig_alg(self, Omega, F):
        """Test making a ProbabilitySpace custom sigma-algebra."""
        prob_space = Omega.make_probability_space(sig_alg=F)

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg is F
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(sig_alg=F)

    def test_make_probability_space_with_custom_prob_measure(self, Omega, P):
        """Test making a ProbabilitySpace with a custom probability measure."""
        prob_space = Omega.make_probability_space(prob_measure=P)

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg == P.sig_alg
        assert prob_space.prob_measure is P

    def test_make_probability_space_with_default_parameters(self, Omega):
        """Test making a ProbabilitySpace with default parameters."""
        prob_space = Omega.make_probability_space()

        assert prob_space.sample_space is Omega
        assert prob_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert prob_space.prob_measure == ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(Omega)
        )


class TestMakeEventSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    def test_make_event_space_with_default_parameters(self, Omega):
        """Test making an EventSpace from a SampleSpace."""
        event_space = Omega.make_event_space()

        assert event_space.sample_space is Omega
        assert event_space.sig_alg == SigmaAlgebra.power_set(Omega)

    def test_make_event_space_with_custom_sig_alg(self, Omega):
        """Test making an EventSpace with a custom sigma-algebra."""
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            }
        )
        event_space = Omega.make_event_space(sig_alg=F)

        assert event_space.sample_space is Omega
        assert event_space.sig_alg is F


# --------------------- test equality --------------------- #


class TestEquality:
    def test_non_equality_different_indices(self):
        """Test inequality when indices are different."""
        Omega1 = SampleSpace(name="Omega").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega").from_list([0, 2])

        assert Omega1 != Omega2

    def test_non_equality_different_order(self):
        """Test inequality when indices are in different order."""
        Omega1 = SampleSpace(name="Omega").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega").from_list([1, 0])

        assert Omega1 != Omega2

    def test_non_equality_different_sizes(self):
        """Test inequality when sample spaces have different sizes."""
        Omega1 = SampleSpace(name="Omega").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega").from_sequence(size=3)

        assert Omega1 != Omega2

    def test_non_equality_wrong_type_list(self):
        """Test inequality when comparing to a list."""
        Omega = SampleSpace(name="Omega").from_sequence(size=2)
        other = [0, 1]

        assert Omega != other

    def test_non_equality_wrong_type_string(self):
        """Test inequality when comparing to a string."""
        Omega = SampleSpace(name="Omega").from_sequence(size=2)
        other = "not a sample space"

        assert Omega != other

    def test_non_equality_wrong_type_int(self):
        """Test inequality when comparing to an integer."""
        Omega = SampleSpace(name="Omega").from_sequence(size=2)
        other = 123

        assert Omega != other

    def test_equality_same_indices(self):
        """Test equality when indices are the same."""
        Omega1 = SampleSpace(name="Omega").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega").from_sequence(size=2)

        assert Omega1 == Omega2

    def test_equality_same_indices_different_names(self):
        """Test equality when indices are same but names differ."""
        Omega1 = SampleSpace(name="Omega1").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega2").from_sequence(size=2)

        assert Omega1 == Omega2
