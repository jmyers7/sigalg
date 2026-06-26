import pandas as pd
import pytest

from sigalg.core import (
    ProbabilityMeasure,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_no_parameters(self):
        """Test constructor with no parameters."""
        Omega = SampleSpace()

        assert Omega.name == "Omega"
        assert Omega.variable_names is None
        assert Omega.indices is None
        assert Omega.data is None

    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        Omega_0 = SampleSpace(name="Omega_0")

        assert Omega_0.name == "Omega_0"
        assert Omega_0.variable_names is None
        assert Omega_0.indices is None
        assert Omega_0.data is None

    def test_single_dim_default_names(self):
        """Test constructor with single dimension and default names."""
        Omega = SampleSpace(["a", "b", "c"])
        expected_data = pd.Index(["a", "b", "c"], name="sample")

        assert isinstance(Omega.data, pd.Index)
        assert not isinstance(Omega.data, pd.MultiIndex)
        assert Omega.indices == ["a", "b", "c"]
        assert Omega.name == "Omega"
        assert Omega.variable_names == ["sample"]
        assert Omega.dimension == 1
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_multi_dim_default_names(self):
        """Test constructor with multiple dimensions and default names."""
        S = SampleSpace(name="S", indices=[("a", 1), ("b", 2), ("c", 3)])
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["sample_0", "sample_1"]
        )

        assert isinstance(S.data, pd.Index)
        assert isinstance(S.data, pd.MultiIndex)
        assert S.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert S.name == "S"
        assert S.variable_names == ["sample_0", "sample_1"]
        assert S.dimension == 2
        pd.testing.assert_index_equal(S.data, expected_data)

    def test_single_dim_custom_names(self):
        """Test constructor with single dimension and custom names."""
        Omega = SampleSpace(["a", "b", "c"], variable_names=["custom_name"])
        expected_data = pd.Index(["a", "b", "c"], name="custom_name")

        assert isinstance(Omega.data, pd.Index)
        assert not isinstance(Omega.data, pd.MultiIndex)
        assert Omega.indices == ["a", "b", "c"]
        assert Omega.name == "Omega"
        assert Omega.variable_names == ["custom_name"]
        assert Omega.dimension == 1
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_multi_dim_custom_names(self):
        """Test constructor with multiple dimensions and custom names."""
        Omega = SampleSpace(
            [("a", 1), ("b", 2), ("c", 3)],
            variable_names=["custom_name_0", "custom_name_1"],
        )
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["custom_name_0", "custom_name_1"]
        )

        assert isinstance(Omega.data, pd.Index)
        assert isinstance(Omega.data, pd.MultiIndex)
        assert Omega.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert Omega.name == "Omega"
        assert Omega.variable_names == ["custom_name_0", "custom_name_1"]
        assert Omega.dimension == 2
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_multi_dim_custom_prefix_name(self):
        """Test constructor with multiple dimensions and a custom prefix name."""
        Omega = SampleSpace(
            [("a", 1), ("b", 2), ("c", 3)], variable_names=["prefix_0", "prefix_1"]
        )
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["prefix_0", "prefix_1"]
        )

        assert isinstance(Omega.data, pd.Index)
        assert isinstance(Omega.data, pd.MultiIndex)
        assert Omega.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert Omega.name == "Omega"
        assert Omega.variable_names == ["prefix_0", "prefix_1"]
        assert Omega.dimension == 2
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_empty_indices_with_default_data_name(self):
        """Test constructor with empty indices and default data_name."""
        S = SampleSpace(name="S", indices=[])
        expected_data = pd.Index([], name="sample")

        assert isinstance(S.data, pd.Index)
        assert not isinstance(S.data, pd.MultiIndex)
        assert S.indices == []
        assert S.name == "S"
        assert S.variable_names == ["sample"]
        assert S.dimension == 1
        pd.testing.assert_index_equal(S.data, expected_data)

    def test_empty_indices_with_custom_data_name(self):
        """Test constructor with empty indices and custom data_name."""
        Omega = SampleSpace(indices=[], variable_names=["custom_name"])
        expected_data = pd.Index([], name="custom_name")

        assert isinstance(Omega.data, pd.Index)
        assert not isinstance(Omega.data, pd.MultiIndex)
        assert Omega.indices == []
        assert Omega.name == "Omega"
        assert Omega.variable_names == ["custom_name"]
        assert Omega.dimension == 1
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_constructor_with_index_with_default_parameters(self):
        """Test constructor with index and default parameters."""
        data = pd.Index([0, 1, 2])
        Omega = SampleSpace(indices=data)
        expected_data = pd.Index([0, 1, 2], name="sample")

        assert Omega.name == "Omega"
        assert Omega.variable_names == ["sample"]
        assert Omega.indices == [0, 1, 2]
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_constructor_with_index_with_custom_parameters(self):
        """Test constructor with index and custom parameters."""
        data = pd.Index([10, 20, 30], name="outcome")
        Omega = SampleSpace(indices=data, name="Omega_1")

        assert Omega.name == "Omega_1"
        assert Omega.variable_names == ["outcome"]
        assert Omega.indices == [10, 20, 30]
        pd.testing.assert_index_equal(Omega.data, data)

    def test_from_sequence_with_default_parameters(self):
        """Test from_sequence method with default parameters."""
        Omega = SampleSpace().from_sequence(size=3)
        expected_data = pd.Index([0, 1, 2], name="sample")

        assert Omega.name == "Omega"
        assert Omega.variable_names == ["sample"]
        assert Omega.indices == [0, 1, 2]
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_from_sequence_with_custom_parameters(self):
        """Test from_sequence method with custom parameters."""
        Omega = SampleSpace.from_sequence(
            size=3,
            prefix="outcome",
            name="Omega_1",
            initial_index=1,
            variable_name="result",
        )
        expected_data = pd.Index(["outcome_1", "outcome_2", "outcome_3"], name="result")

        assert Omega.name == "Omega_1"
        assert Omega.variable_names == ["result"]
        assert Omega.indices == ["outcome_1", "outcome_2", "outcome_3"]
        pd.testing.assert_index_equal(Omega.data, expected_data)


# --------------------- test conversion methods --------------------- #


class TestMakeProbabilitySpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure.on(
            sig_alg=F,
            mapping={
                0: 0.3,
                1: 0.7,
            },
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
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            },
        )
        event_space = Omega.make_event_space(sig_alg=F)

        assert event_space.sample_space is Omega
        assert event_space.sig_alg is F


# --------------------- test equality --------------------- #


class TestEquality:
    def test_non_equality_different_indices(self):
        """Test inequality when indices are different."""
        Omega1 = SampleSpace.from_sequence(size=2, name="Omega1")
        Omega2 = SampleSpace([0, 2], name="Omega2")

        assert Omega1 != Omega2

    def test_non_equality_different_order(self):
        """Test inequality when indices are in different order."""
        Omega1 = SampleSpace.from_sequence(size=2, name="Omega1")
        Omega2 = SampleSpace([1, 0], name="Omega2")

        assert Omega1 != Omega2

    def test_non_equality_different_sizes(self):
        """Test inequality when sample spaces have different sizes."""
        Omega1 = SampleSpace.from_sequence(size=2, name="Omega1")
        Omega2 = SampleSpace.from_sequence(size=3, name="Omega2")

        assert Omega1 != Omega2

    def test_non_equality_wrong_type_list(self):
        """Test inequality when comparing to a list."""
        Omega = SampleSpace.from_sequence(size=2)
        other = [0, 1]

        assert Omega != other

    def test_non_equality_wrong_type_string(self):
        """Test inequality when comparing to a string."""
        Omega = SampleSpace.from_sequence(size=2)
        other = "not a sample space"

        assert Omega != other

    def test_non_equality_wrong_type_int(self):
        """Test inequality when comparing to an integer."""
        Omega = SampleSpace.from_sequence(size=2)
        other = 123

        assert Omega != other

    def test_equality_same_indices(self):
        """Test equality when indices are the same."""
        Omega1 = SampleSpace.from_sequence(size=2, name="Omega1")
        Omega2 = SampleSpace.from_sequence(size=2, name="Omega2")

        assert Omega1 == Omega2

    def test_equality_same_indices_different_names(self):
        """Test equality when indices are same but names differ."""
        Omega1 = SampleSpace.from_sequence(size=2, name="Omega1")
        Omega2 = SampleSpace.from_sequence(size=2, name="Omega2")

        assert Omega1 == Omega2
