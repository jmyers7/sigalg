import pandas as pd
import pytest
from sigalg.core import (
    Measure,
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
        assert Omega.data is None

    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        Omega_0 = SampleSpace(name="Omega_0")

        assert Omega_0.name == "Omega_0"
        assert Omega_0.variable_names is None
        assert Omega_0.data is None

    def test_single_dim_default_names(self):
        """Test constructor with single dimension and default names."""
        Omega = SampleSpace(["a", "b", "c"])
        expected_data = pd.Index(["a", "b", "c"], name="sample")

        assert isinstance(Omega.data, pd.Index)
        assert not isinstance(Omega.data, pd.MultiIndex)
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
        pd.testing.assert_index_equal(Omega.data, expected_data)

    def test_constructor_with_index_with_custom_parameters(self):
        """Test constructor with index and custom parameters."""
        data = pd.Index([10, 20, 30], name="outcome")
        Omega = SampleSpace(indices=data, name="Omega_1")

        assert Omega.name == "Omega_1"
        assert Omega.variable_names == ["outcome"]
        pd.testing.assert_index_equal(Omega.data, data)

    def test_from_sequence_with_default_parameters(self):
        """Test from_sequence method with default parameters."""
        Omega = SampleSpace().from_sequence(size=3)
        expected_data = pd.Index([0, 1, 2], name="sample")

        assert Omega.name == "Omega"
        assert Omega.variable_names == ["sample"]
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
        pd.testing.assert_index_equal(Omega.data, expected_data)


# --------------------- test conversion methods --------------------- #


class TestMakeMeasureSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.3,
                1: 0.7,
            },
        )

    def test_make_measure_space_with_all_parameters(self, Omega, F, P):
        """Test making a MeasureSpace with all parameters."""
        prob_space = Omega.make_measure_space(sig_alg=F, measure=P)

        assert prob_space.domain is Omega
        assert prob_space.measure is P
        assert prob_space.sig_alg is F

    def test_make_measure_space_with_custom_sig_alg(self, Omega, F):
        """Test making a MeasureSpace custom sigma-algebra."""
        prob_space = Omega.make_measure_space(sig_alg=F)

        assert prob_space.domain is Omega
        assert prob_space.sig_alg is F
        assert prob_space.measure == Measure.counting(F)

    def test_make_measure_space_with_custom_prob_measure(self, Omega, P):
        """Test making a MeasureSpace with a custom probability measure."""
        prob_space = Omega.make_measure_space(measure=P)

        assert prob_space.domain is Omega
        assert prob_space.sig_alg == P.sig_alg
        assert prob_space.measure is P

    def test_make_measure_space_with_default_parameters(self, Omega):
        """Test making a MeasureSpace with default parameters."""
        prob_space = Omega.make_measure_space()

        assert prob_space.domain is Omega
        assert prob_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert prob_space.measure == Measure.counting(
            domain=SigmaAlgebra.power_set(Omega)
        )


class TestMakeMeasurableSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    def test_make_measurable_space_with_default_parameters(self, Omega):
        """Test making an MeasurableSpace from a SampleSpace."""
        measurable_space = Omega.make_measurable_space()

        assert measurable_space.domain is Omega
        assert measurable_space.sig_alg == SigmaAlgebra.power_set(Omega)

    def test_make_measurable_space_with_custom_sig_alg(self, Omega):
        """Test making an MeasurableSpace with a custom sigma-algebra."""
        F = SigmaAlgebra(
            domain=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            },
        )
        measurable_space = Omega.make_measurable_space(sig_alg=F)

        assert measurable_space.domain is Omega
        assert measurable_space.sig_alg is F
