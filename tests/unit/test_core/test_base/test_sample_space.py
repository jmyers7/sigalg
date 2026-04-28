import pandas as pd
import pytest

from sigalg.core import (
    EventSpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:
    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        indices = [0, 1, 2]
        name = "Omega"
        data_name = "sample"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(Omega.data, expected_index)
        assert Omega.indices == indices
        assert Omega.name == name
        assert Omega.data.name == data_name

    def test_constructor_none_names(self):
        """Test constructor with None for name and data_name."""
        Omega = SampleSpace(name=None, data_name=None).from_sequence(size=3)
        indices = [0, 1, 2]
        name = None
        data_name = None
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(Omega.data, expected_index)
        assert Omega.indices == indices
        assert Omega.name == name
        assert Omega.data.name == data_name

    def test_constructor_empty_indices(self):
        """Test constructor with empty list of indices."""
        indices = []
        name = "Omega"
        data_name = "sample"
        Omega = SampleSpace(name=name, data_name=data_name).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(Omega.data, expected_index)
        assert Omega.indices == indices
        assert Omega.name == name
        assert Omega.data.name == data_name

    def test_constructor_default_name(self):
        """Test constructor with default name."""
        indices = ["a"]
        data_name = "sample"
        Omega = SampleSpace(data_name=data_name).from_list(indices)
        name = "Omega"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(Omega.data, expected_index)
        assert Omega.indices == indices
        assert Omega.name == name
        assert Omega.data.name == data_name

    def test_constructor_default_data_name(self):
        """Test constructor with default data_name."""
        indices = ["b"]
        name = "Omega"
        Omega = SampleSpace(name=name).from_list(indices)
        data_name = "sample"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(Omega.data, expected_index)
        assert Omega.indices == indices
        assert Omega.name == name
        assert Omega.data.name == data_name

    def test_invalid_indices_not_list_raises(self):
        """Test that non-list indices raise TypeError."""
        with pytest.raises(TypeError):
            SampleSpace(name="Omega", data_name="sample").from_list("not_a_list")

    def test_invalid_unhashable_elements_raises(self):
        """Test that unhashable elements raise TypeError."""
        with pytest.raises(TypeError):
            SampleSpace(name="Omega", data_name="sample").from_list([{"a": 1}])

    def test_invalid_duplicate_elements_raises(self):
        """Test that duplicate elements raise ValueError."""
        with pytest.raises(ValueError):
            SampleSpace(name="Omega", data_name="sample").from_list(["a", "b", "a"])


class TestFromPandas:
    def test_from_pandas_with_name(self):
        """Test from_pandas with custom name."""
        pd_index = pd.Index(["s0", "s1", "s2"], name="pandas")
        name = "Omega"
        Omega = SampleSpace(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(Omega.data, pd_index)
        assert Omega.indices == list(pd_index)
        assert Omega.name == name
        assert Omega.data.name == pd_index.name

    def test_from_pandas_with_none_name(self):
        """Test from_pandas with None name."""
        pd_index = pd.Index(["s0", "s1", "s2"])
        name = None
        Omega = SampleSpace(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(Omega.data, pd_index)
        assert Omega.indices == list(pd_index)
        assert Omega.name == name
        assert Omega.data.name == pd_index.name

    def test_from_pandas_with_default_name(self):
        """Test from_pandas with default name."""
        pd_index = pd.Index(["s0", "s1", "s2"])
        Omega = SampleSpace().from_pandas(data=pd_index)
        name = "Omega"

        pd.testing.assert_index_equal(Omega.data, pd_index)
        assert Omega.indices == list(pd_index)
        assert Omega.name == name
        assert Omega.data.name == pd_index.name

    def test_from_pandas_empty_index(self):
        """Test from_pandas with empty pandas Index."""
        pd_index = pd.Index([], name="empty")
        name = "Omega"
        Omega = SampleSpace(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(Omega.data, pd_index)
        assert Omega.indices == list(pd_index)
        assert Omega.name == name
        assert Omega.data.name == pd_index.name


class TestMakeProbabilitySpace:
    def test_make_probability_space_with_all_parameters(self):
        """Test making a ProbabilitySpace with all parameters."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        probabilities = {0: 0.3, 1: 0.7}
        P = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(Omega, name="F")
        ).from_dict(probabilities)
        sample_id_to_atom_id = {0: "A", 1: "B"}
        F = SigmaAlgebra(sample_space=Omega, name="F").from_dict(sample_id_to_atom_id)
        prob_space = Omega.make_probability_space(prob_measure=P, sig_alg=F)

        assert isinstance(prob_space, ProbabilitySpace)
        assert prob_space.prob_measure == P
        assert prob_space.sample_space == Omega
        assert prob_space.sig_alg == F

    def test_make_probability_space_with_defaults(self):
        """Test making a ProbabilitySpace with default parameters."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        prob_space = Omega.make_probability_space()
        expected_prob_measure = ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(Omega)
        )
        expected_sigma_algebra = SigmaAlgebra.power_set(Omega)

        assert isinstance(prob_space, ProbabilitySpace)
        assert prob_space.sample_space == Omega
        assert prob_space.sig_alg == expected_sigma_algebra
        assert prob_space.prob_measure == expected_prob_measure


class TestMakeEventSpace:
    def test_make_event_space_with_custom_sigma_algebra(self):
        """Test making an EventSpace with a custom SigmaAlgebra."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)
        sample_id_to_atom_id = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id)
        event_space = Omega.make_event_space(sig_alg=F)

        assert isinstance(event_space, EventSpace)
        assert event_space.sample_space == Omega
        assert event_space.sig_alg == F

    def test_make_event_space_with_default_sigma_algebra(self):
        """Test making an EventSpace with the default SigmaAlgebra."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        event_space = Omega.make_event_space()
        expected_sigma_algebra = SigmaAlgebra.power_set(Omega)

        assert isinstance(event_space, EventSpace)
        assert event_space.sample_space == Omega
        assert event_space.sig_alg == expected_sigma_algebra


class TestEquality:
    def test_non_equality_different_indices(self):
        """Test inequality when indices are different."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_list([0, 2])

        assert Omega1 != Omega2

    def test_non_equality_different_order(self):
        """Test inequality when indices are in different order."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_list([1, 0])

        assert Omega1 != Omega2

    def test_non_equality_different_sizes(self):
        """Test inequality when sample spaces have different sizes."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)

        assert Omega1 != Omega2

    def test_non_equality_wrong_type_list(self):
        """Test inequality when comparing to a list."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        other = [0, 1]

        assert Omega != other

    def test_non_equality_wrong_type_string(self):
        """Test inequality when comparing to a string."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        other = "not a sample space"

        assert Omega != other

    def test_non_equality_wrong_type_int(self):
        """Test inequality when comparing to an integer."""
        Omega = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        other = 123

        assert Omega != other

    def test_equality_same_indices(self):
        """Test equality when indices are the same."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)

        assert Omega1 == Omega2

    def test_equality_same_indices_different_names(self):
        """Test equality when indices are same but names differ."""
        Omega1 = SampleSpace(name="Omega1", data_name="sample").from_sequence(size=2)
        Omega2 = SampleSpace(name="Omega2", data_name="sample").from_sequence(size=2)

        assert Omega1 == Omega2
