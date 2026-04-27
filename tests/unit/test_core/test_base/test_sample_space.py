import pandas as pd
import pytest

from sigalg.core import (
    Event,
    EventSpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:

    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        sample_space = SampleSpace(name="my_sample_space", data_name="my_data").from_sequence(
            size=3,
            initial_index=0,
            prefix="omega",
        )
        indices = ["omega_0", "omega_1", "omega_2"]
        name = "my_sample_space"
        data_name = "my_data"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(sample_space.data, expected_index)
        assert sample_space.indices == indices
        assert sample_space.name == name
        assert sample_space.data.name == data_name

    def test_constructor_none_names(self):
        """Test constructor with None for name and data_name."""
        sample_space = SampleSpace(name=None, data_name=None).from_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        indices = ["omega_0", "omega_1", "omega_2"]
        name = None
        data_name = None
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(sample_space.data, expected_index)
        assert sample_space.indices == indices
        assert sample_space.name == name
        assert sample_space.data.name == data_name

    def test_constructor_empty_indices(self):
        """Test constructor with empty list of indices."""
        indices = []
        name = "empty_space"
        data_name = "empty_data"
        sample_space = SampleSpace(name=name, data_name=data_name).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(sample_space.data, expected_index)
        assert sample_space.indices == indices
        assert sample_space.name == name
        assert sample_space.data.name == data_name

    def test_constructor_default_name(self):
        """Test constructor with default name."""
        indices = ["a"]
        data_name = "data_name"
        sample_space = SampleSpace(data_name=data_name).from_list(indices)
        name = "Omega"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(sample_space.data, expected_index)
        assert sample_space.indices == indices
        assert sample_space.name == name
        assert sample_space.data.name == data_name

    def test_constructor_default_data_name(self):
        """Test constructor with default data_name."""
        indices = ["b"]
        name = "name"
        sample_space = SampleSpace(name=name).from_list(indices)
        data_name = "sample"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(sample_space.data, expected_index)
        assert sample_space.indices == indices
        assert sample_space.name == name
        assert sample_space.data.name == data_name

    def test_invalid_indices_not_list_raises(self):
        """Test that non-list indices raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            SampleSpace(name="name", data_name="data").from_list("not_a_list")

    def test_invalid_unhashable_elements_raises(self):
        """Test that unhashable elements raise TypeError."""
        with pytest.raises((TypeError, ValueError)):
            SampleSpace(name="name", data_name="data").from_list([{"a": 1}])

    def test_invalid_duplicate_elements_raises(self):
        """Test that duplicate elements raise ValueError."""
        with pytest.raises((TypeError, ValueError)):
            SampleSpace(name="name", data_name="data").from_list(["a", "b", "a"])


class TestFromPandas:

    def test_from_pandas_with_name(self):
        """Test from_pandas with custom name."""
        pd_index = pd.Index(["s0", "s1", "s2"], name="pandas")
        name = "my_sample_space"
        sample_space = SampleSpace(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(sample_space.data, pd_index)
        assert sample_space.indices == list(pd_index)
        assert sample_space.name == name
        assert sample_space.data.name == pd_index.name

    def test_from_pandas_with_none_name(self):
        """Test from_pandas with None name."""
        pd_index = pd.Index(["s0", "s1", "s2"])
        name = None
        sample_space = SampleSpace(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(sample_space.data, pd_index)
        assert sample_space.indices == list(pd_index)
        assert sample_space.name == name
        assert sample_space.data.name == pd_index.name

    def test_from_pandas_with_default_name(self):
        """Test from_pandas with default name."""
        pd_index = pd.Index(["s0", "s1", "s2"])
        sample_space = SampleSpace().from_pandas(data=pd_index)
        name = "Omega"

        pd.testing.assert_index_equal(sample_space.data, pd_index)
        assert sample_space.indices == list(pd_index)
        assert sample_space.name == name
        assert sample_space.data.name == pd_index.name

    def test_from_pandas_empty_index(self):
        """Test from_pandas with empty pandas Index."""
        pd_index = pd.Index([], name="empty")
        name = "empty_space"
        sample_space = SampleSpace(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(sample_space.data, pd_index)
        assert sample_space.indices == list(pd_index)
        assert sample_space.name == name
        assert sample_space.data.name == pd_index.name


class TestMakeProbabilitySpace:

    def test_make_probability_space_with_all_parameters(self):
        """Test making a ProbabilitySpace with all parameters."""
        sample_space = SampleSpace(name="S", data_name="sample").from_sequence(size=2, initial_index=0, prefix="s")
        probabilities = {"s_0": 0.3, "s_1": 0.7}
        probability_measure = ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(sample_space, name="Q")
        ).from_dict(probabilities)
        sample_id_to_atom_id = {"s_0": "A", "s_1": "B"}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space, name="G").from_dict(
            sample_id_to_atom_id
        )
        prob_space = sample_space.make_probability_space(
            prob_measure=probability_measure, sig_alg=sigma_algebra
        )

        assert isinstance(prob_space, ProbabilitySpace)
        assert prob_space.prob_measure == probability_measure
        assert prob_space.sample_space == sample_space
        assert prob_space.sig_alg == sigma_algebra

    def test_make_probability_space_with_defaults(self):
        """Test making a ProbabilitySpace with default parameters."""
        sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="s")
        prob_space = sample_space.make_probability_space()
        expected_prob_measure = ProbabilityMeasure.uniform(sig_alg=SigmaAlgebra.power_set(sample_space))
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)

        assert isinstance(prob_space, ProbabilitySpace)
        assert prob_space.sample_space == sample_space
        assert prob_space.sig_alg == expected_sigma_algebra
        assert prob_space.prob_measure == expected_prob_measure


class TestMakeEventSpace:

    def test_make_event_space_with_custom_sigma_algebra(self):
        """Test making an EventSpace with a custom SigmaAlgebra."""
        sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=4, initial_index=0, prefix="s")
        sample_id_to_atom_id = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id
        )
        event_space = sample_space.make_event_space(sig_alg=sigma_algebra)

        assert isinstance(event_space, EventSpace)
        assert event_space.sample_space == sample_space
        assert event_space.sig_alg == sigma_algebra

    def test_make_event_space_with_default_sigma_algebra(self):
        """Test making an EventSpace with the default SigmaAlgebra."""
        sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="s")
        event_space = sample_space.make_event_space()
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)

        assert isinstance(event_space, EventSpace)
        assert event_space.sample_space == sample_space
        assert event_space.sig_alg == expected_sigma_algebra


class TestEquality:

    def test_non_equality_different_indices(self):
        """Test inequality when indices are different."""
        given = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = SampleSpace(name="Omega", data_name="sample").from_list(
            ["omega_0", "omega_2"]
        )
        assert given != other

    def test_non_equality_different_order(self):
        """Test inequality when indices are in different order."""
        given = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = SampleSpace(name="Omega", data_name="sample").from_list(
            ["omega_1", "omega_0"]
        )
        assert given != other

    def test_non_equality_different_sizes(self):
        """Test inequality when sample spaces have different sizes."""
        given = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3, initial_index=0, prefix="omega")
        assert given != other

    def test_non_equality_wrong_type_list(self):
        """Test inequality when comparing to a list."""
        given = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = ["omega_0", "omega_1"]
        assert given != other

    def test_non_equality_wrong_type_string(self):
        """Test inequality when comparing to a string."""
        given = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = "not a sample space"
        assert given != other

    def test_non_equality_wrong_type_int(self):
        """Test inequality when comparing to an integer."""
        given = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = 123
        assert given != other

    def test_equality_same_indices(self):
        """Test equality when indices are the same."""
        given = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        assert given == other

    def test_equality_same_indices_different_names(self):
        """Test equality when indices are same but names differ."""
        given = SampleSpace(name="S1", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        other = SampleSpace(name="S2", data_name="sample").from_sequence(size=2, initial_index=0, prefix="omega")
        assert given == other
