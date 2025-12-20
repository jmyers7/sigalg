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

    @pytest.mark.parametrize(
        "indices,name,data_name",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                "my_sample_space",
                "my_data",
                id="all_parameters",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"], None, None, id="minimal_parameters"
            ),
            pytest.param([], "empty_space", "empty_data", id="empty_indices"),
            pytest.param(["s0"], "single_sample", None, id="single_index"),
        ],
    )
    def test_constructor(self, indices, name, data_name):
        """Test constructor with various combinations of parameters."""
        sample_space = SampleSpace(indices=indices, name=name, data_name=data_name)
        expected_name = name if name is not None else "Omega"
        expected_data_name = data_name if data_name is not None else "sample"
        expected_index = pd.Index(data=indices, name=expected_data_name)

        assert sample_space.indices == indices
        assert sample_space.name == expected_name
        assert sample_space.data.name == expected_data_name
        pd.testing.assert_index_equal(sample_space.data, expected_index)

    @pytest.mark.parametrize(
        "indices,name,data_name",
        [
            pytest.param("not_a_list", "name", "data", id="indices_not_list"),
            pytest.param([{"a": 1}], "name", "data", id="unhashable_elements"),
            pytest.param(["a", "b", "a"], "name", "data", id="duplicate_elements"),
        ],
    )
    def test_invalid_inputs_raise(self, indices, name, data_name):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises((TypeError, ValueError)):
            SampleSpace(indices=indices, name=name, data_name=data_name)


class TestFromPandas:

    @pytest.mark.parametrize(
        "pandas_data,name",
        [
            pytest.param(
                pd.Index(["s0", "s1", "s2"], name="pandas"),
                "my_sample_space",
                id="with_name",
            ),
            pytest.param(
                pd.Index(["s0", "s1", "s2"]),
                None,
                id="without_name",
            ),
            pytest.param(
                pd.Index([], name="empty"),
                "empty_space",
                id="empty_index",
            ),
        ],
    )
    def test_from_pandas(self, pandas_data, name):
        """Test the from_pandas class method."""
        sample_space = SampleSpace.from_pandas(data=pandas_data, name=name)
        expected_name = name if name is not None else "Omega"

        assert sample_space.indices == list(pandas_data)
        assert sample_space.name == expected_name
        assert sample_space.data.name == pandas_data.name
        pd.testing.assert_index_equal(sample_space.data, pandas_data)


class TestMakeProbabilitySpace:

    def test_make_probability_space_with_all_parameters(self):
        """Test making a ProbabilitySpace with all parameters."""
        sample_space = SampleSpace(indices=["s0", "s1"], name="S")
        probabilities = {"s0": 0.3, "s1": 0.7}
        probability_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities, name="Q"
        )
        sample_id_to_atom_id = {"s0": "A", "s1": "B"}
        sigma_algebra = SigmaAlgebra(
            sample_space=sample_space,
            sample_id_to_atom_id=sample_id_to_atom_id,
            name="G",
        )
        prob_space = sample_space.make_probability_space(
            probability_measure=probability_measure, sigma_algebra=sigma_algebra
        )

        assert isinstance(prob_space, ProbabilitySpace)
        assert prob_space.probability_measure == probability_measure
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == sigma_algebra

    def test_make_probability_space_with_defaults(self):
        """Test making a ProbabilitySpace with default parameters."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        prob_space = sample_space.make_probability_space()
        expected_prob_measure = ProbabilityMeasure.uniform(sample_space=sample_space)
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)

        assert isinstance(prob_space, ProbabilitySpace)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == expected_sigma_algebra
        assert prob_space.probability_measure == expected_prob_measure


class TestMakeEventSpace:

    def test_make_event_space_with_custom_sigma_algebra(self):
        """Test making an EventSpace with a custom SigmaAlgebra."""
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma_algebra = SigmaAlgebra(
            sample_space=sample_space,
            sample_id_to_atom_id=sample_id_to_atom_id,
            name="F",
        )
        event_space = sample_space.make_event_space(sigma_algebra=sigma_algebra)

        assert isinstance(event_space, EventSpace)
        assert event_space.sample_space == sample_space
        assert event_space.sigma_algebra == sigma_algebra

    def test_make_event_space_with_default_sigma_algebra(self):
        """Test making an EventSpace with the default SigmaAlgebra."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        event_space = sample_space.make_event_space()
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)

        assert isinstance(event_space, EventSpace)
        assert event_space.sample_space == sample_space
        assert event_space.sigma_algebra == expected_sigma_algebra


class TestGetEvent:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.mark.parametrize(
        "indices,name,expected_name",
        [
            pytest.param(["omega0", "omega1"], None, "A", id="default_name"),
            pytest.param(["omega0", "omega1"], "B", "B", id="user_provided_name"),
            pytest.param([], "empty", "empty", id="empty_list"),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                "full",
                "full",
                id="all_indices",
            ),
        ],
    )
    def test_get_event(self, sample_space, indices, name, expected_name):
        """Test get_event method with various parameters."""
        event = sample_space.get_event(indices, name=name)
        expected_index = pd.Index(data=indices, name="sample")

        assert isinstance(event, Event)
        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == expected_name


class TestGetItem:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.mark.parametrize(
        "pos,name,expected_result,expected_indices",
        [
            pytest.param(
                [0, 2], "D", Event, ["omega0", "omega2"], id="list_of_positions"
            ),
            pytest.param(slice(1, 3), "E", Event, ["omega1", "omega2"], id="slice"),
            pytest.param(0, "F", str, None, id="single_position"),
            pytest.param(
                [1, 3], "G", Event, ["omega1", "omega3"], id="non_contiguous_list"
            ),
        ],
    )
    def test_getitem(self, sample_space, pos, name, expected_result, expected_indices):
        """Test __getitem__ method with various position types."""
        result = sample_space[pos, name]

        if expected_result is str:
            assert result == sample_space.indices[pos]
        else:
            assert isinstance(result, Event)
            expected_index = pd.Index(data=expected_indices, name="sample")
            pd.testing.assert_index_equal(result.data, expected_index)
            assert result.name == name


class TestEquality:

    @pytest.mark.parametrize(
        "given,other",
        [
            pytest.param(
                SampleSpace(["omega0", "omega1"]),
                SampleSpace(["omega0", "omega2"]),
                id="different_indices",
            ),
            pytest.param(
                SampleSpace(["omega0", "omega1"]),
                SampleSpace(["omega1", "omega0"]),
                id="different_order",
            ),
            pytest.param(
                SampleSpace(["omega0", "omega1"]),
                SampleSpace(["omega0", "omega1", "omega2"]),
                id="different_sizes",
            ),
            pytest.param(
                SampleSpace(["omega0", "omega1"]),
                ["omega0", "omega1"],
                id="wrong_type_list",
            ),
            pytest.param(
                SampleSpace(["omega0", "omega1"]),
                "not a sample space",
                id="wrong_type_string",
            ),
            pytest.param(
                SampleSpace(["omega0", "omega1"]),
                123,
                id="wrong_type_int",
            ),
        ],
    )
    def test_non_equality(self, given, other):
        """Test the __eq__ method for inequality."""
        assert given != other

    @pytest.mark.parametrize(
        "given,other",
        [
            pytest.param(
                SampleSpace(["omega0", "omega1"]),
                SampleSpace(["omega0", "omega1"]),
                id="same_indices",
            ),
            pytest.param(
                SampleSpace(["omega0", "omega1"], name="S1"),
                SampleSpace(["omega0", "omega1"], name="S2"),
                id="same_indices_different_names",
            ),
        ],
    )
    def test_equality(self, given, other):
        """Test the __eq__ method for equality."""
        assert given == other
