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

    def test_construction_from_all_parameters(self):
        """Test constructor with all parameters provided."""
        indices = ["omega0", "omega1", "omega2"]
        name = "my_sample_space"
        data_name = "my_data"
        sample_space = SampleSpace(indices=indices, name=name, data_name=data_name)
        expected_index = pd.Index(data=["omega0", "omega1", "omega2"], name="my_data")
        assert sample_space.indices == indices
        assert sample_space.name == name
        assert sample_space.data.name == data_name
        pd.testing.assert_index_equal(sample_space.data, expected_index)

    def test_construction_minimal_parameters(self):
        """Test constructor with minimal parameters provided."""
        indices = ["omega0", "omega1", "omega2"]
        sample_space = SampleSpace(indices=indices)
        expected_index = pd.Index(data=indices, name="sample")
        assert sample_space.indices == indices
        assert sample_space.name == "Omega"
        assert sample_space.data.name == "sample"
        pd.testing.assert_index_equal(sample_space.data, expected_index)

    def test_construction_from_pandas_with_data_name(self):
        """Test constructor from a pd.Index."""
        data = pd.Index(["s0", "s1", "s2"], name="pandas")
        name = "my_sample_space"
        sample_space = SampleSpace.from_pandas(data=data, name=name)
        assert sample_space.indices == ["s0", "s1", "s2"]
        assert sample_space.name == name
        assert sample_space.data.name == "pandas"
        pd.testing.assert_index_equal(sample_space.data, data)

    def test_construction_from_pandas_without_data_name(self):
        """Test constructor from a pd.Index without providing name."""
        data = pd.Index(["s0", "s1", "s2"])
        sample_space = SampleSpace.from_pandas(data=data)
        assert sample_space.indices == ["s0", "s1", "s2"]
        assert sample_space.name == "Omega"
        assert sample_space.data.name is None
        pd.testing.assert_index_equal(sample_space.data, data)


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

    def test_get_event_default_name(self, sample_space):
        """Test get_event with default name."""
        event = sample_space.get_event(["omega0", "omega1"])
        assert isinstance(event, Event)
        expected_index = pd.Index(data=["omega0", "omega1"], name="sample")
        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == "A"

    def test_get_event_user_provided_name(self, sample_space):
        """Test get_event with user-provided name."""
        event = sample_space.get_event(["omega0", "omega1"], name="B")
        assert isinstance(event, Event)
        expected_index = pd.Index(data=["omega0", "omega1"], name="sample")
        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == "B"

    def test_get_event_with_empty_list(self, sample_space):
        """Test get_event with an empty list of indices."""
        event = sample_space.get_event([])
        assert isinstance(event, Event)
        assert len(event) == 0


class TestGetItem:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_getitem_with_list_of_positions(self, sample_space):
        """Test getitem with a list of positions."""
        event = sample_space[[0, 2], "D"]
        assert isinstance(event, Event)
        expected_index = pd.Index(data=["omega0", "omega2"], name="sample")
        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == "D"

    def test_getitem_with_single_pos(self, sample_space):
        """Test getitem with a single position."""
        assert sample_space[0, "E"] == "omega0"

    def test_getitem_with_slice(self, sample_space):
        """Test getitem with a slice."""
        event = sample_space[1:3, "D"]
        assert isinstance(event, Event)
        expected_index = pd.Index(data=["omega1", "omega2"], name="sample")
        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == "D"


class TestEquality:

    def test_equality_same_indices(self):
        """Test equality of two SampleSpace instances with the same indices."""
        sample_space1 = SampleSpace(["omega0", "omega1"])
        sample_space2 = SampleSpace(["omega0", "omega1"])
        assert sample_space1 == sample_space2

    def test_equality_different_indices(self):
        """Test inequality of two SampleSpace instances with different indices."""
        sample_space1 = SampleSpace(["omega0", "omega1"])
        sample_space2 = SampleSpace(["omega0", "omega2"])
        assert sample_space1 != sample_space2

    def test_equality_different_order(self):
        """Test inequality of two SampleSpace instances with same indices in different order."""
        sample_space1 = SampleSpace(["omega0", "omega1"])
        sample_space2 = SampleSpace(["omega1", "omega0"])
        assert sample_space1 != sample_space2

    def test_equality_with_non_sample_space(self):
        """Test inequality with non-SampleSpace objects."""
        sample_space = SampleSpace(["omega0", "omega1"])
        assert sample_space != ["omega0", "omega1"]
        assert sample_space != "not a sample space"
        assert sample_space != 123

    def test_equality_different_sizes(self):
        """Test inequality of two SampleSpace instances with different sizes."""
        sample_space1 = SampleSpace(["omega0", "omega1"])
        sample_space2 = SampleSpace(["omega0", "omega1", "omega2"])
        assert sample_space1 != sample_space2


class TestValidation:
    pass
