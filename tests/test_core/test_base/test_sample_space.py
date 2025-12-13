import pandas as pd
import pytest

import sigalg as sa

pytestmark = pytest.mark.unit


class TestConstructor:
    def test_construction_with_valid_list(self):
        sample_space = sa.SampleSpace(indices=["omega0", "omega1", "omega2"])
        expected_index = pd.Index(data=["omega0", "omega1", "omega2"], name="sample")
        pd.testing.assert_index_equal(sample_space.values, expected_index)

    def test_construction_with_integers(self):
        sample_space = sa.SampleSpace(indices=[1, 2, 3])
        expected_index = pd.Index(data=[1, 2, 3], name="sample")
        pd.testing.assert_index_equal(sample_space.values, expected_index)

    def test_construction_with_user_provided_name(self):
        sample_space = sa.SampleSpace(indices=[1, 2, 3], name="S")
        expected_index = pd.Index(data=[1, 2, 3], name="sample")
        pd.testing.assert_index_equal(sample_space.values, expected_index)

    def test_construction_with_values_parameter(self):
        values = pd.Index(data=["a", "b", "c"], name="test")
        sample_space = sa.SampleSpace(values=values, name="S")
        expected_index = pd.Index(data=["a", "b", "c"], name="test")
        pd.testing.assert_index_equal(sample_space.values, expected_index)
        assert sample_space.name == "S"

    def test_construction_with_values_parameter_default_name(self):
        values = pd.Index(data=[1, 2, 3], name="numbers")
        sample_space = sa.SampleSpace(values=values)
        expected_index = pd.Index(data=[1, 2, 3], name="numbers")
        pd.testing.assert_index_equal(sample_space.values, expected_index)
        assert sample_space.name == "Omega"

    def test_construction_with_values_parameter_preserves_values_name(self):
        values = pd.Index(data=["x", "y", "z"], name="custom")
        sample_space = sa.SampleSpace(values=values, name="MySpace")
        assert sample_space.values.name == "custom"

    def test_construction_with_values_parameter_no_name(self):
        values = pd.Index(data=["p", "q", "r"])
        sample_space = sa.SampleSpace(values=values)
        expected_index = pd.Index(data=["p", "q", "r"])
        pd.testing.assert_index_equal(sample_space.values, expected_index)


class TestValidation:
    def test_construction_with_duplicates_raises_error(self):
        with pytest.raises(ValueError):
            sa.SampleSpace(indices=["omega0", "omega1", "omega0"])

    def test_construction_with_non_list_raises_error(self):
        with pytest.raises(TypeError):
            sa.SampleSpace(indices={"omega0", "omega1"})

    def test_construction_with_both_indices_and_values_raises_error(self):
        values = pd.Index(data=["a", "b", "c"])
        with pytest.raises(ValueError, match="Cannot specify both"):
            sa.SampleSpace(indices=["a", "b", "c"], values=values)

    def test_construction_with_neither_indices_nor_values_raises_error(self):
        with pytest.raises(ValueError, match="Must specify either"):
            sa.SampleSpace()

    def test_construction_with_non_pandas_index_values_raises_error(self):
        with pytest.raises(TypeError, match="values must be a pandas Index"):
            sa.SampleSpace(values=["a", "b", "c"])

    def test_construction_with_duplicate_values_raises_error(self):
        values = pd.Index(data=["a", "b", "a"])
        with pytest.raises(ValueError, match="must be unique"):
            sa.SampleSpace(values=values)


class TestDataAccessMethods:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_get_event_default_name(self, sample_space):
        event = sample_space.get_event(["omega0", "omega1"])
        expected_index = pd.Index(data=["omega0", "omega1"], name="sample")
        pd.testing.assert_index_equal(event.values, expected_index)
        assert event.name == "A"

    def test_get_event_user_provided_name(self, sample_space):
        event = sample_space.get_event(["omega0", "omega1"], name="B")
        expected_index = pd.Index(data=["omega0", "omega1"], name="sample")
        pd.testing.assert_index_equal(event.values, expected_index)
        assert event.name == "B"

    def test_get_event_with_empty_list(self, sample_space):
        event = sample_space.get_event([])
        assert len(event) == 0

    def test_get_event_with_non_list_raises_error(self, sample_space):
        with pytest.raises(TypeError, match="must be a list"):
            sample_space.get_event("omega0")

    def test_get_event_with_invalid_index_raises_error(self, sample_space):
        with pytest.raises(ValueError, match="not found in sample space"):
            sample_space.get_event(["omega0", "invalid"])

    def test_getitem_with_list_of_positions(self, sample_space):
        event = sample_space[[0, 2], "D"]
        expected_index = pd.Index(data=["omega0", "omega2"], name="sample")
        pd.testing.assert_index_equal(event.values, expected_index)
        assert event.name == "D"

    def test_getitem_with_single_pos(self, sample_space):
        event = sample_space[0, "E"]
        expected_index = pd.Index(data=["omega0"], name="sample")
        pd.testing.assert_index_equal(event.values, expected_index)
        assert event.name == "E"

    def test_getitem_with_slice(self, sample_space):
        event = sample_space[1:3, "D"]
        expected_index = pd.Index(data=["omega1", "omega2"], name="sample")
        pd.testing.assert_index_equal(event.values, expected_index)
        assert event.name == "D"

    def test_getitem_with_invalid_type_raises_error(self, sample_space):
        with pytest.raises(IndexError):
            sample_space["invalid"]

    def test_getitem_with_out_of_bounds_index_raises_error(self, sample_space):
        with pytest.raises(IndexError):
            sample_space[[0, 5]]


class TestSetters:

    def test_set_name(self):
        sample_space = sa.SampleSpace(["omega0", "omega1"])
        sample_space.name = "NewName"
        assert sample_space.name == "NewName"


class TestLen:
    def test_len_returns_correct_size(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert len(sample_space) == 3

    def test_len_single_element(self):
        sample_space = sa.SampleSpace(["omega0"])
        assert len(sample_space) == 1

    def test_len_large_space(self):
        indices = [f"omega{i}" for i in range(100)]
        sample_space = sa.SampleSpace(indices)
        assert len(sample_space) == 100


class TestIteration:
    def test_iteration_returns_all_indices(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        indices = list(sample_space)
        assert indices == ["omega0", "omega1", "omega2"]

    def test_iteration_preserves_order(self):
        sample_space = sa.SampleSpace(["z", "a", "m"])
        indices = list(sample_space)
        assert indices == ["z", "a", "m"]

    def test_can_iterate_multiple_times(self):
        sample_space = sa.SampleSpace(["omega0", "omega1"])
        list1 = list(sample_space)
        list2 = list(sample_space)
        assert list1 == list2

    def test_iteration_with_for_loop(self):
        sample_space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        collected = []
        for idx in sample_space:
            collected.append(idx)
        assert collected == ["omega0", "omega1", "omega2"]


class TestEquality:
    def test_equality_same_indices(self):
        sample_space1 = sa.SampleSpace(["omega0", "omega1"])
        sample_space2 = sa.SampleSpace(["omega0", "omega1"])
        assert sample_space1 == sample_space2

    def test_equality_different_indices(self):
        sample_space1 = sa.SampleSpace(["omega0", "omega1"])
        sample_space2 = sa.SampleSpace(["omega0", "omega2"])
        assert sample_space1 != sample_space2

    def test_equality_different_order(self):
        sample_space1 = sa.SampleSpace(["omega0", "omega1"])
        sample_space2 = sa.SampleSpace(["omega1", "omega0"])
        assert sample_space1 != sample_space2

    def test_equality_with_non_sample_space(self):
        sample_space = sa.SampleSpace(["omega0", "omega1"])
        assert sample_space != ["omega0", "omega1"]
        assert sample_space != "not a sample space"
        assert sample_space != 123

    def test_equality_different_sizes(self):
        sample_space1 = sa.SampleSpace(["omega0", "omega1"])
        sample_space2 = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert sample_space1 != sample_space2


class TestConversionMethods:

    def test_make_probability_space_with_probabilities(self):
        sample_space = sa.SampleSpace(["s0", "s1"], name="S")
        probabilities = {"s0": 0.25, "s1": 0.75}
        probability_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sample_space.make_probability_space(
            probability_measure=probability_measure
        )
        assert isinstance(prob_space, sa.ProbabilitySpace)
        assert prob_space.probability_measure == probability_measure
        assert prob_space.sample_space == sample_space

    def test_make_probability_space_with_defaults(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        prob_space = sample_space.make_probability_space()
        assert isinstance(prob_space, sa.ProbabilitySpace)
        assert prob_space.sample_space == sample_space
        expected_sigma_algebra = sa.SigmaAlgebra.power_set(sample_space)
        assert prob_space.sigma_algebra == expected_sigma_algebra

    def test_make_probability_space_with_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space,
            sample_id_to_atom_id={"s0": "A", "s1": "A", "s2": "B"},
        )
        prob_space = sample_space.make_probability_space(sigma_algebra=sigma_algebra)
        assert isinstance(prob_space, sa.ProbabilitySpace)
        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == sigma_algebra

    def test_make_event_space_with_default_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        event_space = sample_space.make_event_space()
        assert isinstance(event_space, sa.EventSpace)
        assert event_space.sample_space == sample_space
        expected_sigma_algebra = sa.SigmaAlgebra.power_set(sample_space)
        assert event_space.sigma_algebra == expected_sigma_algebra

    def test_make_event_space_with_custom_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space,
            sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1},
        )
        event_space = sample_space.make_event_space(sigma_algebra=sigma_algebra)
        assert isinstance(event_space, sa.EventSpace)
        assert event_space.sample_space == sample_space
        assert event_space.sigma_algebra == sigma_algebra
