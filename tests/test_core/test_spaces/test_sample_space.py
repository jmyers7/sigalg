import pytest

import sigalg as sa


class TestConstruction:
    def test_construction_with_valid_list(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert len(space) == 3
        assert list(space.index) == ["omega0", "omega1", "omega2"]

    def test_construction_with_integers(self):
        space = sa.SampleSpace([1, 2, 3])
        assert len(space) == 3
        assert list(space.index) == [1, 2, 3]

    def test_construction_with_mixed_types(self):
        space = sa.SampleSpace(["a", 1, (2, 3)])
        assert len(space) == 3

    def test_construction_with_duplicates_raises_error(self):
        with pytest.raises(ValueError, match="must be unique"):
            sa.SampleSpace(["omega0", "omega1", "omega0"])

    def test_construction_with_non_list_raises_error(self):
        with pytest.raises(TypeError, match="must be provided as a list"):
            sa.SampleSpace({"omega0", "omega1"})

    def test_construction_with_empty_list_raises_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            sa.SampleSpace([])

    def test_construction_preserves_order(self):
        space = sa.SampleSpace(["z", "a", "m"])
        assert list(space.index) == ["z", "a", "m"]


class TestIndexProperty:
    def test_index_property_returns_pandas_index(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        import pandas as pd

        assert isinstance(space.index, pd.Index)

    def test_index_property_has_correct_values(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert list(space.index) == ["omega0", "omega1", "omega2"]

    def test_index_property_is_immutable(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        original_index = space.index
        assert space.index.equals(original_index)


class TestGetEvent:
    @pytest.fixture
    def space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_get_event_returns_event(self, space):
        event = space.get_event(["omega0", "omega1"])
        assert isinstance(event, sa.Event)

    def test_get_event_with_valid_indices(self, space):
        event = space.get_event(["omega0", "omega2"])
        assert list(event.index) == ["omega0", "omega2"]

    def test_get_event_with_empty_list(self, space):
        event = space.get_event([])
        assert len(event) == 0

    def test_get_event_with_non_list_raises_error(self, space):
        with pytest.raises(TypeError, match="must be a list"):
            space.get_event("omega0")

    def test_get_event_with_invalid_index_raises_error(self, space):
        with pytest.raises(ValueError, match="not found in sample space"):
            space.get_event(["omega0", "invalid"])


class TestLen:
    def test_len_returns_correct_size(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert len(space) == 3

    def test_len_single_element(self):
        space = sa.SampleSpace(["omega0"])
        assert len(space) == 1

    def test_len_large_space(self):
        indices = [f"omega{i}" for i in range(100)]
        space = sa.SampleSpace(indices)
        assert len(space) == 100


class TestGetItem:
    @pytest.fixture
    def space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_getitem_with_single_string_index(self, space):
        result = space["omega0"]
        assert result == "omega0" or isinstance(result, int)

    def test_getitem_with_integer_positional_access(self, space):
        result = space[0]
        assert result == "omega0"

    def test_getitem_with_negative_index(self, space):
        result = space[-1]
        assert result == "omega2"

    def test_getitem_with_list_returns_event(self, space):
        event = space[["omega0", "omega1"]]
        assert isinstance(event, sa.Event)
        assert list(event.index) == ["omega0", "omega1"]

    def test_getitem_with_empty_list(self, space):
        event = space[[]]
        assert isinstance(event, sa.Event)
        assert len(event) == 0

    def test_getitem_with_invalid_index_raises_error(self, space):
        with pytest.raises((KeyError, IndexError)):
            space["invalid"]

    def test_getitem_with_out_of_range_integer(self, space):
        with pytest.raises(IndexError):
            space[10]


class TestIter:
    def test_iteration_returns_all_indices(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        indices = list(space)
        assert indices == ["omega0", "omega1", "omega2"]

    def test_iteration_preserves_order(self):
        space = sa.SampleSpace(["z", "a", "m"])
        indices = list(space)
        assert indices == ["z", "a", "m"]

    def test_can_iterate_multiple_times(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        list1 = list(space)
        list2 = list(space)
        assert list1 == list2

    def test_iteration_with_for_loop(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        collected = []
        for idx in space:
            collected.append(idx)
        assert collected == ["omega0", "omega1", "omega2"]


class TestEquality:
    def test_equality_same_indices(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1"])
        assert space1 == space2

    def test_equality_different_indices(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega2"])
        assert space1 != space2

    def test_equality_different_order(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega1", "omega0"])
        assert space1 != space2

    def test_equality_with_non_sample_space(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        assert space != ["omega0", "omega1"]
        assert space != "not a sample space"
        assert space != 123

    def test_equality_different_sizes(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert space1 != space2


class TestHashing:
    def test_sample_space_is_hashable(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        hash_value = hash(space)
        assert isinstance(hash_value, int)

    def test_equal_spaces_have_same_hash(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1"])
        assert hash(space1) == hash(space2)

    def test_different_spaces_likely_different_hash(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega2"])
        assert hash(space1) != hash(space2)

    def test_can_use_in_set(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega2"])
        space_set = {space1, space2}
        assert len(space_set) == 2

    def test_can_use_as_dict_key(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega2"])
        space_dict = {space1: "value1", space2: "value2"}
        assert space_dict[space1] == "value1"

    def test_hash_is_cached(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        hash1 = hash(space)
        hash2 = hash(space)
        assert hash1 == hash2
        assert hasattr(space, "_cached_hash")


class TestEdgeCases:
    def test_single_element_space(self):
        space = sa.SampleSpace(["omega0"])
        assert len(space) == 1
        assert list(space) == ["omega0"]

    def test_large_space(self):
        indices = [f"omega{i}" for i in range(1000)]
        space = sa.SampleSpace(indices)
        assert len(space) == 1000

    def test_space_with_tuple_indices(self):
        space = sa.SampleSpace([(1, 2), (3, 4), (5, 6)])
        assert len(space) == 3
        assert (1, 2) in list(space)

    def test_space_with_integer_indices(self):
        space = sa.SampleSpace([10, 20, 30])
        assert len(space) == 3
        assert space[0] == 10

    def test_contains_check(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert "omega0" in list(space)
        assert "invalid" not in list(space)
