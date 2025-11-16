import pytest

import sigalg as sa


class TestConstruction:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_construction_with_valid_indices(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert len(event) == 2
        assert list(event.index) == ["omega0", "omega1"]

    def test_construction_preserves_sample_space_order(self, sample_space):
        # Even if we pass indices out of order, they should be ordered by sample space
        event = sa.Event(sample_space, ["omega2", "omega0", "omega1"])
        assert list(event.index) == ["omega0", "omega1", "omega2"]

    def test_construction_removes_duplicates(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1", "omega0"])
        assert len(event) == 2
        assert list(event.index) == ["omega0", "omega1"]

    def test_construction_with_empty_list(self, sample_space):
        event = sa.Event(sample_space, [])
        assert len(event) == 0
        assert list(event.index) == []

    def test_construction_with_all_indices(self, sample_space):
        all_indices = list(sample_space)
        event = sa.Event(sample_space, all_indices)
        assert len(event) == len(sample_space)

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.Event("not a space", ["omega0"])

    def test_construction_with_non_list_indices(self, sample_space):
        with pytest.raises(TypeError, match="must be a list"):
            sa.Event(sample_space, {"omega0", "omega1"})

    def test_construction_with_invalid_index(self, sample_space):
        with pytest.raises(ValueError, match="not in sample_space"):
            sa.Event(sample_space, ["omega0", "invalid"])


class TestProperties:
    @pytest.fixture
    def event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        return sa.Event(space, ["omega0", "omega2"])

    def test_index_property(self, event):
        import pandas as pd

        assert isinstance(event.index, pd.Index)
        assert list(event.index) == ["omega0", "omega2"]

    def test_sample_space_property(self, event):
        assert isinstance(event.sample_space, sa.SampleSpace)
        assert len(event.sample_space) == 3


class TestSequenceMethods:
    @pytest.fixture
    def event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        return sa.Event(space, ["omega0", "omega1", "omega2"])

    def test_len(self, event):
        assert len(event) == 3

    def test_iteration(self, event):
        indices = list(event)
        assert indices == ["omega0", "omega1", "omega2"]

    def test_getitem_with_integer(self, event):
        assert event[0] == "omega0"
        assert event[1] == "omega1"
        assert event[-1] == "omega2"

    def test_getitem_with_list_returns_event(self, event):
        sub_event = event[["omega0", "omega2"]]
        assert isinstance(sub_event, sa.Event)
        assert list(sub_event.index) == ["omega0", "omega2"]

    def test_getitem_with_invalid_list_index(self, event):
        with pytest.raises(ValueError, match="not found in this event"):
            event[["omega0", "omega3"]]

    def test_getitem_with_slice(self, event):
        result = event[0:2]
        assert list(result) == ["omega0", "omega1"]


class TestComplementMethod:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_complement_basic(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        comp = event.complement()
        assert isinstance(comp, sa.Event)
        assert set(comp.index) == {"omega2", "omega3"}

    def test_complement_using_tilde(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        comp = ~event
        assert set(comp.index) == {"omega2", "omega3"}

    def test_complement_of_empty_event(self, sample_space):
        event = sa.Event(sample_space, [])
        comp = ~event
        assert set(comp.index) == set(sample_space)

    def test_complement_of_full_event(self, sample_space):
        event = sa.Event(sample_space, list(sample_space))
        comp = ~event
        assert len(comp) == 0

    def test_double_complement(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        double_comp = ~~event
        assert double_comp == event


class TestUnionMethod:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_union_basic(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega2", "omega3"])
        union = event_A.union(event_B)
        assert set(union.index) == {"omega0", "omega1", "omega2", "omega3"}

    def test_union_using_pipe(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega2", "omega3"])
        union = event_A | event_B
        assert set(union.index) == {"omega0", "omega1", "omega2", "omega3"}

    def test_union_with_overlap(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega1", "omega2"])
        union = event_A | event_B
        assert set(union.index) == {"omega0", "omega1", "omega2"}

    def test_union_with_self(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        union = event | event
        assert union == event

    def test_union_with_empty(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        empty = sa.Event(sample_space, [])
        union = event | empty
        assert union == event

    def test_union_different_spaces_raises_error(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        event1 = sa.Event(space1, ["omega0"])
        event2 = sa.Event(space2, ["a"])
        with pytest.raises(ValueError, match="same sample space"):
            event1 | event2


class TestIntersectionMethod:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_intersection_basic(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1", "omega2"])
        event_B = sa.Event(sample_space, ["omega1", "omega2", "omega3"])
        intersection = event_A.intersection(event_B)
        assert set(intersection.index) == {"omega1", "omega2"}

    def test_intersection_using_ampersand(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega1", "omega2"])
        intersection = event_A & event_B
        assert set(intersection.index) == {"omega1"}

    def test_intersection_disjoint(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega2", "omega3"])
        intersection = event_A & event_B
        assert len(intersection) == 0

    def test_intersection_with_self(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        intersection = event & event
        assert intersection == event

    def test_intersection_with_empty(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        empty = sa.Event(sample_space, [])
        intersection = event & empty
        assert len(intersection) == 0

    def test_intersection_different_spaces_raises_error(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        event1 = sa.Event(space1, ["omega0"])
        event2 = sa.Event(space2, ["a"])
        with pytest.raises(ValueError, match="same sample space"):
            event1 & event2


class TestDifferenceMethod:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_difference_basic(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1", "omega2"])
        event_B = sa.Event(sample_space, ["omega1", "omega2"])
        difference = event_A.difference(event_B)
        assert set(difference.index) == {"omega0"}

    def test_difference_using_minus(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1", "omega2"])
        event_B = sa.Event(sample_space, ["omega2"])
        difference = event_A - event_B
        assert set(difference.index) == {"omega0", "omega1"}

    def test_difference_disjoint(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega2", "omega3"])
        difference = event_A - event_B
        assert difference == event_A

    def test_difference_with_self(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        difference = event - event
        assert len(difference) == 0

    def test_difference_with_empty(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        empty = sa.Event(sample_space, [])
        difference = event - empty
        assert difference == event

    def test_difference_different_spaces_raises_error(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        event1 = sa.Event(space1, ["omega0"])
        event2 = sa.Event(space2, ["a"])
        with pytest.raises(ValueError, match="same sample space"):
            event1 - event2


class TestSubsetSuperset:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_subset_proper(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega0", "omega1", "omega2"])
        assert event_A <= event_B
        assert event_A < event_B

    def test_subset_equal(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega0", "omega1"])
        assert event_A <= event_B
        assert not (event_A < event_B)

    def test_not_subset(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega2"])
        event_B = sa.Event(sample_space, ["omega0", "omega1"])
        assert not (event_A <= event_B)
        assert not (event_A < event_B)

    def test_superset_proper(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1", "omega2"])
        event_B = sa.Event(sample_space, ["omega0", "omega1"])
        assert event_A >= event_B
        assert event_A > event_B

    def test_superset_equal(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega0", "omega1"])
        assert event_A >= event_B
        assert not (event_A > event_B)

    def test_empty_subset_of_all(self, sample_space):
        empty = sa.Event(sample_space, [])
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert empty <= event
        assert empty < event

    def test_all_superset_of_all(self, sample_space):
        full = sa.Event(sample_space, list(sample_space))
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert full >= event
        assert full > event


class TestEquality:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_equality_same_indices(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega0", "omega1"])
        assert event1 == event2

    def test_equality_different_indices(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega0", "omega2"])
        assert event1 != event2

    def test_equality_different_sample_spaces(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        event1 = sa.Event(space1, ["omega0"])
        event2 = sa.Event(space2, ["a"])
        assert event1 != event2

    def test_equality_with_non_event(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert event != ["omega0", "omega1"]
        assert event != "not an event"
        assert event != 123


class TestHashing:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_event_is_hashable(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        hash_value = hash(event)
        assert isinstance(hash_value, int)

    def test_equal_events_have_same_hash(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega0", "omega1"])
        assert hash(event1) == hash(event2)

    def test_can_use_in_set(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega0", "omega2"])
        event_set = {event1, event2}
        assert len(event_set) == 2

    def test_can_use_as_dict_key(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega0", "omega2"])
        event_dict = {event1: "value1", event2: "value2"}
        assert event_dict[event1] == "value1"


class TestDeMorgansLaws:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_de_morgan_union(self, sample_space):
        A = sa.Event(sample_space, ["omega0", "omega1"])
        B = sa.Event(sample_space, ["omega1", "omega2"])
        left = ~(A | B)
        right = (~A) & (~B)
        assert left == right

    def test_de_morgan_intersection(self, sample_space):
        A = sa.Event(sample_space, ["omega0", "omega1"])
        B = sa.Event(sample_space, ["omega1", "omega2"])
        left = ~(A & B)
        right = (~A) | (~B)
        assert left == right


class TestEdgeCases:
    def test_empty_event(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        event = sa.Event(space, [])
        assert len(event) == 0
        assert list(event) == []

    def test_single_element_event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        event = sa.Event(space, ["omega1"])
        assert len(event) == 1
        assert list(event) == ["omega1"]

    def test_full_space_event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        event = sa.Event(space, list(space))
        assert len(event) == len(space)
        assert set(event.index) == set(space)

    def test_operations_with_empty_events(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        empty = sa.Event(space, [])
        event = sa.Event(space, ["omega0", "omega1"])

        assert (empty | event) == event
        assert (empty & event) == empty
        assert (event - empty) == event
        assert len(~empty) == len(space)
