import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_construction_with_valid_indices(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"], name="B")
        expected_index = pd.Index(data=["omega0", "omega1"], name="B")
        pd.testing.assert_index_equal(event.values, expected_index)

    def test_construction_preserves_sample_space_order(self, sample_space):
        event = sa.Event(sample_space, ["omega2", "omega0", "omega1"])
        expected_index = pd.Index(data=["omega0", "omega1", "omega2"], name="A")
        pd.testing.assert_index_equal(event.values, expected_index)

    def test_construction_removes_duplicates(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1", "omega0"])
        assert len(event) == 2
        assert list(event.values) == ["omega0", "omega1"]

    def test_construction_with_empty_list(self, sample_space):
        event = sa.Event(sample_space, [])
        assert len(event) == 0


class TestValidation:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.Event("not a space", ["omega0"])

    def test_construction_with_non_list_indices(self, sample_space):
        with pytest.raises(TypeError, match="must be a list"):
            sa.Event(sample_space, {"omega0", "omega1"})

    def test_construction_with_invalid_index(self, sample_space):
        with pytest.raises(ValueError, match="not in sample_space"):
            sa.Event(sample_space, ["omega0", "invalid"])


class TestLen:
    @pytest.fixture
    def event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        return sa.Event(space, ["omega0", "omega1", "omega2"])

    def test_len(self, event):
        assert len(event) == 3


class TestIteration:
    @pytest.fixture
    def event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        return sa.Event(space, ["omega0", "omega1", "omega2"])

    def test_iteration(self, event):
        indices = list(event)
        assert indices == ["omega0", "omega1", "omega2"]


class TestProperties:
    @pytest.fixture
    def event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        return sa.Event(space, ["omega0", "omega1", "omega2"])

    def test_get_event(self, event):
        sub_event = event.get_event(["omega0", "omega2"], name="D")
        expected_index = pd.Index(data=["omega0", "omega2"], name="D")
        pd.testing.assert_index_equal(sub_event.values, expected_index)

    def test_get_event_at(self, event):
        sub_event = event.get_event_at[1:3, "E"]
        expected_index = pd.Index(data=["omega1", "omega2"], name="E")
        pd.testing.assert_index_equal(sub_event.values, expected_index)


class TestSetTheoreticOperations:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_complement_basic(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"], name="A")
        comp = event.complement()
        assert isinstance(comp, sa.Event)
        assert set(comp.values) == {"omega2", "omega3"}
        assert comp.name == "A complement"

    def test_complement_using_tilde(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"], name="B")
        comp = ~event
        assert set(comp.values) == {"omega2", "omega3"}
        assert comp.name == "B complement"

    def test_complement_of_empty_event(self, sample_space):
        event = sa.Event(sample_space, [])
        comp = ~event
        assert set(comp.values) == set(sample_space)

    def test_complement_of_full_event(self, sample_space):
        event = sa.Event(sample_space, list(sample_space))
        comp = ~event
        assert len(comp) == 0

    def test_double_complement(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        double_comp = ~~event
        assert double_comp == event

    def test_union_basic(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"], name="A")
        event_B = sa.Event(sample_space, ["omega2", "omega3"], name="B")
        union = event_A.union(event_B)
        assert set(union.values) == {"omega0", "omega1", "omega2", "omega3"}
        assert union.name == "A union B"

    def test_union_using_pipe(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"], name="C")
        event_B = sa.Event(sample_space, ["omega2", "omega3"], name="D")
        union = event_A | event_B
        assert set(union.values) == {"omega0", "omega1", "omega2", "omega3"}
        assert union.name == "C union D"

    def test_union_with_overlap(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"])
        event_B = sa.Event(sample_space, ["omega1", "omega2"])
        union = event_A | event_B
        assert set(union.values) == {"omega0", "omega1", "omega2"}

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

    def test_intersection_basic(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1", "omega2"], name="A")
        event_B = sa.Event(sample_space, ["omega1", "omega2", "omega3"], name="B")
        intersection = event_A.intersection(event_B)
        assert set(intersection.values) == {"omega1", "omega2"}
        assert intersection.name == "A intersect B"

    def test_intersection_using_ampersand(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1"], name="E")
        event_B = sa.Event(sample_space, ["omega1", "omega2"], name="F")
        intersection = event_A & event_B
        assert set(intersection.values) == {"omega1"}
        assert intersection.name == "E intersect F"

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

    def test_difference_basic(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1", "omega2"], name="A")
        event_B = sa.Event(sample_space, ["omega1", "omega2"], name="B")
        difference = event_A.difference(event_B)
        assert set(difference.values) == {"omega0"}
        assert difference.name == "A difference B"

    def test_difference_using_minus(self, sample_space):
        event_A = sa.Event(sample_space, ["omega0", "omega1", "omega2"], name="G")
        event_B = sa.Event(sample_space, ["omega2"], name="H")
        difference = event_A - event_B
        assert set(difference.values) == {"omega0", "omega1"}
        assert difference.name == "G difference H"

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
