import pandas as pd
import pytest

from sigalg.core import Event, SampleSpace


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])

    @pytest.mark.parametrize(
        "indices, name, data_name",
        [
            pytest.param(["omega0", "omega1"], "B", "new_name", id="custom_names"),
            pytest.param(["omega0", "omega1"], None, None, id="none_names"),
            pytest.param([], "empty_event", "empty_data", id="empty_indices"),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                "full_event",
                None,
                id="all_sample_points",
            ),
            pytest.param(
                ["omega0"], None, "custom_data", id="single_index_custom_data"
            ),
            pytest.param(
                ["omega1", "omega2"],
                "default_name_flag",
                "data_name",
                id="default_name",
            ),
            pytest.param(
                ["omega1", "omega2"],
                "name",
                "default_data_name_flag",
                id="default_data_name",
            ),
        ],
    )
    def test_constructor(self, sample_space, indices, name, data_name):
        """Test constructor with various combinations of parameters."""
        if name == "default_name_flag":
            event = Event(sample_space=sample_space, data_name=data_name).from_list(
                indices
            )
            name = "A"
        elif data_name == "default_data_name_flag":
            event = Event(sample_space=sample_space, name=name).from_list(indices)
            data_name = "sample"
        else:
            event = Event(
                sample_space=sample_space, name=name, data_name=data_name
            ).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    @pytest.mark.parametrize(
        "indices, name, data_name",
        [
            pytest.param(
                ["omega0", "omega5"], "A", None, id="index_not_in_sample_space"
            ),
            pytest.param("omega0", "A", None, id="indices_not_list"),
            pytest.param(["omega0"], ["not", "hashable"], None, id="name_not_hashable"),
            pytest.param(
                ["omega0"], "A", {"not": "hashable"}, id="data_name_not_hashable"
            ),
        ],
    )
    def test_invalid_inputs_raise(self, sample_space, indices, name, data_name):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises((TypeError, ValueError, KeyError)):
            Event(sample_space=sample_space, name=name, data_name=data_name).from_list(
                indices
            )


class TestGetEvent:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def event(self, sample_space):
        return Event(sample_space=sample_space, name="C").from_list(
            ["omega0", "omega1", "omega2"]
        )

    @pytest.mark.parametrize(
        "indices,name,expected_indices",
        [
            pytest.param(
                ["omega0", "omega2"], "D", ["omega0", "omega2"], id="subset_indices"
            ),
            pytest.param(["omega1"], "E", ["omega1"], id="single_index"),
            pytest.param([], "empty", [], id="empty_indices"),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                "F",
                ["omega0", "omega1", "omega2"],
                id="all_indices",
            ),
        ],
    )
    def test_get_event(self, event, indices, name, expected_indices):
        """Test that get_event method returns a new Event with specified indices."""
        result = event.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == event.sample_space

    @pytest.mark.parametrize(
        "indices",
        [
            pytest.param(["omega0", "omega5"], id="index_not_in_sample_space"),
        ],
    )
    def test_invalid_inputs_raise(self, event, indices):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises((ValueError, KeyError)):
            event.get_event(indices)


class TestGetItem:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def event(self, sample_space):
        return Event(sample_space=sample_space).from_list(
            ["omega0", "omega1", "omega2"],
        )

    @pytest.mark.parametrize(
        "pos,name,expected_indices",
        [
            pytest.param(slice(1, 3), "E", ["omega1", "omega2"], id="slice_index"),
            pytest.param(slice(0, 1), "F", ["omega0"], id="slice_single"),
            pytest.param([0, 2], "G", ["omega0", "omega2"], id="list_index"),
            pytest.param(
                slice(None, None), "H", ["omega0", "omega1", "omega2"], id="slice_all"
            ),
        ],
    )
    def test_getitem(self, event, pos, name, expected_indices):
        """Test that __getitem__ method returns a new Event with specified indices."""
        result = event[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == event.sample_space

    @pytest.mark.parametrize(
        "pos",
        [
            pytest.param([0, 5], id="list_index_out_of_bounds"),
        ],
    )
    def test_invalid_pos_raises(self, event, pos):
        """Test that invalid positions raise IndexError."""
        with pytest.raises(IndexError):
            event[pos, "invalid"]


class TestSetTheoreticOperations:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])

    @pytest.mark.parametrize(
        "indices,expected_complement",
        [
            pytest.param(
                ["omega0", "omega1"], ["omega2", "omega3"], id="basic_complement"
            ),
            pytest.param(
                [], ["omega0", "omega1", "omega2", "omega3"], id="complement_of_empty"
            ),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"], [], id="complement_of_full"
            ),
            pytest.param(
                ["omega0"], ["omega1", "omega2", "omega3"], id="complement_single"
            ),
        ],
    )
    def test_complement(self, sample_space, indices, expected_complement):
        """Test complement of an Event."""
        A = Event(sample_space=sample_space).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    @pytest.mark.parametrize(
        "indices,expected_complement",
        [
            pytest.param(["omega0", "omega1"], ["omega2", "omega3"], id="tilde_basic"),
            pytest.param(["omega0"], ["omega1", "omega2", "omega3"], id="tilde_single"),
        ],
    )
    def test_complement_using_tilde(self, sample_space, indices, expected_complement):
        """Test complement of an Event using ~ operator."""
        B = Event(sample_space=sample_space, name="B").from_list(indices)
        comp = ~B

        assert set(comp.data) == set(expected_complement)
        assert comp.name == "B complement"

    def test_double_complement(self, sample_space):
        """Test that double complement returns the original Event."""
        A = Event(sample_space=sample_space).from_list(["omega0", "omega1"])
        double_comp = ~~A

        assert double_comp == A

    @pytest.mark.parametrize(
        "indices_a,indices_b,expected_union",
        [
            pytest.param(
                ["omega0", "omega1"],
                ["omega2", "omega3"],
                ["omega0", "omega1", "omega2", "omega3"],
                id="disjoint_union",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega1", "omega2"],
                ["omega0", "omega1", "omega2"],
                id="overlapping_union",
            ),
            pytest.param(
                ["omega0", "omega1"],
                [],
                ["omega0", "omega1"],
                id="union_with_empty",
            ),
            pytest.param(
                ["omega0"],
                ["omega0"],
                ["omega0"],
                id="union_identical",
            ),
        ],
    )
    def test_union(self, sample_space, indices_a, indices_b, expected_union):
        """Test union of two Events."""
        A = Event(sample_space=sample_space, name="A").from_list(indices_a)
        B = Event(sample_space=sample_space, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    @pytest.mark.parametrize(
        "indices_a,indices_b,expected_union",
        [
            pytest.param(
                ["omega0", "omega1"],
                ["omega2", "omega3"],
                ["omega0", "omega1", "omega2", "omega3"],
                id="pipe_disjoint",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega1", "omega2"],
                ["omega0", "omega1", "omega2"],
                id="pipe_overlapping",
            ),
        ],
    )
    def test_union_using_pipe(self, sample_space, indices_a, indices_b, expected_union):
        """Test union of two Events using | operator."""
        C = Event(sample_space=sample_space, name="C").from_list(indices_a)
        D = Event(sample_space=sample_space, name="D").from_list(indices_b)
        union = C | D

        assert set(union.data) == set(expected_union)
        assert union.name == "C union D"

    def test_union_with_self(self, sample_space):
        """Test union of an Event with itself."""
        A = Event(sample_space=sample_space).from_list(["omega0", "omega1"])
        union = A | A

        assert union == A

    @pytest.mark.parametrize(
        "indices_a,indices_b,expected_intersection",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                ["omega1", "omega2", "omega3"],
                ["omega1", "omega2"],
                id="overlapping_intersection",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega2", "omega3"],
                [],
                id="disjoint_intersection",
            ),
            pytest.param(
                ["omega0", "omega1"],
                [],
                [],
                id="intersection_with_empty",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega1"],
                ["omega0", "omega1"],
                id="intersection_identical",
            ),
        ],
    )
    def test_intersection(
        self, sample_space, indices_a, indices_b, expected_intersection
    ):
        """Test intersection of two Events."""
        A = Event(sample_space=sample_space, name="A").from_list(indices_a)
        B = Event(sample_space=sample_space, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    @pytest.mark.parametrize(
        "indices_a,indices_b,expected_intersection",
        [
            pytest.param(
                ["omega0", "omega1"],
                ["omega1", "omega2"],
                ["omega1"],
                id="ampersand_overlapping",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega2", "omega3"],
                [],
                id="ampersand_disjoint",
            ),
        ],
    )
    def test_intersection_using_ampersand(
        self, sample_space, indices_a, indices_b, expected_intersection
    ):
        """Test intersection of two Events using & operator."""
        E = Event(sample_space=sample_space, name="E").from_list(indices_a)
        F = Event(sample_space=sample_space, name="F").from_list(indices_b)
        intersection = E & F

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "E intersect F"

    def test_intersection_with_self(self, sample_space):
        """Test intersection of an Event with itself."""
        A = Event(sample_space=sample_space).from_list(["omega0", "omega1"])
        intersection = A & A

        assert intersection == A

    @pytest.mark.parametrize(
        "indices_a,indices_b,expected_difference",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                ["omega1", "omega2"],
                ["omega0"],
                id="basic_difference",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega2", "omega3"],
                ["omega0", "omega1"],
                id="disjoint_difference",
            ),
            pytest.param(
                ["omega0", "omega1"],
                [],
                ["omega0", "omega1"],
                id="difference_with_empty",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega1"],
                [],
                id="difference_identical",
            ),
        ],
    )
    def test_difference(self, sample_space, indices_a, indices_b, expected_difference):
        """Test difference of two Events."""
        A = Event(sample_space=sample_space, name="A").from_list(indices_a)
        B = Event(sample_space=sample_space, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    @pytest.mark.parametrize(
        "indices_a,indices_b,expected_difference",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                ["omega2"],
                ["omega0", "omega1"],
                id="minus_basic",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega2", "omega3"],
                ["omega0", "omega1"],
                id="minus_disjoint",
            ),
        ],
    )
    def test_difference_using_minus(
        self, sample_space, indices_a, indices_b, expected_difference
    ):
        """Test difference of two Events using - operator."""
        G = Event(sample_space=sample_space, name="G").from_list(indices_a)
        H = Event(sample_space=sample_space, name="H").from_list(indices_b)
        difference = G - H

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "G difference H"

    def test_difference_with_self(self, sample_space):
        """Test difference of an Event with itself."""
        A = Event(sample_space=sample_space).from_list(["omega0", "omega1"])
        difference = A - A

        assert len(difference) == 0


class TestSubsetSuperset:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])

    @pytest.mark.parametrize(
        "indices_a,indices_b,is_subset,is_proper_subset",
        [
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega1", "omega2"],
                True,
                True,
                id="proper_subset",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega1"],
                True,
                False,
                id="equal_subset",
            ),
            pytest.param(
                ["omega0", "omega2"],
                ["omega0", "omega1"],
                False,
                False,
                id="not_subset",
            ),
            pytest.param(
                [],
                ["omega0", "omega1"],
                True,
                True,
                id="empty_subset",
            ),
            pytest.param(
                ["omega0"],
                ["omega0", "omega1", "omega2", "omega3"],
                True,
                True,
                id="subset_of_full",
            ),
        ],
    )
    def test_subset(
        self, sample_space, indices_a, indices_b, is_subset, is_proper_subset
    ):
        """Test subset relationships between two Events."""
        A = Event(sample_space=sample_space).from_list(indices_a)
        B = Event(sample_space=sample_space).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    @pytest.mark.parametrize(
        "indices_a,indices_b,is_superset,is_proper_superset",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                ["omega0", "omega1"],
                True,
                True,
                id="proper_superset",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega1"],
                True,
                False,
                id="equal_superset",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega2"],
                False,
                False,
                id="not_superset",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                ["omega0"],
                True,
                True,
                id="full_superset",
            ),
            pytest.param(
                ["omega0", "omega1"],
                [],
                True,
                True,
                id="superset_of_empty",
            ),
        ],
    )
    def test_superset(
        self, sample_space, indices_a, indices_b, is_superset, is_proper_superset
    ):
        """Test superset relationships between two Events."""
        A = Event(sample_space=sample_space).from_list(indices_a)
        B = Event(sample_space=sample_space).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset


class TestEquality:

    @pytest.mark.parametrize(
        "given_sample_space, given_event, other_sample_space, other_event",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                ["omega0", "omega1"],
                ["omega0", "omega1", "omega2"],
                ["omega0", "omega2"],
                id="different_indices",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega0"],
                ["a", "b"],
                ["a"],
                id="different_sample_spaces",
            ),
        ],
    )
    def test_non_equality(
        self, given_sample_space, given_event, other_sample_space, other_event
    ):
        """Test the __eq__ method for inequality."""
        given_sample_space = SampleSpace().from_list(given_sample_space)
        other_sample_space = SampleSpace().from_list(other_sample_space)
        given = Event(sample_space=given_sample_space).from_list(given_event)
        other = Event(sample_space=other_sample_space).from_list(other_event)

        assert given != other

    @pytest.mark.parametrize(
        "given_sample_space, given_event, given_name, other_sample_space, other_event, other_name",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                ["omega0", "omega1"],
                "A",
                ["omega0", "omega1", "omega2"],
                ["omega0", "omega1"],
                "B",
                id="equal_different_names",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                ["omega0", "omega1"],
                "A",
                ["omega0", "omega1", "omega2"],
                ["omega0", "omega1"],
                "B",
                id="equal_all_attributes_match",
            ),
        ],
    )
    def test_equality(
        self,
        given_sample_space,
        given_event,
        given_name,
        other_sample_space,
        other_event,
        other_name,
    ):
        """Test the __eq__ method for equality."""
        given_sample_space = SampleSpace().from_list(given_sample_space)
        other_sample_space = SampleSpace().from_list(other_sample_space)
        given = Event(sample_space=given_sample_space, name=given_name).from_list(
            given_event
        )
        other = Event(sample_space=other_sample_space, name=other_name).from_list(
            other_event
        )

        assert given == other
