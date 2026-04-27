import pandas as pd
import pytest

from sigalg.core import Event, SampleSpace, SigmaAlgebra


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4, initial_index=0, prefix="omega")

    @pytest.fixture
    def sig_alg(self, sample_space):
        return SigmaAlgebra.power_set(sample_space)

    def test_constructor_custom_names(self, sample_space, sig_alg):
        """Test constructor with custom names."""
        indices = ["omega_0", "omega_1"]
        name = "B"
        data_name = "new_name"
        event = Event(
            sig_alg=sig_alg, name=name, data_name=data_name
        ).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    def test_constructor_none_names(self, sample_space, sig_alg):
        """Test constructor with None names."""
        indices = ["omega_0", "omega_1"]
        name = None
        data_name = None
        event = Event(
            sig_alg=sig_alg, name=name, data_name=data_name
        ).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    def test_constructor_empty_indices(self, sample_space, sig_alg):
        """Test constructor with empty indices."""
        indices = []
        name = "empty_event"
        data_name = "empty_data"
        event = Event(
            sig_alg=sig_alg, name=name, data_name=data_name
        ).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    def test_constructor_all_sample_points(self, sample_space, sig_alg):
        """Test constructor with all sample points."""
        indices = ["omega_0", "omega_1", "omega_2", "omega_3"]
        name = "full_event"
        data_name = None
        event = Event(
            sig_alg=sig_alg, name=name, data_name=data_name
        ).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    def test_constructor_single_index_custom_data(self, sample_space, sig_alg):
        """Test constructor with single index and custom data name."""
        indices = ["omega_0"]
        name = None
        data_name = "custom_data"
        event = Event(
            sig_alg=sig_alg, name=name, data_name=data_name
        ).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    def test_constructor_default_name(self, sample_space, sig_alg):
        """Test constructor with default name."""
        indices = ["omega_1", "omega_2"]
        data_name = "data_name"
        event = Event(sig_alg=sig_alg, data_name=data_name).from_list(indices)
        name = "A"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    def test_constructor_default_data_name(self, sample_space, sig_alg):
        """Test constructor with default data name."""
        indices = ["omega_1", "omega_2"]
        name = "name"
        event = Event(sig_alg=sig_alg, name=name).from_list(indices)
        data_name = "sample"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == sample_space
        assert len(event) == len(indices)

    def test_invalid_index_not_in_sample_space_raises(self, sample_space, sig_alg):
        """Test that index not in sample space raises exception."""
        indices = ["omega_0", "omega_5"]
        name = "A"
        data_name = None
        with pytest.raises((TypeError, ValueError, KeyError)):
            Event(sig_alg=sig_alg, name=name, data_name=data_name).from_list(
                indices
            )

    def test_invalid_indices_not_list_raises(self, sample_space, sig_alg):
        """Test that indices not being a list raises exception."""
        indices = "omega_0"
        name = "A"
        data_name = None
        with pytest.raises((TypeError, ValueError, KeyError)):
            Event(sig_alg=sig_alg, name=name, data_name=data_name).from_list(
                indices
            )

    def test_invalid_name_not_hashable_raises(self, sample_space, sig_alg):
        """Test that non-hashable name raises exception."""
        indices = ["omega_0"]
        name = ["not", "hashable"]
        data_name = None
        with pytest.raises((TypeError, ValueError, KeyError)):
            Event(sig_alg=sig_alg, name=name, data_name=data_name).from_list(
                indices
            )

    def test_invalid_data_name_not_hashable_raises(self, sample_space, sig_alg):
        """Test that non-hashable data name raises exception."""
        indices = ["omega_0"]
        name = "A"
        data_name = {"not": "hashable"}
        with pytest.raises((TypeError, ValueError, KeyError)):
            Event(sig_alg=sig_alg, name=name, data_name=data_name).from_list(
                indices
            )


class TestGetEvent:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4, initial_index=0, prefix="omega")

    @pytest.fixture
    def sig_alg(self, sample_space):
        return SigmaAlgebra.power_set(sample_space)

    @pytest.fixture
    def event(self, sig_alg):
        return Event(sig_alg=sig_alg, name="C").from_list(
            ["omega_0", "omega_1", "omega_2"]
        )

    def test_get_event_subset_indices(self, sig_alg):
        """Test get_event with subset indices."""
        indices = ["omega_0", "omega_2"]
        name = "D"
        expected_indices = ["omega_0", "omega_2"]
        result = sig_alg.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == sig_alg.sample_space

    def test_get_event_single_index(self, sig_alg):
        """Test get_event with single index."""
        indices = ["omega_1"]
        name = "E"
        expected_indices = ["omega_1"]
        result = sig_alg.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == sig_alg.sample_space

    def test_get_event_empty_indices(self, sig_alg):
        """Test get_event with empty indices."""
        indices = []
        name = "empty"
        expected_indices = []
        result = sig_alg.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == sig_alg.sample_space

    def test_get_event_all_indices(self, sig_alg):
        """Test get_event with all indices."""
        indices = ["omega_0", "omega_1", "omega_2", "omega_3"]
        name = "F"
        expected_indices = ["omega_0", "omega_1", "omega_2", "omega_3"]
        result = sig_alg.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == sig_alg.sample_space

    def test_invalid_index_not_in_sample_space_raises(self, sig_alg):
        """Test that index not in sample space raises exception."""
        indices = ["omega_0", "omega_5"]
        with pytest.raises(ValueError):
            sig_alg.get_event(indices)


class TestGetItem:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4, initial_index=0, prefix="omega")

    @pytest.fixture
    def sig_alg(self, sample_space):
        return SigmaAlgebra.power_set(sample_space)

    @pytest.fixture
    def event(self, sig_alg):
        return Event(sig_alg=sig_alg).from_list(
            ["omega_0", "omega_1", "omega_2"],
        )

    def test_getitem_slice_index(self, event):
        """Test __getitem__ with slice index."""
        pos = slice(1, 3)
        name = "E"
        expected_indices = ["omega_1", "omega_2"]
        result = event[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == event.sample_space

    def test_getitem_slice_single(self, event):
        """Test __getitem__ with slice for single element."""
        pos = slice(0, 1)
        name = "F"
        expected_indices = ["omega_0"]
        result = event[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == event.sample_space

    def test_getitem_list_index(self, event):
        """Test __getitem__ with list index."""
        pos = [0, 2]
        name = "G"
        expected_indices = ["omega_0", "omega_2"]
        result = event[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == event.sample_space

    def test_getitem_slice_all(self, event):
        """Test __getitem__ with slice for all elements."""
        pos = slice(None, None)
        name = "H"
        expected_indices = ["omega_0", "omega_1", "omega_2"]
        result = event[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == event.sample_space

    def test_invalid_list_index_out_of_bounds_raises(self, event):
        """Test that list index out of bounds raises IndexError."""
        pos = [0, 5]
        with pytest.raises(IndexError):
            event[pos, "invalid"]


class TestSetTheoreticOperations:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4, initial_index=0, prefix="omega")

    @pytest.fixture
    def sig_alg(self, sample_space):
        return SigmaAlgebra.power_set(sample_space)

    def test_complement_basic(self, sample_space, sig_alg):
        """Test basic complement of an Event."""
        indices = ["omega_0", "omega_1"]
        expected_complement = ["omega_2", "omega_3"]
        A = Event(sig_alg=sig_alg).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_of_empty(self, sample_space, sig_alg):
        """Test complement of empty event."""
        indices = []
        expected_complement = ["omega_0", "omega_1", "omega_2", "omega_3"]
        A = Event(sig_alg=sig_alg).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_of_full(self, sample_space, sig_alg):
        """Test complement of full sample space."""
        indices = ["omega_0", "omega_1", "omega_2", "omega_3"]
        expected_complement = []
        A = Event(sig_alg=sig_alg).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_single(self, sample_space, sig_alg):
        """Test complement of single element event."""
        indices = ["omega_0"]
        expected_complement = ["omega_1", "omega_2", "omega_3"]
        A = Event(sig_alg=sig_alg).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_using_tilde_basic(self, sample_space, sig_alg):
        """Test basic complement using ~ operator."""
        indices = ["omega_0", "omega_1"]
        expected_complement = ["omega_2", "omega_3"]
        B = Event(sig_alg=sig_alg, name="B").from_list(indices)
        comp = ~B

        assert set(comp.data) == set(expected_complement)
        assert comp.name == "B complement"

    def test_complement_using_tilde_single(self, sample_space, sig_alg):
        """Test complement of single element using ~ operator."""
        indices = ["omega_0"]
        expected_complement = ["omega_1", "omega_2", "omega_3"]
        B = Event(sig_alg=sig_alg, name="B").from_list(indices)
        comp = ~B

        assert set(comp.data) == set(expected_complement)
        assert comp.name == "B complement"

    def test_double_complement(self, sample_space, sig_alg):
        """Test that double complement returns the original Event."""
        A = Event(sig_alg=sig_alg).from_list(["omega_0", "omega_1"])
        double_comp = ~~A

        assert double_comp == A

    def test_union_disjoint(self, sample_space, sig_alg):
        """Test union of disjoint Events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_2", "omega_3"]
        expected_union = ["omega_0", "omega_1", "omega_2", "omega_3"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_overlapping(self, sample_space, sig_alg):
        """Test union of overlapping Events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_1", "omega_2"]
        expected_union = ["omega_0", "omega_1", "omega_2"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_with_empty(self, sample_space, sig_alg):
        """Test union with empty event."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = []
        expected_union = ["omega_0", "omega_1"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_identical(self, sample_space, sig_alg):
        """Test union of identical Events."""
        indices_a = ["omega_0"]
        indices_b = ["omega_0"]
        expected_union = ["omega_0"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_using_pipe_disjoint(self, sample_space, sig_alg):
        """Test union using | operator with disjoint events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_2", "omega_3"]
        expected_union = ["omega_0", "omega_1", "omega_2", "omega_3"]
        C = Event(sig_alg=sig_alg, name="C").from_list(indices_a)
        D = Event(sig_alg=sig_alg, name="D").from_list(indices_b)
        union = C | D

        assert set(union.data) == set(expected_union)
        assert union.name == "C union D"

    def test_union_using_pipe_overlapping(self, sample_space, sig_alg):
        """Test union using | operator with overlapping events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_1", "omega_2"]
        expected_union = ["omega_0", "omega_1", "omega_2"]
        C = Event(sig_alg=sig_alg, name="C").from_list(indices_a)
        D = Event(sig_alg=sig_alg, name="D").from_list(indices_b)
        union = C | D

        assert set(union.data) == set(expected_union)
        assert union.name == "C union D"

    def test_union_with_self(self, sample_space, sig_alg):
        """Test union of an Event with itself."""
        A = Event(sig_alg=sig_alg).from_list(["omega_0", "omega_1"])
        union = A | A

        assert union == A

    def test_intersection_overlapping(self, sample_space, sig_alg):
        """Test intersection of overlapping Events."""
        indices_a = ["omega_0", "omega_1", "omega_2"]
        indices_b = ["omega_1", "omega_2", "omega_3"]
        expected_intersection = ["omega_1", "omega_2"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_disjoint(self, sample_space, sig_alg):
        """Test intersection of disjoint Events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_2", "omega_3"]
        expected_intersection = []
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_with_empty(self, sample_space, sig_alg):
        """Test intersection with empty event."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = []
        expected_intersection = []
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_identical(self, sample_space, sig_alg):
        """Test intersection of identical Events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_0", "omega_1"]
        expected_intersection = ["omega_0", "omega_1"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_using_ampersand_overlapping(self, sample_space, sig_alg):
        """Test intersection using & operator with overlapping events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_1", "omega_2"]
        expected_intersection = ["omega_1"]
        E = Event(sig_alg=sig_alg, name="E").from_list(indices_a)
        F = Event(sig_alg=sig_alg, name="F").from_list(indices_b)
        intersection = E & F

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "E intersect F"

    def test_intersection_using_ampersand_disjoint(self, sample_space, sig_alg):
        """Test intersection using & operator with disjoint events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_2", "omega_3"]
        expected_intersection = []
        E = Event(sig_alg=sig_alg, name="E").from_list(indices_a)
        F = Event(sig_alg=sig_alg, name="F").from_list(indices_b)
        intersection = E & F

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "E intersect F"

    def test_intersection_with_self(self, sample_space, sig_alg):
        """Test intersection of an Event with itself."""
        A = Event(sig_alg=sig_alg).from_list(["omega_0", "omega_1"])
        intersection = A & A

        assert intersection == A

    def test_difference_basic(self, sample_space, sig_alg):
        """Test basic difference of two Events."""
        indices_a = ["omega_0", "omega_1", "omega_2"]
        indices_b = ["omega_1", "omega_2"]
        expected_difference = ["omega_0"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_disjoint(self, sample_space, sig_alg):
        """Test difference of disjoint Events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_2", "omega_3"]
        expected_difference = ["omega_0", "omega_1"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_with_empty(self, sample_space, sig_alg):
        """Test difference with empty event."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = []
        expected_difference = ["omega_0", "omega_1"]
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_identical(self, sample_space, sig_alg):
        """Test difference of identical Events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_0", "omega_1"]
        expected_difference = []
        A = Event(sig_alg=sig_alg, name="A").from_list(indices_a)
        B = Event(sig_alg=sig_alg, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_using_minus_basic(self, sample_space, sig_alg):
        """Test difference using - operator with basic case."""
        indices_a = ["omega_0", "omega_1", "omega_2"]
        indices_b = ["omega_2"]
        expected_difference = ["omega_0", "omega_1"]
        G = Event(sig_alg=sig_alg, name="G").from_list(indices_a)
        H = Event(sig_alg=sig_alg, name="H").from_list(indices_b)
        difference = G - H

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "G difference H"

    def test_difference_using_minus_disjoint(self, sample_space, sig_alg):
        """Test difference using - operator with disjoint events."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_2", "omega_3"]
        expected_difference = ["omega_0", "omega_1"]
        G = Event(sig_alg=sig_alg, name="G").from_list(indices_a)
        H = Event(sig_alg=sig_alg, name="H").from_list(indices_b)
        difference = G - H

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "G difference H"

    def test_difference_with_self(self, sample_space, sig_alg):
        """Test difference of an Event with itself."""
        A = Event(sig_alg=sig_alg).from_list(["omega_0", "omega_1"])
        difference = A - A

        assert len(difference) == 0


class TestSubsetSuperset:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4, initial_index=0, prefix="omega")

    @pytest.fixture
    def sig_alg(self, sample_space):
        return SigmaAlgebra.power_set(sample_space)

    def test_subset_proper_subset(self, sample_space, sig_alg):
        """Test proper subset relationship."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_0", "omega_1", "omega_2"]
        is_subset = True
        is_proper_subset = True
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_equal_subset(self, sample_space, sig_alg):
        """Test equal subset relationship."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_0", "omega_1"]
        is_subset = True
        is_proper_subset = False
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_not_subset(self, sample_space, sig_alg):
        """Test not subset relationship."""
        indices_a = ["omega_0", "omega_2"]
        indices_b = ["omega_0", "omega_1"]
        is_subset = False
        is_proper_subset = False
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_empty_subset(self, sample_space, sig_alg):
        """Test empty subset relationship."""
        indices_a = []
        indices_b = ["omega_0", "omega_1"]
        is_subset = True
        is_proper_subset = True
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_subset_of_full(self, sample_space, sig_alg):
        """Test subset of full sample space."""
        indices_a = ["omega_0"]
        indices_b = ["omega_0", "omega_1", "omega_2", "omega_3"]
        is_subset = True
        is_proper_subset = True
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_superset_proper_superset(self, sample_space, sig_alg):
        """Test proper superset relationship."""
        indices_a = ["omega_0", "omega_1", "omega_2"]
        indices_b = ["omega_0", "omega_1"]
        is_superset = True
        is_proper_superset = True
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_equal_superset(self, sample_space, sig_alg):
        """Test equal superset relationship."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_0", "omega_1"]
        is_superset = True
        is_proper_superset = False
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_not_superset(self, sample_space, sig_alg):
        """Test not superset relationship."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = ["omega_0", "omega_2"]
        is_superset = False
        is_proper_superset = False
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_full_superset(self, sample_space, sig_alg):
        """Test full superset relationship."""
        indices_a = ["omega_0", "omega_1", "omega_2", "omega_3"]
        indices_b = ["omega_0"]
        is_superset = True
        is_proper_superset = True
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_superset_of_empty(self, sample_space, sig_alg):
        """Test superset of empty event."""
        indices_a = ["omega_0", "omega_1"]
        indices_b = []
        is_superset = True
        is_proper_superset = True
        A = Event(sig_alg=sig_alg).from_list(indices_a)
        B = Event(sig_alg=sig_alg).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset


class TestEquality:

    def test_non_equality_different_indices(self):
        """Test inequality with different indices."""
        given_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given_event = ["omega_0", "omega_1"]
        other_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        other_event = ["omega_0", "omega_2"]
        given = Event(sig_alg=SigmaAlgebra.power_set(given_sample_space)).from_list(given_event)
        other = Event(sig_alg=SigmaAlgebra.power_set(other_sample_space)).from_list(other_event)

        assert given != other

    def test_non_equality_different_sample_spaces(self):
        """Test inequality with different sample spaces."""
        given_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        given_event = ["omega_0"]
        other_sample_space = SampleSpace(name="Omega", data_name="sample").from_list(
            ["a", "b"]
        )
        other_event = ["a"]
        given = Event(sig_alg=SigmaAlgebra.power_set(given_sample_space)).from_list(given_event)
        other = Event(sig_alg=SigmaAlgebra.power_set(other_sample_space)).from_list(other_event)

        assert given != other

    def test_equality_different_names(self):
        """Test equality with different names."""
        given_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given_event = ["omega_0", "omega_1"]
        given_name = "A"
        other_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        other_event = ["omega_0", "omega_1"]
        other_name = "B"
        given = Event(sig_alg=SigmaAlgebra.power_set(given_sample_space), name=given_name).from_list(
            given_event
        )
        other = Event(sig_alg=SigmaAlgebra.power_set(other_sample_space), name=other_name).from_list(
            other_event
        )

        assert given == other

    def test_equality_all_attributes_match(self):
        """Test equality with all attributes matching."""
        given_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given_event = ["omega_0", "omega_1"]
        given_name = "A"
        other_sample_space = SampleSpace(name="Omega", data_name="sample").from_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        other_event = ["omega_0", "omega_1"]
        other_name = "B"
        given = Event(sig_alg=SigmaAlgebra.power_set(given_sample_space), name=given_name).from_list(
            given_event
        )
        other = Event(sig_alg=SigmaAlgebra.power_set(other_sample_space), name=other_name).from_list(
            other_event
        )

        assert given == other
