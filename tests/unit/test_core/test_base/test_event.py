import pandas as pd
import pytest

from sigalg.core import Event, SampleSpace, SigmaAlgebra


class TestConstructor:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    def test_constructor_custom_names(self, Omega, F):
        """Test constructor with custom names."""
        indices = [0, 1]
        name = "B"
        data_name = "new_name"
        event = Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == Omega
        assert len(event) == len(indices)

    def test_constructor_none_names(self, Omega, F):
        """Test constructor with None names."""
        indices = [0, 1]
        name = None
        data_name = None
        event = Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == Omega
        assert len(event) == len(indices)

    def test_constructor_empty_indices(self, Omega, F):
        """Test constructor with empty indices."""
        indices = []
        name = "A"
        data_name = "sample"
        event = Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == Omega
        assert len(event) == len(indices)

    def test_constructor_all_sample_points(self, Omega, F):
        """Test constructor with all sample points."""
        indices = [0, 1, 2, 3]
        name = "A"
        data_name = None
        event = Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == Omega
        assert len(event) == len(indices)

    def test_constructor_single_index_custom_data(self, Omega, F):
        """Test constructor with single index and custom data name."""
        indices = [0]
        name = None
        data_name = "custom_data"
        event = Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == Omega
        assert len(event) == len(indices)

    def test_constructor_default_name(self, Omega, F):
        """Test constructor with default name."""
        indices = [1, 2]
        data_name = "sample"
        event = Event(sig_alg=F, data_name=data_name).from_list(indices)
        name = "A"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == Omega
        assert len(event) == len(indices)

    def test_constructor_default_data_name(self, Omega, F):
        """Test constructor with default data name."""
        indices = [1, 2]
        name = "A"
        event = Event(sig_alg=F, name=name).from_list(indices)
        data_name = "sample"
        expected_index = pd.Index(data=indices, name=data_name)

        pd.testing.assert_index_equal(event.data, expected_index)
        assert event.name == name
        assert event.sample_space == Omega
        assert len(event) == len(indices)

    def test_invalid_index_not_in_sample_space_raises(self, Omega, F):
        """Test that index not in sample space raises exception."""
        indices = [0, 5]
        name = "A"
        data_name = None
        with pytest.raises(ValueError):
            Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)

    def test_invalid_indices_not_list_raises(self, Omega, F):
        """Test that indices not being a list raises exception."""
        indices = 0
        name = "A"
        data_name = None
        with pytest.raises(TypeError):
            Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)

    def test_invalid_name_not_hashable_raises(self, Omega, F):
        """Test that non-hashable name raises exception."""
        indices = [0]
        name = ["not", "hashable"]
        data_name = None
        with pytest.raises(TypeError):
            Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)

    def test_invalid_data_name_not_hashable_raises(self, Omega, F):
        """Test that non-hashable data name raises exception."""
        indices = [0]
        name = "A"
        data_name = {"not": "hashable"}
        with pytest.raises(TypeError):
            Event(sig_alg=F, name=name, data_name=data_name).from_list(indices)


class TestGetEvent:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def A(self, F):
        return Event(sig_alg=F, name="A").from_list([0, 1, 2])

    def test_get_event_subset_indices(self, F):
        """Test get_event with subset indices."""
        indices = [0, 2]
        name = "B"
        expected_indices = [0, 2]
        result = F.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == F.sample_space

    def test_get_event_single_index(self, F):
        """Test get_event with single index."""
        indices = [1]
        name = "C"
        expected_indices = [1]
        result = F.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == F.sample_space

    def test_get_event_empty_indices(self, F):
        """Test get_event with empty indices."""
        indices = []
        name = "empty"
        expected_indices = []
        result = F.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == F.sample_space

    def test_get_event_all_indices(self, F):
        """Test get_event with all indices."""
        indices = [0, 1, 2, 3]
        name = "D"
        expected_indices = [0, 1, 2, 3]
        result = F.get_event(indices, name=name)
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == F.sample_space

    def test_invalid_index_not_in_sample_space_raises(self, F):
        """Test that index not in sample space raises exception."""
        indices = [0, 5]
        with pytest.raises(ValueError):
            F.get_event(indices)


class TestGetItem:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def A(self, F):
        return Event(sig_alg=F).from_list([0, 1, 2])

    def test_getitem_slice_index(self, A):
        """Test __getitem__ with slice index."""
        pos = slice(1, 3)
        name = "B"
        expected_indices = [1, 2]
        result = A[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == A.sample_space

    def test_getitem_slice_single(self, A):
        """Test __getitem__ with slice for single element."""
        pos = slice(0, 1)
        name = "C"
        expected_indices = [0]
        result = A[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == A.sample_space

    def test_getitem_list_index(self, A):
        """Test __getitem__ with list index."""
        pos = [0, 2]
        name = "D"
        expected_indices = [0, 2]
        result = A[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == A.sample_space

    def test_getitem_slice_all(self, A):
        """Test __getitem__ with slice for all elements."""
        pos = slice(None, None)
        name = "E"
        expected_indices = [0, 1, 2]
        result = A[pos, name]
        expected_index = pd.Index(data=expected_indices, name="sample")

        assert isinstance(result, Event)
        pd.testing.assert_index_equal(result.data, expected_index)
        assert result.name == name
        assert result.sample_space == A.sample_space

    def test_invalid_list_index_out_of_bounds_raises(self, A):
        """Test that list index out of bounds raises IndexError."""
        pos = [0, 5]
        with pytest.raises(IndexError):
            A[pos, "invalid"]


class TestSetTheoreticOperations:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    def test_complement_basic(self, Omega, F):
        """Test basic complement of an Event."""
        indices = [0, 1]
        expected_complement = [2, 3]
        A = Event(sig_alg=F).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_of_empty(self, Omega, F):
        """Test complement of empty event."""
        indices = []
        expected_complement = [0, 1, 2, 3]
        A = Event(sig_alg=F).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_of_full(self, Omega, F):
        """Test complement of full sample space."""
        indices = [0, 1, 2, 3]
        expected_complement = []
        A = Event(sig_alg=F).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_single(self, Omega, F):
        """Test complement of single element event."""
        indices = [0]
        expected_complement = [1, 2, 3]
        A = Event(sig_alg=F).from_list(indices)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_using_tilde_basic(self, Omega, F):
        """Test basic complement using ~ operator."""
        indices = [0, 1]
        expected_complement = [2, 3]
        B = Event(sig_alg=F, name="B").from_list(indices)
        comp = ~B

        assert set(comp.data) == set(expected_complement)
        assert comp.name == "B complement"

    def test_complement_using_tilde_single(self, Omega, F):
        """Test complement of single element using ~ operator."""
        indices = [0]
        expected_complement = [1, 2, 3]
        B = Event(sig_alg=F, name="B").from_list(indices)
        comp = ~B

        assert set(comp.data) == set(expected_complement)
        assert comp.name == "B complement"

    def test_double_complement(self, Omega, F):
        """Test that double complement returns the original Event."""
        A = Event(sig_alg=F).from_list([0, 1])
        double_comp = ~~A

        assert double_comp == A

    def test_union_disjoint(self, Omega, F):
        """Test union of disjoint Events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_union = [0, 1, 2, 3]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_overlapping(self, Omega, F):
        """Test union of overlapping Events."""
        indices_a = [0, 1]
        indices_b = [1, 2]
        expected_union = [0, 1, 2]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_with_empty(self, Omega, F):
        """Test union with empty event."""
        indices_a = [0, 1]
        indices_b = []
        expected_union = [0, 1]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_identical(self, Omega, F):
        """Test union of identical Events."""
        indices_a = [0]
        indices_b = [0]
        expected_union = [0]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_using_pipe_disjoint(self, Omega, F):
        """Test union using | operator with disjoint events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_union = [0, 1, 2, 3]
        C = Event(sig_alg=F, name="C").from_list(indices_a)
        D = Event(sig_alg=F, name="D").from_list(indices_b)
        union = C | D

        assert set(union.data) == set(expected_union)
        assert union.name == "C union D"

    def test_union_using_pipe_overlapping(self, Omega, F):
        """Test union using | operator with overlapping events."""
        indices_a = [0, 1]
        indices_b = [1, 2]
        expected_union = [0, 1, 2]
        C = Event(sig_alg=F, name="C").from_list(indices_a)
        D = Event(sig_alg=F, name="D").from_list(indices_b)
        union = C | D

        assert set(union.data) == set(expected_union)
        assert union.name == "C union D"

    def test_union_with_self(self, Omega, F):
        """Test union of an Event with itself."""
        A = Event(sig_alg=F).from_list([0, 1])
        union = A | A

        assert union == A

    def test_intersection_overlapping(self, Omega, F):
        """Test intersection of overlapping Events."""
        indices_a = [0, 1, 2]
        indices_b = [1, 2, 3]
        expected_intersection = [1, 2]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_disjoint(self, Omega, F):
        """Test intersection of disjoint Events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_intersection = []
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_with_empty(self, Omega, F):
        """Test intersection with empty event."""
        indices_a = [0, 1]
        indices_b = []
        expected_intersection = []
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_identical(self, Omega, F):
        """Test intersection of identical Events."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        expected_intersection = [0, 1]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_using_ampersand_overlapping(self, Omega, F):
        """Test intersection using & operator with overlapping events."""
        indices_a = [0, 1]
        indices_b = [1, 2]
        expected_intersection = [1]
        E = Event(sig_alg=F, name="E").from_list(indices_a)
        F = Event(sig_alg=F, name="F").from_list(indices_b)
        intersection = E & F

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "E intersect F"

    def test_intersection_using_ampersand_disjoint(self, Omega, F):
        """Test intersection using & operator with disjoint events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_intersection = []
        E = Event(sig_alg=F, name="E").from_list(indices_a)
        F = Event(sig_alg=F, name="F").from_list(indices_b)
        intersection = E & F

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "E intersect F"

    def test_intersection_with_self(self, Omega, F):
        """Test intersection of an Event with itself."""
        A = Event(sig_alg=F).from_list([0, 1])
        intersection = A & A

        assert intersection == A

    def test_difference_basic(self, Omega, F):
        """Test basic difference of two Events."""
        indices_a = [0, 1, 2]
        indices_b = [1, 2]
        expected_difference = [0]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_disjoint(self, Omega, F):
        """Test difference of disjoint Events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_difference = [0, 1]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_with_empty(self, Omega, F):
        """Test difference with empty event."""
        indices_a = [0, 1]
        indices_b = []
        expected_difference = [0, 1]
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_identical(self, Omega, F):
        """Test difference of identical Events."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        expected_difference = []
        A = Event(sig_alg=F, name="A").from_list(indices_a)
        B = Event(sig_alg=F, name="B").from_list(indices_b)
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_using_minus_basic(self, Omega, F):
        """Test difference using - operator with basic case."""
        indices_a = [0, 1, 2]
        indices_b = [2]
        expected_difference = [0, 1]
        G = Event(sig_alg=F, name="G").from_list(indices_a)
        H = Event(sig_alg=F, name="H").from_list(indices_b)
        difference = G - H

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "G difference H"

    def test_difference_using_minus_disjoint(self, Omega, F):
        """Test difference using - operator with disjoint events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_difference = [0, 1]
        G = Event(sig_alg=F, name="G").from_list(indices_a)
        H = Event(sig_alg=F, name="H").from_list(indices_b)
        difference = G - H

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "G difference H"

    def test_difference_with_self(self, Omega, F):
        """Test difference of an Event with itself."""
        A = Event(sig_alg=F).from_list([0, 1])
        difference = A - A

        assert len(difference) == 0


class TestSubsetSuperset:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    def test_subset_proper_subset(self, Omega, F):
        """Test proper subset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 1, 2]
        is_subset = True
        is_proper_subset = True
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_equal_subset(self, Omega, F):
        """Test equal subset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        is_subset = True
        is_proper_subset = False
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_not_subset(self, Omega, F):
        """Test not subset relationship."""
        indices_a = [0, 2]
        indices_b = [0, 1]
        is_subset = False
        is_proper_subset = False
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_empty_subset(self, Omega, F):
        """Test empty subset relationship."""
        indices_a = []
        indices_b = [0, 1]
        is_subset = True
        is_proper_subset = True
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_subset_subset_of_full(self, Omega, F):
        """Test subset of full sample space."""
        indices_a = [0]
        indices_b = [0, 1, 2, 3]
        is_subset = True
        is_proper_subset = True
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A <= B) == is_subset
        assert (A < B) == is_proper_subset

    def test_superset_proper_superset(self, Omega, F):
        """Test proper superset relationship."""
        indices_a = [0, 1, 2]
        indices_b = [0, 1]
        is_superset = True
        is_proper_superset = True
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_equal_superset(self, Omega, F):
        """Test equal superset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        is_superset = True
        is_proper_superset = False
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_not_superset(self, Omega, F):
        """Test not superset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 2]
        is_superset = False
        is_proper_superset = False
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_full_superset(self, Omega, F):
        """Test full superset relationship."""
        indices_a = [0, 1, 2, 3]
        indices_b = [0]
        is_superset = True
        is_proper_superset = True
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset

    def test_superset_superset_of_empty(self, Omega, F):
        """Test superset of empty event."""
        indices_a = [0, 1]
        indices_b = []
        is_superset = True
        is_proper_superset = True
        A = Event(sig_alg=F).from_list(indices_a)
        B = Event(sig_alg=F).from_list(indices_b)

        assert (A >= B) == is_superset
        assert (A > B) == is_proper_superset


class TestEquality:
    def test_non_equality_different_indices(self):
        """Test inequality with different indices."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        A1 = [0, 1]
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        A2 = [0, 2]
        event1 = Event(sig_alg=SigmaAlgebra.power_set(Omega1)).from_list(A1)
        event2 = Event(sig_alg=SigmaAlgebra.power_set(Omega2)).from_list(A2)

        assert event1 != event2

    def test_non_equality_different_sample_spaces(self):
        """Test inequality with different sample spaces."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=2)
        A1 = [0]
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_list(["a", "b"])
        A2 = ["a"]
        event1 = Event(sig_alg=SigmaAlgebra.power_set(Omega1)).from_list(A1)
        event2 = Event(sig_alg=SigmaAlgebra.power_set(Omega2)).from_list(A2)

        assert event1 != event2

    def test_equality_different_names(self):
        """Test equality with different names."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        A1 = [0, 1]
        name1 = "A"
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        A2 = [0, 1]
        name2 = "B"
        event1 = Event(sig_alg=SigmaAlgebra.power_set(Omega1), name=name1).from_list(A1)
        event2 = Event(sig_alg=SigmaAlgebra.power_set(Omega2), name=name2).from_list(A2)

        assert event1 == event2

    def test_equality_all_attributes_match(self):
        """Test equality with all attributes matching."""
        Omega1 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        A1 = [0, 1]
        name1 = "A"
        Omega2 = SampleSpace(name="Omega", data_name="sample").from_sequence(size=3)
        A2 = [0, 1]
        name2 = "B"
        event1 = Event(sig_alg=SigmaAlgebra.power_set(Omega1), name=name1).from_list(A1)
        event2 = Event(sig_alg=SigmaAlgebra.power_set(Omega2), name=name2).from_list(A2)

        assert event1 == event2
