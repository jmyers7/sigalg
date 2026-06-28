import pandas as pd
import pytest

from sigalg.core import Event, RandomVariable, SampleSpace, SigmaAlgebra

# --------------------- test constructors --------------------- #


class TestConstructor:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 2,
            },
        )

    def test_constructor_no_parameters(self):
        """Test constructor with no parameters."""
        A = Event()

        assert A.name == "A"
        assert A.variable_names is None
        assert A.data is None
        assert A.sig_alg is None
        assert A.sample_space is None
        assert A.indicator is None

    def test_from_list_single_atom(self, Omega, F):
        """Test from list with indices from a single atom."""
        A = Event.from_list([0, 1], sig_alg=F)
        expected_data = pd.Index(data=[0, 1], name=Omega.variable_names[0])
        expected_indicator = RandomVariable(
            sample_space=Omega,
            sig_alg=F,
            name="I_A",
            mapping={
                0: 1,
                1: 1,
                2: 0,
                3: 0,
            },
        )

        assert A.name == "A"
        assert A.variable_names == ["sample"]
        pd.testing.assert_index_equal(A.data, expected_data)
        assert A.sig_alg is F
        assert A.sample_space is Omega
        assert A.indicator == expected_indicator
        assert A.indicator.name == "I_A"

    def test_from_list_union_of_two_atoms(self, Omega, F):
        """Test from list with indices from a union of two atoms."""
        B = Event.from_list([0, 1, 2], sig_alg=F, name="B")
        expected_data = pd.Index(data=[0, 1, 2], name=Omega.variable_names[0])
        expected_indicator = RandomVariable(
            sample_space=Omega,
            sig_alg=F,
            name="I_B",
            mapping={
                0: 1,
                1: 1,
                2: 1,
                3: 0,
            },
        )

        assert B.name == "B"
        assert B.variable_names == Omega.variable_names
        pd.testing.assert_index_equal(B.data, expected_data)
        assert B.sig_alg is F
        assert B.sample_space is Omega
        assert B.indicator == expected_indicator
        assert B.indicator.name == "I_B"

    def test_from_list_empty_set(self, Omega, F):
        """Test from list with empty set of indices."""
        empty = Event.from_list([], sig_alg=F, name="empty")
        expected_data = pd.Index(data=[], name=Omega.variable_names[0])
        expected_indicator = RandomVariable(
            sample_space=Omega,
            sig_alg=F,
            name="I_empty",
            mapping={
                0: 0,
                1: 0,
                2: 0,
                3: 0,
            },
        )

        assert empty.name == "empty"
        assert empty.variable_names == Omega.variable_names
        pd.testing.assert_index_equal(empty.data, expected_data)
        assert empty.sig_alg is F
        assert empty.sample_space is Omega
        assert empty.indicator == expected_indicator
        assert empty.indicator.name == "I_empty"

    def test_from_list_all_sample_points(self, Omega, F):
        """Test from list with all sample points."""
        full = Event.from_list([0, 1, 2, 3], sig_alg=F, name="full")
        expected_data = pd.Index(data=[0, 1, 2, 3], name=Omega.variable_names[0])
        expected_indicator = RandomVariable(
            sample_space=Omega,
            sig_alg=F,
            name="I_full",
            mapping={
                0: 1,
                1: 1,
                2: 1,
                3: 1,
            },
        )

        assert full.name == "full"
        assert full.variable_names == Omega.variable_names
        pd.testing.assert_index_equal(full.data, expected_data)
        assert full.sig_alg is F
        assert full.sample_space is Omega
        assert full.indicator == expected_indicator
        assert full.indicator.name == "I_full"

    def test_from_list_singleton(self, Omega, F):
        """Test from list with a single index."""
        singleton = Event.from_list([2], sig_alg=F, name="singleton")
        expected_data = pd.Index(data=[2], name=Omega.variable_names[0])
        expected_indicator = RandomVariable(
            sample_space=Omega,
            sig_alg=F,
            name="I_singleton",
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 0,
            },
        )

        assert singleton.name == "singleton"
        assert singleton.variable_names == Omega.variable_names
        pd.testing.assert_index_equal(singleton.data, expected_data)
        assert singleton.sig_alg is F
        assert singleton.sample_space is Omega
        assert singleton.indicator == expected_indicator
        assert singleton.indicator.name == "I_singleton"

    def test_indices_not_list_raises(self, F):
        """Test that non-list indices raise exception."""
        with pytest.raises(TypeError, match="The indices must be a list"):
            Event.from_list("not a list", sig_alg=F)

    def test_event_not_subset_of_sample_space_raises(self, F):
        """Test that indices not subset of sample space raise exception."""
        with pytest.raises(ValueError, match="not a subset of the sample space"):
            Event.from_list([5], sig_alg=F)

    def test_non_measurable_subset_raises(self, F):
        """Test that non-measurable subset raises exception."""
        with pytest.raises(ValueError, match="The event is not measurable"):
            Event.from_list([0, 2], sig_alg=F)


# --------------------- test properties --------------------- #

# --------------------- test data access --------------------- #


class TestGetItem:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def A(self, F):
        return Event.from_list([0, 1, 2], sig_alg=F)

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


# --------------------- test set-theoretic operations --------------------- #


class TestSetTheoreticOperations:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    def test_complement_basic(self, Omega, F):
        """Test basic complement of an Event."""
        indices = [0, 1]
        expected_complement = [2, 3]
        A = Event.from_list(indices, sig_alg=F)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_of_empty(self, Omega, F):
        """Test complement of empty event."""
        indices = []
        expected_complement = [0, 1, 2, 3]
        A = Event.from_list(indices, sig_alg=F)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_of_full(self, Omega, F):
        """Test complement of full sample space."""
        indices = [0, 1, 2, 3]
        expected_complement = []
        A = Event.from_list(indices, sig_alg=F)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_single(self, Omega, F):
        """Test complement of single element event."""
        indices = [0]
        expected_complement = [1, 2, 3]
        A = Event.from_list(indices, sig_alg=F)
        comp = A.complement()

        assert isinstance(comp, Event)
        assert set(comp.data) == set(expected_complement)
        assert comp.name == "A complement"

    def test_complement_using_tilde_basic(self, Omega, F):
        """Test basic complement using ~ operator."""
        indices = [0, 1]
        expected_complement = [2, 3]
        B = Event.from_list(indices, sig_alg=F, name="B")
        comp = ~B

        assert set(comp.data) == set(expected_complement)
        assert comp.name == "B complement"

    def test_complement_using_tilde_single(self, Omega, F):
        """Test complement of single element using ~ operator."""
        indices = [0]
        expected_complement = [1, 2, 3]
        B = Event.from_list(indices, sig_alg=F, name="B")
        comp = ~B

        assert set(comp.data) == set(expected_complement)
        assert comp.name == "B complement"

    def test_double_complement(self, Omega, F):
        """Test that double complement returns the original Event."""
        A = Event.from_list([0, 1], sig_alg=F)
        double_comp = ~~A

        assert double_comp == A

    def test_union_disjoint(self, Omega, F):
        """Test union of disjoint Events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_union = [0, 1, 2, 3]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_overlapping(self, Omega, F):
        """Test union of overlapping Events."""
        indices_a = [0, 1]
        indices_b = [1, 2]
        expected_union = [0, 1, 2]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_with_empty(self, Omega, F):
        """Test union with empty event."""
        indices_a = [0, 1]
        indices_b = []
        expected_union = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_identical(self, Omega, F):
        """Test union of identical Events."""
        indices_a = [0]
        indices_b = [0]
        expected_union = [0]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        union = A.union(B)

        assert set(union.data) == set(expected_union)
        assert union.name == "A union B"

    def test_union_using_pipe_disjoint(self, Omega, F):
        """Test union using | operator with disjoint events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_union = [0, 1, 2, 3]
        C = Event.from_list(indices_a, sig_alg=F, name="C")
        D = Event.from_list(indices_b, sig_alg=F, name="D")
        union = C | D

        assert set(union.data) == set(expected_union)
        assert union.name == "C union D"

    def test_union_using_pipe_overlapping(self, Omega, F):
        """Test union using | operator with overlapping events."""
        indices_a = [0, 1]
        indices_b = [1, 2]
        expected_union = [0, 1, 2]
        C = Event.from_list(indices_a, sig_alg=F, name="C")
        D = Event.from_list(indices_b, sig_alg=F, name="D")
        union = C | D

        assert set(union.data) == set(expected_union)
        assert union.name == "C union D"

    def test_union_with_self(self, Omega, F):
        """Test union of an Event with itself."""
        A = Event.from_list([0, 1], sig_alg=F)
        union = A | A

        assert union == A

    def test_intersection_overlapping(self, Omega, F):
        """Test intersection of overlapping Events."""
        indices_a = [0, 1, 2]
        indices_b = [1, 2, 3]
        expected_intersection = [1, 2]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_disjoint(self, Omega, F):
        """Test intersection of disjoint Events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_intersection = []
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_with_empty(self, Omega, F):
        """Test intersection with empty event."""
        indices_a = [0, 1]
        indices_b = []
        expected_intersection = []
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_identical(self, Omega, F):
        """Test intersection of identical Events."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        expected_intersection = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        intersection = A.intersection(B)

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "A intersect B"

    def test_intersection_using_ampersand_overlapping(self, Omega, F):
        """Test intersection using & operator with overlapping events."""
        indices_a = [0, 1]
        indices_b = [1, 2]
        expected_intersection = [1]
        E = Event.from_list(indices_a, sig_alg=F, name="E")
        F = Event.from_list(indices_b, sig_alg=F, name="F")
        intersection = E & F

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "E intersect F"

    def test_intersection_using_ampersand_disjoint(self, Omega, F):
        """Test intersection using & operator with disjoint events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_intersection = []
        E = Event.from_list(indices_a, sig_alg=F, name="E")
        F = Event.from_list(indices_b, sig_alg=F, name="F")
        intersection = E & F

        assert set(intersection.data) == set(expected_intersection)
        assert intersection.name == "E intersect F"

    def test_intersection_with_self(self, Omega, F):
        """Test intersection of an Event with itself."""
        A = Event.from_list([0, 1], sig_alg=F)
        intersection = A & A

        assert intersection == A

    def test_difference_basic(self, Omega, F):
        """Test basic difference of two Events."""
        indices_a = [0, 1, 2]
        indices_b = [1, 2]
        expected_difference = [0]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_disjoint(self, Omega, F):
        """Test difference of disjoint Events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_difference = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_with_empty(self, Omega, F):
        """Test difference with empty event."""
        indices_a = [0, 1]
        indices_b = []
        expected_difference = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_identical(self, Omega, F):
        """Test difference of identical Events."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        expected_difference = []
        A = Event.from_list(indices_a, sig_alg=F, name="A")
        B = Event.from_list(indices_b, sig_alg=F, name="B")
        difference = A.difference(B)

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "A difference B"

    def test_difference_using_minus_basic(self, Omega, F):
        """Test difference using - operator with basic case."""
        indices_a = [0, 1, 2]
        indices_b = [2]
        expected_difference = [0, 1]
        G = Event.from_list(indices_a, sig_alg=F, name="G")
        H = Event.from_list(indices_b, sig_alg=F, name="H")
        difference = G - H

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "G difference H"

    def test_difference_using_minus_disjoint(self, Omega, F):
        """Test difference using - operator with disjoint events."""
        indices_a = [0, 1]
        indices_b = [2, 3]
        expected_difference = [0, 1]
        G = Event.from_list(indices_a, sig_alg=F, name="G")
        H = Event.from_list(indices_b, sig_alg=F, name="H")
        difference = G - H

        assert set(difference.data) == set(expected_difference)
        assert difference.name == "G difference H"

    def test_difference_with_self(self, Omega, F):
        """Test difference of an Event with itself."""
        A = Event.from_list([0, 1], sig_alg=F)
        difference = A - A

        assert len(difference) == 0


# --------------------- test sub/superset methods --------------------- #


class TestSubsetSuperset:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    def test_subset_proper_subset(self, Omega, F):
        """Test proper subset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 1, 2]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A <= B
        assert A < B

    def test_subset_equal_subset(self, Omega, F):
        """Test equal subset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A <= B
        assert not (A < B)

    def test_subset_not_subset(self, Omega, F):
        """Test not subset relationship."""
        indices_a = [0, 2]
        indices_b = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert not (A <= B)
        assert not (A < B)

    def test_subset_empty_subset(self, Omega, F):
        """Test empty subset relationship."""
        indices_a = []
        indices_b = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A <= B
        assert A < B

    def test_subset_subset_of_full(self, Omega, F):
        """Test subset of full sample space."""
        indices_a = [0]
        indices_b = [0, 1, 2, 3]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A <= B
        assert A < B

    def test_superset_proper_superset(self, Omega, F):
        """Test proper superset relationship."""
        indices_a = [0, 1, 2]
        indices_b = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A >= B
        assert A > B

    def test_superset_equal_superset(self, Omega, F):
        """Test equal superset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 1]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A >= B
        assert not (A > B)

    def test_superset_not_superset(self, Omega, F):
        """Test not superset relationship."""
        indices_a = [0, 1]
        indices_b = [0, 2]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert not (A >= B)
        assert not (A > B)

    def test_superset_full_superset(self, Omega, F):
        """Test full superset relationship."""
        indices_a = [0, 1, 2, 3]
        indices_b = [0]
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A >= B
        assert A > B

    def test_superset_superset_of_empty(self, Omega, F):
        """Test superset of empty event."""
        indices_a = [0, 1]
        indices_b = []
        A = Event.from_list(indices_a, sig_alg=F)
        B = Event.from_list(indices_b, sig_alg=F)

        assert A >= B
        assert A > B


# --------------------- test equality --------------------- #


class TestEquality:
    def test_non_equality_different_indices(self):
        """Test inequality with different indices."""
        Omega1 = SampleSpace.from_sequence(size=3)
        Omega2 = SampleSpace.from_sequence(size=3)
        event1 = Event.from_list([0, 1], sig_alg=SigmaAlgebra.power_set(Omega1))
        event2 = Event.from_list([0, 2], sig_alg=SigmaAlgebra.power_set(Omega2))

        assert event1 != event2

    def test_non_equality_different_sample_spaces(self):
        """Test inequality with different sample spaces."""
        Omega1 = SampleSpace.from_sequence(size=2)
        Omega2 = SampleSpace(["a", "b"])
        event1 = Event.from_list([0], sig_alg=SigmaAlgebra.power_set(Omega1))
        event2 = Event.from_list(["a"], sig_alg=SigmaAlgebra.power_set(Omega2))

        assert event1 != event2

    def test_equality_different_names(self):
        """Test equality with different names."""
        Omega1 = SampleSpace().from_sequence(size=3)
        Omega2 = SampleSpace().from_sequence(size=3)
        event1 = Event.from_list(
            [0, 1], sig_alg=SigmaAlgebra.power_set(Omega1), name="A"
        )
        event2 = Event.from_list(
            [0, 1], sig_alg=SigmaAlgebra.power_set(Omega2), name="B"
        )

        assert event1 == event2

    def test_equality_all_attributes_match(self):
        """Test equality with all attributes matching."""
        Omega1 = SampleSpace().from_sequence(size=3)
        Omega2 = SampleSpace().from_sequence(size=3)
        event1 = Event.from_list(
            [0, 1], sig_alg=SigmaAlgebra.power_set(Omega1), name="A"
        )
        event2 = Event.from_list(
            [0, 1], sig_alg=SigmaAlgebra.power_set(Omega2), name="B"
        )

        assert event1 == event2
