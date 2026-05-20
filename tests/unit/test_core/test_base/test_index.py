import pandas as pd
import pytest

from sigalg.core import Index

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test constructor with no parameters."""
        I = Index()

        assert I.name == "I"
        assert I.data_name is None
        assert I.indices is None
        assert I.dimension is None
        assert I.data is None

    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        index = Index(name="index")

        assert index.name == "index"
        assert index.data_name is None
        assert index.indices is None
        assert index.dimension is None
        assert index.data is None


class TestFromList:
    def test_from_list_single_dim_with_no_names(self):
        """Test from_list with no names provided and single dimension."""
        I = Index().from_list(["a", "b", "c"])
        expected_data = pd.Index(["a", "b", "c"], name="index")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.indices == ["a", "b", "c"]
        assert I.name == "I"
        assert I.data_name == ["index"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_list_multi_dim_with_no_names(self):
        """Test from_list with no names provided and multiple dimensions."""
        I = Index().from_list([("a", 1), ("b", 2), ("c", 3)])
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["index_0", "index_1"]
        )

        assert isinstance(I.data, pd.Index)
        assert isinstance(I.data, pd.MultiIndex)
        assert I.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert I.name == "I"
        assert I.data_name == ["index_0", "index_1"]
        assert I.dimension == 2
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_list_single_dim_with_name(self):
        """Test from_list with name provided and single dimension."""
        index = Index(name="index").from_list([1, 2, 3], data_name=["data_name"])
        expected_data = pd.Index([1, 2, 3], name="data_name")

        assert isinstance(index.data, pd.Index)
        assert not isinstance(index.data, pd.MultiIndex)
        assert index.indices == [1, 2, 3]
        assert index.name == "index"
        assert index.data_name == ["data_name"]
        assert index.dimension == 1
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_from_list_multi_dim_with_name(self):
        """Test from_list with name provided and multiple dimensions."""
        J = Index(name="J").from_list(
            [("a", 1), ("b", 2), ("c", 3)], data_name=["data_name_0", "data_name_1"]
        )
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["data_name_0", "data_name_1"]
        )

        assert isinstance(J.data, pd.Index)
        assert isinstance(J.data, pd.MultiIndex)
        assert J.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert J.name == "J"
        assert J.data_name == ["data_name_0", "data_name_1"]
        assert J.dimension == 2
        pd.testing.assert_index_equal(J.data, expected_data)

    def test_tuples_not_all_same_length_raises(self):
        """Test that tuples of different lengths raise ValueError."""
        with pytest.raises(
            ValueError,
            match="All items in 'indices' must be tuples of the same length.",
        ):
            Index().from_list([("a", 1), ("b", 2, "extra"), ("c", 3)])

    def test_tuple_length_mismatch_with_data_name_raises(self):
        """Test that tuple length mismatch with data_name raises ValueError."""
        with pytest.raises(
            ValueError,
            match="If 'indices' is a list of tuples, 'data_name' must have the same length as the tuples.",
        ):
            Index().from_list([("a", 1), ("b", 2), ("c", 3)], data_name=["data_name_0"])

    def test_list_of_non_tuples_with_multiple_data_names_raises(self):
        """Test that list of non-tuples with multiple data names raises ValueError."""
        with pytest.raises(
            ValueError,
            match="If 'indices' is a list of non-tuples, 'data_name' must have length 1.",
        ):
            Index().from_list([1, 2, 3], data_name=["data_name_0", "data_name_1"])

    def test_invalid_indices_not_list_raises(self):
        """Test that non-list indices raise TypeError."""
        with pytest.raises(TypeError):
            Index().from_list("not_a_list")

    def test_invalid_duplicate_elements_raises(self):
        """Test that duplicate elements in indices raise ValueError."""
        with pytest.raises(ValueError):
            Index().from_list(["a", "b", "a"])


class TestFromPandas:
    def test_from_pandas_single_dim_with_names(self):
        """Test from_pandas with names and a single dimension."""
        indices = ["a", "b", "c"]
        data = pd.Index(indices, name="data_name")
        J = Index(name="J").from_pandas(data=data)

        assert isinstance(J.data, pd.Index)
        assert not isinstance(J.data, pd.MultiIndex)
        assert J.indices == ["a", "b", "c"]
        assert J.name == "J"
        assert J.data_name == ["data_name"]
        assert J.dimension == 1
        pd.testing.assert_index_equal(J.data, data)

    def test_from_pandas_multi_dim_with_names(self):
        """Test from_pandas with names and multiple dimensions."""
        indices = [("a", 1), ("b", 2), ("c", 3)]
        data = pd.MultiIndex.from_tuples(indices, names=["data_name_0", "data_name_1"])
        J = Index(name="J").from_pandas(data=data)

        assert isinstance(J.data, pd.Index)
        assert isinstance(J.data, pd.MultiIndex)
        assert J.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert J.name == "J"
        assert J.data_name == ["data_name_0", "data_name_1"]
        assert J.dimension == 2
        pd.testing.assert_index_equal(J.data, data)

    def test_from_pandas_single_dim_with_no_names(self):
        """Test from_pandas with no names and a single dimension."""
        indices = ["a", "b", "c"]
        data = pd.Index(indices)
        I = Index().from_pandas(data)

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.indices == ["a", "b", "c"]
        assert I.name == "I"
        assert I.data_name == [None]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, data)

    def test_from_pandas_multi_dim_with_no_names(self):
        """Test from_pandas with no names and multiple dimensions."""
        indices = [("a", 1), ("b", 2), ("c", 3)]
        data = pd.MultiIndex.from_tuples(indices)
        I = Index().from_pandas(data)

        assert isinstance(I.data, pd.Index)
        assert isinstance(I.data, pd.MultiIndex)
        assert I.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert I.name == "I"
        assert I.data_name == [None, None]
        assert I.dimension == 2
        pd.testing.assert_index_equal(I.data, data)

    def test_invalid_inputs_raise(self):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(TypeError):
            Index().from_pandas(["not", "a", "pandas", "Index"])


class TestFromSequence:
    def test_from_sequence_with_default_parameters(self):
        """Test from_sequence with default parameters."""
        I = Index().from_sequence(size=3)
        expected_indices = [0, 1, 2]
        expected_data = pd.Index(expected_indices, name="index")

        assert I.indices == expected_indices
        assert I.name == "I"
        assert I.data_name == ["index"]
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_sequence_with_custom_initial_index_and_data_name(self):
        """Test from_sequence with custom initial index and data name."""
        I = Index().from_sequence(size=3, initial_index=1, data_name=["numbers"])
        expected_indices = [1, 2, 3]
        expected_data = pd.Index(expected_indices, name="numbers")

        assert I.indices == expected_indices
        assert I.name == "I"
        assert I.data_name == ["numbers"]
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_sequence_with_custom_prefix_and_initial_index(self):
        """Test from_sequence with custom prefix and initial index."""
        J = Index(name="J").from_sequence(size=3, prefix="item", initial_index=1)
        expected_indices = ["item_1", "item_2", "item_3"]
        expected_data = pd.Index(expected_indices, name="index")

        assert J.indices == expected_indices
        assert J.name == "J"
        assert J.data_name == ["index"]
        pd.testing.assert_index_equal(J.data, expected_data)

    def test_invalid_size_raises(self):
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            Index().from_sequence(size=-1)

    def test_invalid_initial_index_raises(self):
        """Test that invalid initial index raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            Index().from_sequence(size=3, initial_index="not_an_integer")

    def test_invalid_prefix_raises(self):
        """Test that invalid prefix raises TypeError."""
        with pytest.raises(TypeError, match="must be hashable"):
            Index().from_sequence(size=3, prefix=[])

    def test_data_name_not_a_list_raises(self):
        """Test that data_name not being a list raises TypeError."""
        with pytest.raises(TypeError, match="'data_name' must be a list"):
            Index().from_sequence(size=3, data_name="not_a_list")

    def test_data_name_list_with_more_than_one_element_raises(self):
        """Test that data_name list with more than one element raises ValueError."""
        with pytest.raises(ValueError, match="must be a list with a single element"):
            Index().from_sequence(size=3, data_name=["name1", "name2"])


# --------------------- test data access --------------------- #


class TestGetItem:
    @pytest.fixture
    def index(self):
        return Index(name="my_index", data_name="my_data").from_list(
            ["a", "b", "c", "d", "e"]
        )

    def test_get_item_with_integer(self, index):
        """Test __getitem__ with integer index."""
        pos = 0
        result = index[pos]
        assert result == index.indices[pos]

    def test_get_item_with_slice(self, index):
        """Test __getitem__ with slice index."""
        pos = slice(1, 4)
        result = index[pos]
        expected_indices = index.indices[pos]
        expected_data = pd.Index(expected_indices, name=index.data.name)

        assert isinstance(result, Index)
        assert result.indices == expected_indices
        assert result.name == index.name
        assert result.data.name == index.data.name
        pd.testing.assert_index_equal(result.data, expected_data)

    def test_get_item_with_list(self, index):
        """Test __getitem__ with list of indices."""
        pos = [0, 2, 4]
        result = index[pos]
        expected_indices = [index.indices[i] for i in pos]
        expected_data = pd.Index(expected_indices, name=index.data.name)

        assert isinstance(result, Index)
        assert result.indices == expected_indices
        assert result.name == index.name
        assert result.data.name == index.data.name
        pd.testing.assert_index_equal(result.data, expected_data)

    def test_invalid_out_of_bounds_integer_raises(self, index):
        """Test that out of bounds integer index raises IndexError."""
        with pytest.raises(IndexError):
            index[10]

    def test_invalid_out_of_bounds_list_raises(self, index):
        """Test that out of bounds list index raises IndexError."""
        with pytest.raises(IndexError):
            index[[0, 5]]

    def test_invalid_type_string_raises(self, index):
        """Test that invalid type (string) raises TypeError."""
        with pytest.raises(TypeError):
            index["invalid_type"]

    def test_invalid_list_contents_raises(self, index):
        """Test that list with invalid contents raises TypeError."""
        with pytest.raises(TypeError):
            index[["a", "b"]]


class TestContains:
    def test_contains(self):
        """Test the __contains__ method."""
        indices = ["a", "b", "c"]
        index = Index().from_list(indices)

        assert "a" in index
        assert "b" in index
        assert "c" in index
        assert "d" not in index


# --------------------- test sequence methods --------------------- #


class TestLength:
    def test_length(self):
        """Test the __len__ method."""
        indices = ["a", "b", "c", "d"]
        index = Index().from_list(indices)

        assert len(index) == 4


class TestIter:
    def test_iteration(self):
        """Test the __iter__ method."""
        indices = ["a", "b", "c"]
        index = Index().from_list(indices)
        iterated_indices = list(index)

        assert iterated_indices == indices


# --------------------- test equality --------------------- #


class TestEquality:
    def test_non_equality_different_order(self):
        """Test inequality when indices are in different order."""
        given = Index().from_list(["a", "b"])
        other = Index().from_list(["b", "a"])
        assert given != other

    def test_non_equality_different_length(self):
        """Test inequality when indices have different lengths."""
        given = Index().from_list(["a", "b"])
        other = Index().from_list(["a", "b", "c"])
        assert given != other

    def test_non_equality_wrong_type(self):
        """Test inequality when comparing to wrong type."""
        given = Index(data_name="index1").from_list(["a", "b"])
        other = "not_an_index"
        assert given != other

    def test_equality_same_indices(self):
        """Test equality when indices are the same."""
        given = Index(name="index", data_name="data").from_list(["a", "b", "c"])
        other = Index(name="index", data_name="data").from_list(["a", "b", "c"])
        assert given == other

    def test_equality_same_indices_different_names(self):
        """Test equality when indices are same but names differ."""
        given = Index(name="index1", data_name="data1").from_list(["a", "b", "c"])
        other = Index(name="index2", data_name="data2").from_list(["a", "b", "c"])
        assert given == other
