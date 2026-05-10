import pandas as pd
import pytest

from sigalg.core import Index

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test constructor with no parameters."""
        index = Index()

        assert index.name is None
        assert index.data_name is None
        assert index.indices is None
        assert index.data is None

    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        index = Index(name="index", data_name="data")

        assert index.name == "index"
        assert index.data_name == "data"
        assert index.indices is None
        assert index.data is None


class TestFromList:
    def test_from_list_with_no_names(self):
        """Test from_list with no names provided."""
        index = Index().from_list(["a", "b", "c"])
        expected_data = pd.Index(["a", "b", "c"])

        assert index.indices == ["a", "b", "c"]
        assert index.name is None
        assert index.data_name is None
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_from_list_with_name(self):
        """Test from_list with name provided."""
        index = Index(name="index", data_name="data_name").from_list([1, 2, 3])
        expected_data = pd.Index([1, 2, 3], name="data_name")

        assert index.indices == [1, 2, 3]
        assert index.name == "index"
        assert index.data_name == "data_name"
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_invalid_indices_not_list(self):
        """Test that non-list indices raise TypeError."""
        with pytest.raises(TypeError):
            Index().from_list("not_a_list")

    def test_invalid_indices_not_iterable(self):
        """Test that non-iterable indices raise TypeError."""
        with pytest.raises(TypeError):
            Index().from_list(123)

    def test_invalid_unhashable_elements(self):
        """Test that unhashable elements in indices raise TypeError."""
        with pytest.raises(TypeError):
            Index().from_list([{"a": 1}])

    def test_invalid_duplicate_elements(self):
        """Test that duplicate elements in indices raise ValueError."""
        with pytest.raises(ValueError):
            Index().from_list(["a", "b", "a"])


class TestFromPandas:
    @pytest.fixture
    def indices(self):
        return ["a", "b", "c"]

    def test_from_pandas_with_names(self, indices):
        """Test from_pandas with names."""
        data = pd.Index(indices, name="data_name")
        index = Index(name="index", data_name="data_name").from_pandas(
            data=data, overwrite_data_name=False
        )

        assert index.indices == indices
        assert index.name == "index"
        assert index.data_name == "data_name"
        pd.testing.assert_index_equal(index.data, data)

    def test_from_pandas_with_misaligned_data_names_raises(self, indices):
        """Test from_pandas with misaligned data names and overwrite_data_name=False."""
        data = pd.Index(indices, name="data_name")

        with pytest.raises(ValueError, match="does not match the current `data_name`"):
            Index(name="index", data_name="wrong_name").from_pandas(
                data=data, overwrite_data_name=False
            )

    def test_from_pandas_with_names_and_overwrite(self, indices):
        """Test from_pandas with names and overwrite_data_name=True."""
        data = pd.Index(indices, name="data_name")
        index = Index(name="index", data_name="new_name").from_pandas(
            data=data, overwrite_data_name=True
        )

        assert index.indices == indices
        assert index.name == "index"
        assert index.data_name == "data_name"
        pd.testing.assert_index_equal(index.data, data)

    def test_from_pandas_with_no_data_name(self, indices):
        """Test from_pandas with no data names."""
        data = pd.Index(indices)
        index = Index(name="index").from_pandas(data)

        assert index.indices == indices
        assert index.name == "index"
        assert index.data_name is None
        pd.testing.assert_index_equal(index.data, data)

    def test_invalid_inputs_raise(self):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(TypeError):
            Index().from_pandas(["not", "a", "pandas", "Index"])


class TestFromSequence:
    def test_from_sequence_with_default_parameters(self):
        """Test from_sequence with default parameters."""
        index = Index(name="index", data_name="data_name").from_sequence(size=3)
        expected_indices = [0, 1, 2]
        expected_data = pd.Index(expected_indices, name="data_name")

        assert index.indices == expected_indices
        assert index.name == "index"
        assert index.data_name == "data_name"
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_from_sequence_with_custom_initial_index(self):
        """Test from_sequence with custom initial index."""
        index = Index(name="index", data_name="data_name").from_sequence(
            size=3, initial_index=1
        )
        expected_indices = [1, 2, 3]
        expected_data = pd.Index(expected_indices, name="data_name")

        assert index.indices == expected_indices
        assert index.name == "index"
        assert index.data_name == "data_name"
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_from_sequence_with_custom_prefix_and_initial_index(self):
        """Test from_sequence with custom prefix and initial index."""
        index = Index(name="index", data_name="data_name").from_sequence(
            size=3, prefix="item", initial_index=1
        )
        expected_indices = ["item_1", "item_2", "item_3"]
        expected_data = pd.Index(expected_indices, name="data_name")

        assert index.indices == expected_indices
        assert index.name == "index"
        assert index.data_name == "data_name"
        pd.testing.assert_index_equal(index.data, expected_data)

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


# --------------------- test properties --------------------- #

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
