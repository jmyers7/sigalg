import pandas as pd
import pytest

from sigalg.core import Index

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test constructor with no parameters."""
        I = Index()

        assert I.name == "I"
        assert I.variable_names is None
        assert I.indices is None
        assert I.dimension is None
        assert I.data is None

    def test_constructor_all_parameters(self):
        """Test constructor with all parameters provided."""
        index = Index(name="index")

        assert index.name == "index"
        assert index.variable_names is None
        assert index.indices is None
        assert index.dimension is None
        assert index.data is None


class TestFromList:
    def test_single_dim_default_names(self):
        """Test from_list with single dimension and default names."""
        I = Index().from_list(["a", "b", "c"])
        expected_data = pd.Index(["a", "b", "c"], name="I")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.indices == ["a", "b", "c"]
        assert I.name == "I"
        assert I.variable_names == ["I"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_multi_dim_default_names(self):
        """Test from_list with multiple dimensions and default names."""
        J = Index(name="J").from_list([("a", 1), ("b", 2), ("c", 3)])
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["J_0", "J_1"]
        )

        assert isinstance(J.data, pd.Index)
        assert isinstance(J.data, pd.MultiIndex)
        assert J.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert J.name == "J"
        assert J.variable_names == ["J_0", "J_1"]
        assert J.dimension == 2
        pd.testing.assert_index_equal(J.data, expected_data)

    def test_single_dim_custom_names(self):
        """Test from_list with single dimension and custom names."""
        I = Index().from_list(["a", "b", "c"], variable_names=["custom_name"])
        expected_data = pd.Index(["a", "b", "c"], name="custom_name")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.indices == ["a", "b", "c"]
        assert I.name == "I"
        assert I.variable_names == ["custom_name"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_multi_dim_custom_names(self):
        """Test from_list with multiple dimensions and custom names."""
        I = Index().from_list(
            [("a", 1), ("b", 2), ("c", 3)],
            variable_names=["custom_name_0", "custom_name_1"],
        )
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["custom_name_0", "custom_name_1"]
        )

        assert isinstance(I.data, pd.Index)
        assert isinstance(I.data, pd.MultiIndex)
        assert I.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert I.name == "I"
        assert I.variable_names == ["custom_name_0", "custom_name_1"]
        assert I.dimension == 2
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_multi_dim_custom_prefix_name(self):
        """Test from_list with multiple dimensions and a custom prefix name."""
        I = Index().from_list([("a", 1), ("b", 2), ("c", 3)], variable_names=["prefix"])
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["prefix_0", "prefix_1"]
        )

        assert isinstance(I.data, pd.Index)
        assert isinstance(I.data, pd.MultiIndex)
        assert I.indices == [("a", 1), ("b", 2), ("c", 3)]
        assert I.name == "I"
        assert I.variable_names == ["prefix_0", "prefix_1"]
        assert I.dimension == 2
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_empty_indices_with_default_data_name(self):
        """Test from_list with empty indices and default data_name."""
        J = Index(name="J").from_list([])
        expected_data = pd.Index([], name="J")

        assert isinstance(J.data, pd.Index)
        assert not isinstance(J.data, pd.MultiIndex)
        assert J.indices == []
        assert J.name == "J"
        assert J.variable_names == ["J"]
        assert J.dimension == 1
        pd.testing.assert_index_equal(J.data, expected_data)

    def test_empty_indices_with_custom_data_name(self):
        """Test from_list with empty indices and custom data_name."""
        I = Index().from_list([], variable_names=["custom_name"])
        expected_data = pd.Index([], name="custom_name")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.indices == []
        assert I.name == "I"
        assert I.variable_names == ["custom_name"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_empty_indices_with_invalid_variable_names_length_raises(self):
        """Test that empty indices with invalid variable_names length raises ValueError."""
        with pytest.raises(
            ValueError,
            match="If 'indices' is empty, 'variable_names' must have length 1.",
        ):
            Index().from_list([], variable_names=["name1", "name2"])

    def test_tuples_of_different_lengths_raises(self):
        """Test that tuples of different lengths raise ValueError."""
        with pytest.raises(
            ValueError,
            match="All items in 'indices' must be tuples of the same length.",
        ):
            Index().from_list([("a", 1), ("b", 2, "extra"), ("c", 3)])

    def test_tuple_length_mismatch_with_variable_names_raises(self):
        """Test that tuple length mismatch with variable_names raises ValueError."""
        with pytest.raises(
            ValueError,
            match="If 'indices' is a list of tuples, 'variable_names' must be None, have length 1, or must have length equal to the tuple length.",
        ):
            Index().from_list(
                [("a", 1), ("b", 2), ("c", 3)], variable_names=["x", "y", "z"]
            )

    def test_list_of_non_tuples_with_multiple_variable_names_raises(self):
        """Test that list of non-tuples with multiple variable names raises ValueError."""
        with pytest.raises(
            ValueError,
            match="If 'indices' is a list of non-tuples, 'variable_names' must be None or have length 1.",
        ):
            Index().from_list(
                [1, 2, 3], variable_names=["variable_name_0", "variable_name_1"]
            )


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
        assert J.variable_names == ["data_name"]
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
        assert J.variable_names == ["data_name_0", "data_name_1"]
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
        assert I.variable_names == [None]
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
        assert I.variable_names == [None, None]
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
        expected_data = pd.Index(expected_indices, name="I")

        assert I.indices == expected_indices
        assert I.name == "I"
        assert I.variable_names == ["I"]
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_sequence_with_custom_initial_index_and_data_name(self):
        """Test from_sequence with custom initial index and data name."""
        I = Index().from_sequence(size=3, initial_index=1, variable_name="numbers")
        expected_indices = [1, 2, 3]
        expected_data = pd.Index(expected_indices, name="numbers")

        assert I.indices == expected_indices
        assert I.name == "I"
        assert I.variable_names == ["numbers"]
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_sequence_with_custom_prefix_and_initial_index(self):
        """Test from_sequence with custom prefix and initial index."""
        J = Index(name="J").from_sequence(size=3, prefix="item", initial_index=1)
        expected_indices = ["item_1", "item_2", "item_3"]
        expected_data = pd.Index(expected_indices, name="J")

        assert J.indices == expected_indices
        assert J.name == "J"
        assert J.variable_names == ["J"]
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
