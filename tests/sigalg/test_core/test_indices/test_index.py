import pandas as pd
import pytest

from sigalg.core import Index

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_with_default_parameters(self):
        """Test constructor with no parameters."""
        I = Index()

        assert I.name == "I"
        assert I.variable_names is None
        assert I.dimension is None
        assert I.data is None

    def test_single_dim_constructor_with_list_and_default_names(self):
        """Test constructor with single dimension, list of indices, and default names."""
        I = Index(indices=["a", "b", "c"])
        expected_data = pd.Index(["a", "b", "c"], name="index")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["index"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_multi_dim_constructor_with_list_and_default_names(self):
        """Test constructor with multiple dimensions, list of indices, and default names."""
        J = Index(name="J", indices=[("a", 1), ("b", 2), ("c", 3)])
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["index_0", "index_1"]
        )

        assert isinstance(J.data, pd.Index)
        assert isinstance(J.data, pd.MultiIndex)
        assert J.name == "J"
        assert J.variable_names == ["index_0", "index_1"]
        assert J.dimension == 2
        pd.testing.assert_index_equal(J.data, expected_data)

    def test_single_dim_constructor_with_list_and_custom_names(self):
        """Test constructor with single dimension, list of indices, and custom names."""
        I = Index(indices=["a", "b", "c"], variable_names=["custom_name"])
        expected_data = pd.Index(["a", "b", "c"], name="custom_name")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["custom_name"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_multi_dim_constructor_with_list_and_custom_names(self):
        """Test constructor with multiple dimensions, list of indices, and custom names."""
        I = Index(
            indices=[("a", 1), ("b", 2), ("c", 3)],
            variable_names=["custom_name_0", "custom_name_1"],
        )
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["custom_name_0", "custom_name_1"]
        )

        assert isinstance(I.data, pd.Index)
        assert isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["custom_name_0", "custom_name_1"]
        assert I.dimension == 2
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_empty_indices_with_default_data_name(self):
        """Test constructor with empty indices and default data_name."""
        J = Index(name="J", indices=[])
        expected_data = pd.Index([], name="index")

        assert isinstance(J.data, pd.Index)
        assert not isinstance(J.data, pd.MultiIndex)
        assert J.name == "J"
        assert J.variable_names == ["index"]
        assert J.dimension == 1
        pd.testing.assert_index_equal(J.data, expected_data)

    def test_empty_indices_with_custom_data_name(self):
        """Test constructor with empty indices and custom data_name."""
        I = Index(indices=[], variable_names=["custom_name"])
        expected_data = pd.Index([], name="custom_name")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["custom_name"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_single_dim_constructor_with_index_with_names(self):
        """Test constructor with single dimension and pd.Index with name."""
        indices = pd.Index(["a", "b", "c"], name="letter")
        J = Index(indices=indices, name="J")

        assert isinstance(J.data, pd.Index)
        assert not isinstance(J.data, pd.MultiIndex)
        assert J.name == "J"
        assert J.variable_names == ["letter"]
        assert J.dimension == 1
        pd.testing.assert_index_equal(J.data, indices)

    def test_multi_dim_constructor_with_index_with_names(self):
        """Test constructor with multiple dimensions and pd.MultiIndex with names."""
        indices = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["letter", "num"]
        )
        J = Index(indices=indices, name="J")

        assert isinstance(J.data, pd.Index)
        assert isinstance(J.data, pd.MultiIndex)
        assert J.name == "J"
        assert J.variable_names == ["letter", "num"]
        assert J.dimension == 2
        pd.testing.assert_index_equal(J.data, indices)

    def test_single_dim_constructor_with_index_with_no_names(self):
        """Test constructor with single dimension and pd.Index with no name."""
        indices = pd.Index(["a", "b", "c"])
        I = Index(indices=indices)
        expected_data = pd.Index(indices, name="index")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["index"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_multi_dim_constructor_with_index_with_no_names(self):
        """Test constructor with multiple dimensions and pd.MultiIndex with no names."""
        indices = pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)])
        I = Index(indices=indices)
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["index_0", "index_1"]
        )

        assert isinstance(I.data, pd.Index)
        assert isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["index_0", "index_1"]
        assert I.dimension == 2
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_single_dim_constructor_with_index_with_custom_variable_names(self):
        indices = pd.Index(["a", "b", "c"])
        I = Index(indices=indices, variable_names=["x"])
        expected_data = pd.Index(indices, name="x")

        assert isinstance(I.data, pd.Index)
        assert not isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["x"]
        assert I.dimension == 1
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_multi_dim_constructor_with_index_with_custom_variable_names(self):
        indices = pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)])
        I = Index(indices=indices, variable_names=["letter", "num"])
        expected_data = pd.MultiIndex.from_tuples(
            [("a", 1), ("b", 2), ("c", 3)], names=["letter", "num"]
        )

        assert isinstance(I.data, pd.Index)
        assert isinstance(I.data, pd.MultiIndex)
        assert I.name == "I"
        assert I.variable_names == ["letter", "num"]
        assert I.dimension == 2
        pd.testing.assert_index_equal(I.data, expected_data)


class TestFromSequence:
    def test_from_sequence_with_default_parameters(self):
        """Test from_sequence with default parameters."""
        I = Index.from_sequence(size=3)
        expected_indices = [0, 1, 2]
        expected_data = pd.Index(expected_indices, name="index")

        assert I.name == "I"
        assert I.variable_names == ["index"]
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_sequence_with_custom_initial_index_and_variable_name(self):
        """Test from_sequence with custom initial index and variable name."""
        I = Index.from_sequence(size=3, initial_index=1, variable_name="numbers")
        expected_indices = [1, 2, 3]
        expected_data = pd.Index(expected_indices, name="numbers")

        assert I.name == "I"
        assert I.variable_names == ["numbers"]
        pd.testing.assert_index_equal(I.data, expected_data)

    def test_from_sequence_with_custom_prefix_and_initial_index(self):
        """Test from_sequence with custom prefix and initial index."""
        J = Index.from_sequence(size=3, name="J", prefix="item", initial_index=1)
        expected_indices = ["item_1", "item_2", "item_3"]
        expected_data = pd.Index(expected_indices, name="index")

        assert J.name == "J"
        assert J.variable_names == ["index"]
        pd.testing.assert_index_equal(J.data, expected_data)

    def test_invalid_size_raises(self):
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            Index.from_sequence(size=-1)

    def test_invalid_initial_index_raises(self):
        """Test that invalid initial index raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            Index.from_sequence(size=3, initial_index="not_an_integer")

    def test_invalid_prefix_raises(self):
        """Test that invalid prefix raises TypeError."""
        with pytest.raises(TypeError, match="must be hashable"):
            Index.from_sequence(size=3, prefix=[])


# --------------------- test data access --------------------- #


class TestGetItem:
    @pytest.fixture
    def I(self):  # noqa: E743
        return Index(indices=["a", "b", "c", "d", "e"])

    def test_get_item_with_integer(self, I):
        """Test __getitem__ with integer index."""
        assert I[0] == "a"

    def test_get_item_with_slice(self, I):
        """Test __getitem__ with slice index."""
        result = I[1:4]
        expected_indices = ["b", "c", "d"]
        expected_data = pd.Index(expected_indices, name=I.data.name)

        assert isinstance(result, Index)
        assert result.name == I.name
        assert result.data.name == I.data.name
        pd.testing.assert_index_equal(result.data, expected_data)

    def test_get_item_with_list(self, I):
        """Test __getitem__ with list of indices."""
        result = I[[0, 2, 4]]
        expected_indices = ["a", "c", "e"]
        expected_data = pd.Index(expected_indices, name=I.data.name)

        assert isinstance(result, Index)
        assert result.name == I.name
        assert result.data.name == I.data.name
        pd.testing.assert_index_equal(result.data, expected_data)

    def test_invalid_out_of_bounds_integer_raises(self, I):
        """Test that out of bounds integer index raises IndexError."""
        with pytest.raises(IndexError):
            I[10]

    def test_invalid_out_of_bounds_list_raises(self, I):
        """Test that out of bounds list index raises IndexError."""
        with pytest.raises(IndexError):
            I[[0, 5]]

    def test_invalid_type_string_raises(self, I):
        """Test that invalid type (string) raises TypeError."""
        with pytest.raises(TypeError):
            I["invalid_type"]

    def test_invalid_list_contents_raises(self, I):
        """Test that list with invalid contents raises TypeError."""
        with pytest.raises(TypeError):
            I[["a", "b"]]


class TestContains:
    def test_contains(self):
        """Test the __contains__ method."""
        I = Index(indices=["a", "b", "c"])

        assert "a" in I
        assert "b" in I
        assert "c" in I
        assert "d" not in I


# --------------------- test sequence methods --------------------- #


class TestLength:
    def test_length(self):
        """Test the __len__ method."""
        index = Index(indices=["a", "b", "c", "d"])

        assert len(index) == 4


class TestIter:
    def test_iteration(self):
        """Test the __iter__ method."""
        index = Index(indices=["a", "b", "c"])
        iterated_indices = list(index)

        assert iterated_indices == ["a", "b", "c"]


# --------------------- test equality --------------------- #


class TestEquality:
    def test_non_equality_different_order(self):
        """Test inequality when indices are in different order."""
        given = Index(indices=["a", "b"])
        other = Index(indices=["b", "a"])
        assert given != other

    def test_non_equality_different_length(self):
        """Test inequality when indices have different lengths."""
        given = Index(indices=["a", "b"])
        other = Index(indices=["a", "b", "c"])
        assert given != other

    def test_equality_same_indices(self):
        """Test equality when indices are the same."""
        given = Index(indices=["a", "b", "c"], name="I", variable_names=["x"])
        other = Index(indices=["a", "b", "c"], name="J", variable_names=["x"])
        assert given == other
