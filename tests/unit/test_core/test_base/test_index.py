import pandas as pd
import pytest

from sigalg.core import Index


class TestConstructor:

    def test_construction_from_all_parameters(self):
        """Test constructor with all parameters provided."""

        indices = ["a", "b", "c"]
        name = "my_index"
        data_name = "my_data"
        index = Index(
            indices=indices,
            name=name,
            data_name=data_name,
        )
        expected_data = pd.Index(indices, name=data_name)
        assert index.indices == indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_construction_minimal_parameters(self):
        """Test constructor with minimal parameters provided."""
        indices = [1, 2, 3]
        index = Index(indices=indices)
        expected_data = pd.Index(indices, name="data")
        assert index.indices == indices
        assert index.name == "index"
        assert index.data.name == "data"
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_construction_from_pandas_with_data_name(self):
        """Test constructor from a pd.Index."""
        data = pd.Index(["x", "y", "z"], name="pandas")
        name = "my_index"
        index = Index.from_pandas(data=data, name=name)
        assert index.indices == ["x", "y", "z"]
        assert index.name == name
        assert index.data.name == "pandas"
        pd.testing.assert_index_equal(index.data, data)

    def test_construction_from_pandas_without_data_name(self):
        """Test constructor from a pd.Index without providing name."""
        data = pd.Index(["x", "y", "z"])
        index = Index.from_pandas(data=data)
        assert index.indices == ["x", "y", "z"]
        assert index.name == "index"
        assert index.data.name is None
        pd.testing.assert_index_equal(index.data, data)


class TestGetItem:

    def test_get_item_by_integer(self):
        """Test getting item by integer index."""
        indices = ["a", "b", "c"]
        index = Index(indices=indices)
        assert index[0] == "a"
        assert index[1] == "b"
        assert index[2] == "c"

    def test_get_item_by_slice(self):
        """Test getting items by slice."""
        indices = ["a", "b", "c", "d", "e"]
        name = "my_index"
        data_name = "my_data"
        index = Index(indices=indices, name=name, data_name=data_name)
        sliced_index = index[1:4]
        expected_indices = ["b", "c", "d"]
        expected_data = pd.Index(expected_indices, name=data_name)
        assert isinstance(sliced_index, Index)
        assert sliced_index.indices == expected_indices
        assert sliced_index.name == name
        assert sliced_index.data.name == data_name
        pd.testing.assert_index_equal(sliced_index.data, expected_data)

    def test_get_item_by_list(self):
        """Test getting items by list of indices."""
        indices = ["a", "b", "c", "d", "e"]
        index = Index(indices=indices)
        selected_index = index[[0, 2, 4]]
        expected_indices = ["a", "c", "e"]
        expected_data = pd.Index(expected_indices, name="data")
        assert isinstance(selected_index, Index)
        assert selected_index.indices == expected_indices
        assert selected_index.name == "index"
        assert selected_index.data.name == "data"
        pd.testing.assert_index_equal(selected_index.data, expected_data)


class TestContains:

    def test_contains(self):
        """Test the __contains__ method."""
        indices = ["a", "b", "c"]
        index = Index(indices=indices)
        assert "a" in index
        assert "b" in index
        assert "c" in index
        assert "d" not in index


class TestGenerateDefault:

    def test_generate_default_with_all_parameters(self):
        """Test the generate_default class method with all parameters."""
        initial_index = 1
        size = 4
        prefix = "F"
        name = "feature_index"
        data_name = "features"
        index = Index.generate_default(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            name=name,
            data_name=data_name,
        )
        expected_indices = [f"F{i}" for i in range(initial_index, initial_index + size)]
        expected_data = pd.Index(expected_indices, name=data_name)
        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_generate_default_with_minimal_parameters(self):
        """Test the generate_default class method with minimal parameters."""
        index = Index.generate_default()
        expected_indices = [f"index{i}" for i in range(10)]
        expected_data = pd.Index(expected_indices, name="data")
        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == "index"
        assert index.data.name == "data"
        pd.testing.assert_index_equal(index.data, expected_data)


class TestEquality:

    def test_equality_same_indices(self):
        """Test equality of two Index objects with the same indices."""
        indices = ["a", "b", "c"]
        index1 = Index(indices=indices, name="index1", data_name="data")
        index2 = Index(indices=indices, name="index2", data_name="data")
        assert index1 == index2

    def test_inequality_different_indices(self):
        """Test inequality of two Index objects with different indices."""
        index1 = Index(indices=["a", "b", "c"], name="index1", data_name="data")
        index2 = Index(indices=["a", "b", "d"], name="index2", data_name="data")
        assert index1 != index2

    def test_inequality_different_order(self):
        """Test inequality of two Index objects with same indices in different order."""
        index1 = Index(indices=["a", "b", "c"], name="index1", data_name="data")
        index2 = Index(indices=["c", "b", "a"], name="index2", data_name="data")
        assert index1 != index2


class TestLength:

    def test_length(self):
        """Test the __len__ method."""
        indices = ["a", "b", "c", "d"]
        index = Index(indices=indices)
        assert len(index) == 4


class TestIter:

    def test_iteration(self):
        """Test the __iter__ method."""
        indices = ["a", "b", "c"]
        index = Index(indices=indices)
        iterated_indices = list(index)
        assert iterated_indices == indices


class TestValidation:

    def test_valid_inputs_all_parameters(self):
        """Test that valid inputs with all parameters pass validation."""
        indices = ["a", "b", "c"]
        name = "my_index"
        data_name = "data"
        index = Index(indices=indices, name=name, data_name=data_name)
        assert index.indices == indices
        assert index.name == name
        assert index.data.name == data_name

    def test_valid_inputs_minimal(self):
        """Test that valid inputs with minimal parameters pass validation."""
        indices = [1, 2, 3]
        index = Index(indices=indices)
        assert index.indices == indices
        assert index.name == "index"
        assert index.data.name == "data"

    def test_indices_not_list_raises_error(self):
        """Test that non-list indices raise TypeError."""
        with pytest.raises(ValueError):
            Index(indices="abc")

    def test_indices_with_non_hashable_items_raises_error(self):
        """Test that indices containing non-hashable items raise TypeError."""
        with pytest.raises(ValueError):
            Index(indices=["a", "b", ["c"]])

    def test_indices_with_duplicates_raises_error(self):
        """Test that duplicate items in indices raise ValueError."""
        with pytest.raises(ValueError):
            Index(indices=["a", "b", "a"])

    def test_non_hashable_name_raises_error(self):
        """Test that non-hashable name raises TypeError."""
        with pytest.raises(ValueError):
            Index(indices=["a", "b", "c"], name=["not_hashable"])

    def test_non_hashable_data_name_raises_error(self):
        """Test that non-hashable data_name raises TypeError."""
        with pytest.raises(ValueError):
            Index(indices=["a", "b", "c"], data_name={"not": "hashable"})

    def test_empty_indices_list_is_valid(self):
        """Test that empty indices list is valid."""
        index = Index(indices=[])
        assert index.indices == []
        assert len(index) == 0

    def test_indices_with_mixed_hashable_types(self):
        """Test that indices can contain mixed hashable types."""
        indices = ["string", 1, 2.5, ("tuple",), None]
        index = Index(indices=indices)
        assert index.indices == indices
