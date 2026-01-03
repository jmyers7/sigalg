import pandas as pd
import pytest

from sigalg.core import Index


class TestConstructor:

    def test_constructor_all_params(self):
        """Test constructor with all parameters provided."""
        indices = ["x", "y", "z"]
        name = "index"
        data_name = "data"
        index = Index(name=name, data_name=data_name).from_list(indices)
        expected_data = pd.Index(indices, name=data_name)

        assert index.indices == indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_constructor_empty(self):
        """Test constructor with empty indices."""
        indices = []
        name = "empty_index"
        data_name = "empty_data"
        index = Index(name=name, data_name=data_name).from_list(indices)
        expected_data = pd.Index(indices, name=data_name)

        assert index.indices == indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_constructor_no_names(self):
        """Test constructor with no names provided."""
        indices = ["a", "b", "c"]
        index = Index().from_list(indices)
        expected_data = pd.Index(indices, name=None)

        assert index.indices == indices
        assert index.name is None
        assert index.data.name is None
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

    def test_invalid_name_not_hashable(self):
        """Test that non-hashable name raises TypeError."""
        with pytest.raises(TypeError):
            Index(name=["not", "hashable"]).from_list(["a", "b", "c"])

    # @pytest.mark.parametrize(
    #     "indices, name, data_name",
    #     [
    #         pytest.param(["x", "y", "z"], "index1", "data1", id="all_params"),
    #         pytest.param([], "empty_index", "empty_data", id="empty"),
    #         pytest.param(["a", "b", "c"], None, None, id="no_names"),
    #     ],
    # )
    # def test_constructor(self, indices, name, data_name):
    #     """Test constructor with various combinations of parameters."""
    #     index = Index(name=name, data_name=data_name).from_list(indices)
    #     expected_data = pd.Index(indices, name=data_name)

    #     assert index.indices == indices
    #     assert index.name == name
    #     assert index.data.name == data_name
    #     pd.testing.assert_index_equal(index.data, expected_data)

    # @pytest.mark.parametrize(
    #     "indices, name, data_name",
    #     [
    #         pytest.param("abc", "index", "data", id="indices-not-list"),
    #         pytest.param(123, "index", "data", id="indices-not-iterable"),
    #         pytest.param([{"a": 1}], "index", "data", id="unhashable-elements"),
    #         pytest.param(["a", "b", "a"], "index", "data", id="duplicate-elements"),
    #         pytest.param(
    #             ["a", "b", "c"], ["not", "hashable"], "data", id="name-not-hashable"
    #         ),
    #         pytest.param(
    #             ["a", "b", "c"],
    #             "index",
    #             {"not": "hashable"},
    #             id="data_name-not-hashable",
    #         ),
    #     ],
    # )
    # def test_invalid_inputs_raise(self, indices, name, data_name):
    #     """Test that invalid inputs raise appropriate exceptions."""
    #     with pytest.raises((TypeError, ValueError)):
    #         Index(name=name, data_name=data_name).from_list(indices)


class TestFromPandas:

    def test_from_pandas_with_name(self):
        """Test from_pandas with pandas Index that has a name."""
        pd_index = pd.Index(["a", "b", "c"], name="my_data")
        name = "my_data"
        index = Index(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(index.data, pd_index)
        assert index.indices == list(pd_index)
        assert index.name == name
        assert index.data.name == pd_index.name

    def test_from_pandas_with_none_name(self):
        """Test from_pandas with None as name parameter."""
        pd_index = pd.Index([1, 2, 3])
        name = None
        index = Index(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(index.data, pd_index)
        assert index.indices == list(pd_index)
        assert index.name == name
        assert index.data.name == pd_index.name

    def test_from_pandas_with_default_name(self):
        """Test from_pandas with default name (not specified)."""
        pd_index = pd.Index([1, 2, 3])
        index = Index().from_pandas(data=pd_index)
        name = None

        pd.testing.assert_index_equal(index.data, pd_index)
        assert index.indices == list(pd_index)
        assert index.name == name
        assert index.data.name == pd_index.name

    def test_from_pandas_empty_index(self):
        """Test from_pandas with empty pandas Index."""
        pd_index = pd.Index([], name="empty_data")
        name = "empty_data"
        index = Index(name=name).from_pandas(data=pd_index)

        pd.testing.assert_index_equal(index.data, pd_index)
        assert index.indices == list(pd_index)
        assert index.name == name
        assert index.data.name == pd_index.name

    def test_invalid_inputs_raise(self):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(TypeError):
            Index().from_pandas(["not", "a", "pandas", "Index"])


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
        with pytest.raises((IndexError, TypeError)):
            index[10]

    def test_invalid_out_of_bounds_list_raises(self, index):
        """Test that out of bounds list index raises IndexError."""
        with pytest.raises((IndexError, TypeError)):
            index[[0, 5]]

    def test_invalid_type_string_raises(self, index):
        """Test that invalid type (string) raises TypeError."""
        with pytest.raises((IndexError, TypeError)):
            index["invalid_type"]

    def test_invalid_list_contents_raises(self, index):
        """Test that list with invalid contents raises TypeError."""
        with pytest.raises((IndexError, TypeError)):
            index[["a", "b"]]


def test_contains():
    """Test the __contains__ method."""
    indices = ["a", "b", "c"]
    index = Index().from_list(indices)

    assert "a" in index
    assert "b" in index
    assert "c" in index
    assert "d" not in index


class TestGenerateSequence:

    def test_generate_sequence_custom_prefix_and_names(self):
        """Test generate_sequence with custom prefix and names."""
        initial_index = 1
        size = 4
        prefix = "f"
        name = "feature_index"
        data_name = "features"
        expected_indices = ["f_1", "f_2", "f_3", "f_4"]

        index = Index.generate_sequence(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            name=name,
            data_name=data_name,
        )
        expected_data = pd.Index(expected_indices, name=data_name)

        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_generate_sequence_none_prefix_and_names(self):
        """Test generate_sequence with None prefix and names."""
        initial_index = 0
        size = 10
        prefix = None
        name = None
        data_name = None
        expected_indices = list(range(0, 10))

        index = Index.generate_sequence(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            name=name,
            data_name=data_name,
        )
        expected_data = pd.Index(expected_indices, name=data_name)

        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_generate_sequence_default_prefix(self):
        """Test generate_sequence with default prefix."""
        initial_index = 5
        size = 2
        name = "custom_name"
        data_name = "custom_data_name"
        expected_indices = list(range(5, 7))

        index = Index.generate_sequence(
            initial_index=initial_index,
            size=size,
            name=name,
            data_name=data_name,
        )
        expected_data = pd.Index(expected_indices, name=data_name)

        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_generate_sequence_default_name(self):
        """Test generate_sequence with default name."""
        initial_index = 8
        size = 2
        prefix = "X"
        data_name = "custom_data_name"
        expected_indices = ["X_8", "X_9"]

        index = Index.generate_sequence(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            data_name=data_name,
        )
        name = None
        expected_data = pd.Index(expected_indices, name=data_name)

        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_generate_sequence_default_data_name(self):
        """Test generate_sequence with default data_name."""
        initial_index = 3
        size = 4
        prefix = "feature"
        name = "feat_idx"
        expected_indices = ["feature_3", "feature_4", "feature_5", "feature_6"]

        index = Index.generate_sequence(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            name=name,
        )
        data_name = None
        expected_data = pd.Index(expected_indices, name=data_name)

        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    def test_invalid_negative_size_raises(self):
        """Test that negative size raises ValueError."""
        with pytest.raises((TypeError, ValueError)):
            Index.generate_sequence(
                initial_index=0,
                size=-10,
                prefix=None,
                name=None,
                data_name=None,
            )

    def test_invalid_non_integer_size_raises(self):
        """Test that non-integer size raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            Index.generate_sequence(
                initial_index=0,
                size="ten",
                prefix=None,
                name=None,
                data_name=None,
            )

    def test_invalid_non_integer_initial_index_raises(self):
        """Test that non-integer initial_index raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            Index.generate_sequence(
                initial_index="zero",
                size=10,
                prefix=None,
                name=None,
                data_name=None,
            )

    def test_invalid_non_hashable_name_raises(self):
        """Test that non-hashable name raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            Index.generate_sequence(
                initial_index=0,
                size=10,
                prefix=None,
                name=["not", "hashable"],
                data_name=None,
            )

    def test_invalid_non_hashable_data_name_raises(self):
        """Test that non-hashable data_name raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            Index.generate_sequence(
                initial_index=0,
                size=10,
                prefix=None,
                name=None,
                data_name={"not": "hashable"},
            )


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


def test_length():
    """Test the __len__ method."""
    indices = ["a", "b", "c", "d"]
    index = Index().from_list(indices)

    assert len(index) == 4


def test_iteration():
    """Test the __iter__ method."""
    indices = ["a", "b", "c"]
    index = Index().from_list(indices)
    iterated_indices = list(index)

    assert iterated_indices == indices
