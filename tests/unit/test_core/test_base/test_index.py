import pandas as pd
import pytest

from sigalg.core import Index


class TestConstructor:

    @pytest.mark.parametrize(
        "indices,name,data_name",
        [
            pytest.param(["x", "y", "z"], "index1", "data1", id="all_params"),
            pytest.param([], "empty_index", "empty_data", id="empty"),
            pytest.param(["a", "b", "c"], None, None, id="no_names"),
            pytest.param(["a", "b", "c"], "custom_index", None, id="custom_name"),
            pytest.param(["a", "b", "c"], None, "custom_data", id="custom_data_name"),
        ],
    )
    def test_constructor(self, indices, name, data_name):
        """Test constructor with various combinations of parameters."""
        index = Index(indices=indices, name=name, data_name=data_name)
        expected_data_name = data_name if data_name is not None else "data"
        expected_name = name if name is not None else "index"
        expected_data = pd.Index(indices, name=expected_data_name)

        assert index.indices == indices
        assert index.name == expected_name
        assert index.data.name == expected_data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    @pytest.mark.parametrize(
        "indices,name,data_name",
        [
            pytest.param("abc", "index", "data", id="indices-not-list"),
            pytest.param(123, "index", "data", id="indices-not-iterable"),
            pytest.param([{"a": 1}], "index", "data", id="unhashable-elements"),
            pytest.param(["a", "b", "a"], "index", "data", id="duplicate-elements"),
            pytest.param(
                ["a", "b", "c"], ["not", "hashable"], "data", id="name-not-hashable"
            ),
            pytest.param(
                ["a", "b", "c"],
                "index",
                {"not": "hashable"},
                id="data_name-not-hashable",
            ),
        ],
    )
    def test_invalid_inputs_raise(self, indices, name, data_name):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises((TypeError, ValueError)):
            Index(indices=indices, name=name, data_name=data_name)


class TestFromPandas:

    @pytest.mark.parametrize(
        "pandas_index",
        [
            pytest.param(
                pd.Index(["a", "b", "c"], name="my_data"), id="index_with_name"
            ),
            pytest.param(pd.Index([1, 2, 3]), id="index_without_name"),
            pytest.param(pd.Index([], name="empty_data"), id="empty_index"),
        ],
    )
    def test_from_pandas(self, pandas_index):
        """Test the from_pandas class method."""
        index = Index.from_pandas(pandas_index)
        expected_name = "index"
        expected_data_name = pandas_index.name

        assert index.indices == list(pandas_index)
        assert index.name == expected_name
        assert index.data.name == expected_data_name
        pd.testing.assert_index_equal(index.data, pandas_index)

    def test_invalid_inputs_raise(self):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises(TypeError):
            Index.from_pandas(["not", "a", "pandas", "Index"])


class TestGetItem:

    @pytest.fixture
    def index(self):
        return Index(
            indices=["a", "b", "c", "d", "e"],
            name="my_index",
            data_name="my_data",
        )

    @pytest.mark.parametrize(
        "pos",
        [
            pytest.param(0, id="integer_index"),
            pytest.param(slice(1, 4), id="slice_index"),
            pytest.param([0, 2, 4], id="list_index"),
        ],
    )
    def test_get_item(self, index, pos):
        """Test the __getitem__ method with various index types."""
        result = index[pos]

        if isinstance(pos, int):
            assert result == index.indices[pos]
            return
        elif isinstance(pos, slice):
            expected_indices = index.indices[pos]
        elif isinstance(pos, list):
            expected_indices = [index.indices[i] for i in pos]
        expected_data = pd.Index(expected_indices, name=index.data.name)

        assert isinstance(result, Index)
        assert result.indices == expected_indices
        assert result.name == index.name
        assert result.data.name == index.data.name
        pd.testing.assert_index_equal(result.data, expected_data)

    @pytest.mark.parametrize(
        "pos",
        [
            pytest.param(10, id="out_of_bounds_integer"),
            pytest.param([0, 5], id="out_of_bounds_list"),
            pytest.param("invalid_type", id="invalid_type"),
            pytest.param(["a", "b"], id="invalid_list_contents"),
        ],
    )
    def test_invalid_pos_raises(self, index, pos):
        """Test that invalid positions raise IndexError."""
        with pytest.raises((IndexError, TypeError)):
            index[pos]


def test_contains():
    """Test the __contains__ method."""
    indices = ["a", "b", "c"]
    index = Index(indices=indices)

    assert "a" in index
    assert "b" in index
    assert "c" in index
    assert "d" not in index


class TestGenerateDefault:

    @pytest.mark.parametrize(
        "initial_index,size,prefix,name,data_name",
        [
            pytest.param(1, 4, "F", "feature_index", "features", id="all_params"),
            pytest.param(0, 10, None, None, None, id="all_defaults"),
            pytest.param(5, 2, None, "custom_index", None, id="custom_name"),
            pytest.param(2, 5, None, None, "custom_data", id="custom_data_name"),
        ],
    )
    def test_generate_default(self, initial_index, size, prefix, name, data_name):
        """Test the generate_default class method with various parameters."""

        initial_index = initial_index if initial_index is not None else 0
        size = size if size is not None else 10
        index = Index.generate_default(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            name=name,
            data_name=data_name,
        )

        expected_prefix = prefix if prefix is not None else "index"
        expected_name = name if name is not None else "index"
        expected_data_name = data_name if data_name is not None else "data"
        expected_indices = [
            f"{expected_prefix}{i}" for i in range(initial_index, initial_index + size)
        ]
        expected_data = pd.Index(expected_indices, name=expected_data_name)

        assert isinstance(index, Index)
        assert index.indices == expected_indices
        assert index.name == expected_name
        assert index.data.name == expected_data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    @pytest.mark.parametrize(
        "initial_index,size,prefix,name,data_name",
        [
            pytest.param(0, -10, None, None, None, id="negative_size"),
            pytest.param(0, "ten", None, None, None, id="non_integer_size"),
            pytest.param("zero", 10, None, None, None, id="non_integer_initial_index"),
            pytest.param(0, 10, 123, None, None, id="non_string_prefix"),
            pytest.param(
                0, 10, None, ["not", "hashable"], None, id="non_hashable_name"
            ),
            pytest.param(
                0, 10, None, None, {"not": "hashable"}, id="non_hashable_data_name"
            ),
        ],
    )
    def test_invalid_inputs_raise(self, initial_index, size, prefix, name, data_name):
        """Test that invalid inputs raise appropriate exceptions."""
        with pytest.raises((TypeError, ValueError)):
            Index.generate_default(
                initial_index=initial_index,
                size=size,
                prefix=prefix,
                name=name,
                data_name=data_name,
            )


class TestEquality:

    @pytest.mark.parametrize(
        "given,other",
        [
            pytest.param(
                Index(indices=["a", "b"]),
                Index(indices=["b", "a"]),
                id="different_order",
            ),
            pytest.param(
                Index(indices=["a", "b"]),
                Index(indices=["a", "b", "c"]),
                id="different_length",
            ),
            pytest.param(
                Index(indices=["a", "b"], data_name="index1"),
                Index(indices=["a", "b"], data_name="index2"),
                id="different_data_name",
            ),
            pytest.param(
                Index(indices=["a", "b"], data_name="index1"),
                "not_an_index",
                id="wrong_type",
            ),
        ],
    )
    def test_non_equality(self, given, other):
        """Test the __eq__ method for inequality."""
        assert given != other

    @pytest.mark.parametrize(
        "given,other",
        [
            pytest.param(
                Index(indices=["a", "b", "c"], name="index", data_name="data"),
                Index(indices=["a", "b", "c"], name="index", data_name="data"),
                id="equal",
            ),
            pytest.param(
                Index(indices=["a", "b", "c"], name="index1", data_name="data"),
                Index(indices=["a", "b", "c"], name="index2", data_name="data"),
                id="equal_but_different_names",
            ),
        ],
    )
    def test_equality(self, given, other):
        """Test the __eq__ method for equality."""
        assert given == other


def test_length():
    """Test the __len__ method."""
    indices = ["a", "b", "c", "d"]
    index = Index(indices=indices)

    assert len(index) == 4


def test_iteration():
    """Test the __iter__ method."""
    indices = ["a", "b", "c"]
    index = Index(indices=indices)
    iterated_indices = list(index)

    assert iterated_indices == indices
