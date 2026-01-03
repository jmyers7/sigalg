import pandas as pd
import pytest

from sigalg.core import Index


class TestConstructor:

    @pytest.mark.parametrize(
        "indices, name, data_name",
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
        index = Index(name=name, data_name=data_name).from_list(indices)
        expected_data = pd.Index(indices, name=data_name)

        assert index.indices == indices
        assert index.name == name
        assert index.data.name == data_name
        pd.testing.assert_index_equal(index.data, expected_data)

    @pytest.mark.parametrize(
        "indices, name, data_name",
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
            Index(name=name, data_name=data_name).from_list(indices)


class TestFromPandas:

    @pytest.mark.parametrize(
        "pd_index, name",
        [
            pytest.param(
                pd.Index(["a", "b", "c"], name="my_data"),
                "my_data",
                id="index_with_name",
            ),
            pytest.param(pd.Index([1, 2, 3]), None, id="none_name"),
            pytest.param(pd.Index([1, 2, 3]), "default_name_flag", id="default_name"),
            pytest.param(
                pd.Index([], name="empty_data"), "empty_data", id="empty_index"
            ),
        ],
    )
    def test_from_pandas(self, pd_index, name):
        """Test the from_pandas class method."""
        if name == "default_name_flag":
            index = Index().from_pandas(data=pd_index)
            name = None
        else:
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
    index = Index().from_list(indices)

    assert "a" in index
    assert "b" in index
    assert "c" in index
    assert "d" not in index


class TestGenerateSequence:

    @pytest.mark.parametrize(
        "initial_index, size, prefix, name, data_name, expected_indices",
        [
            pytest.param(
                1,
                4,
                "f",
                "feature_index",
                "features",
                ["f_1", "f_2", "f_3", "f_4"],
                id="custom_prefix_and_names",
            ),
            pytest.param(
                0,
                10,
                None,
                None,
                None,
                list(range(0, 10)),
                id="none_prefix_and_names",
            ),
            pytest.param(
                5,
                2,
                "default_prefix_flag",
                "custom_name",
                "custom_data_name",
                list(range(5, 7)),
                id="default_prefix",
            ),
            pytest.param(
                8,
                2,
                "X",
                "default_name_flag",
                "custom_data_name",
                ["X_8", "X_9"],
                id="default_name",
            ),
            pytest.param(
                3,
                4,
                "feature",
                "feat_idx",
                "default_data_name",
                ["feature_3", "feature_4", "feature_5", "feature_6"],
                id="default_data_name",
            ),
        ],
    )
    def test_generate_sequence(
        self, initial_index, size, prefix, name, data_name, expected_indices
    ):
        """Test the generate_sequence class method with various parameters."""
        if prefix == "default_prefix_flag":
            index = Index.generate_sequence(
                initial_index=initial_index,
                size=size,
                name=name,
                data_name=data_name,
            )
            prefix = None
        elif name == "default_name_flag":
            index = Index.generate_sequence(
                initial_index=initial_index,
                size=size,
                prefix=prefix,
                data_name=data_name,
            )
            name = None
        elif data_name == "default_data_name":
            index = Index.generate_sequence(
                initial_index=initial_index,
                size=size,
                prefix=prefix,
                name=name,
            )
            data_name = None
        else:
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

    @pytest.mark.parametrize(
        "initial_index, size, prefix, name, data_name",
        [
            pytest.param(0, -10, None, None, None, id="negative_size"),
            pytest.param(0, "ten", None, None, None, id="non_integer_size"),
            pytest.param("zero", 10, None, None, None, id="non_integer_initial_index"),
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
            Index.generate_sequence(
                initial_index=initial_index,
                size=size,
                prefix=prefix,
                name=name,
                data_name=data_name,
            )


class TestEquality:

    @pytest.mark.parametrize(
        "given, other",
        [
            pytest.param(
                Index().from_list(["a", "b"]),
                Index().from_list(["b", "a"]),
                id="different_order",
            ),
            pytest.param(
                Index().from_list(["a", "b"]),
                Index().from_list(["a", "b", "c"]),
                id="different_length",
            ),
            pytest.param(
                Index(data_name="index1").from_list(["a", "b"]),
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
                Index(name="index", data_name="data").from_list(["a", "b", "c"]),
                Index(name="index", data_name="data").from_list(["a", "b", "c"]),
                id="equal",
            ),
            pytest.param(
                Index(name="index1", data_name="data1").from_list(["a", "b", "c"]),
                Index(name="index2", data_name="data2").from_list(["a", "b", "c"]),
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
    index = Index().from_list(indices)

    assert len(index) == 4


def test_iteration():
    """Test the __iter__ method."""
    indices = ["a", "b", "c"]
    index = Index().from_list(indices)
    iterated_indices = list(index)

    assert iterated_indices == indices
