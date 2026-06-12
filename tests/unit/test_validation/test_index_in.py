import pandas as pd
import pytest

from sigalg.validation.index_in import IndexIn


class TestIndexLike:
    def test_pd_index_with_duplicates_raises(self):
        data = pd.Index([1, 2, 2])

        with pytest.raises(ValueError, match="index must not contain duplicate values"):
            IndexIn(indices=data, name="I", variable_names=["x"])

    def test_pd_multiindex_with_duplicates_raises(self):
        tuples = [(1, "a"), (1, "a")]
        data = pd.MultiIndex.from_tuples(tuples)

        with pytest.raises(ValueError, match="index must not contain duplicate values"):
            IndexIn(indices=data, name="I", variable_names=["x", "y"])

    def test_list_with_duplicates_raises(self):
        indices = [1, 2, 2]

        with pytest.raises(ValueError, match="index must not contain duplicate values"):
            IndexIn(indices=indices, name="I", variable_names=["x"])

    def test_list_with_duplicate_tuples_raises(self):
        indices = [(1, "a"), (1, "a")]

        with pytest.raises(ValueError, match="index must not contain duplicate values"):
            IndexIn(indices=indices, name="I", variable_names=["x", "y"])

    def test_list_of_tuples_with_other_items_raises(self):
        indices = [(1, "a"), 2]

        with pytest.raises(
            ValueError,
            match="If the list contains tuples, all elements must be tuples",
        ):
            IndexIn(indices=indices, name="I", variable_names=["x", "y"])

    def test_list_of_tuples_with_different_lengths_raises(self):
        indices = [(1, "a"), (2, "b", "extra")]

        with pytest.raises(ValueError, match="All tuples must have the same length"):
            IndexIn(indices=indices, name="I", variable_names=["x", "y"])

    def test_list_with_non_hashable_raises(self):
        indices = [1, 2, [3]]

        with pytest.raises(
            ValueError, match="All elements in the index must be Hashable"
        ):
            IndexIn(indices=indices, name="I", variable_names=["x"])

    def test_list_coerces_to_pd_index(self):
        indices = [1, 2, 3]
        v = IndexIn(indices=indices, name="I", variable_names=["x"])
        expected_index = pd.Index(indices, name="x")

        pd.testing.assert_index_equal(v.indices, expected_index)

    def test_list_of_tuples_coerces_to_multiindex(self):
        indices = [(1, "a"), (2, "b")]
        v = IndexIn(indices=indices, name="I", variable_names=["x", "y"])
        expected_index = pd.MultiIndex.from_tuples(indices, names=["x", "y"])

        pd.testing.assert_index_equal(v.indices, expected_index)

    def test_preserves_pd_index(self):
        indices = pd.Index([1, 2, 3], name="x")
        v = IndexIn(indices=indices, name="I", variable_names=["x"])

        pd.testing.assert_index_equal(v.indices, indices)

    def test_preserves_pd_multiindex(self):
        tuples = [(1, "a"), (2, "b")]
        indices = pd.MultiIndex.from_tuples(tuples, names=["x", "y"])
        v = IndexIn(indices=indices, name="I", variable_names=["x", "y"])

        pd.testing.assert_index_equal(v.indices, indices)


class TestVariableNames:
    def test_variable_names_with_non_list_raises(self):
        with pytest.raises(
            TypeError, match="variable_names must be a list of Hashable."
        ):
            IndexIn(indices=[1, 2, 3], name="I", variable_names="not a list")

    def test_variable_names_with_non_hashable_raises(self):
        with pytest.raises(
            TypeError, match="All elements in variable_names must be Hashable."
        ):
            IndexIn(indices=[1, 2, 3], name="I", variable_names=["x", ["not hashable"]])

    def test_variable_names_with_duplicates_raises(self):
        with pytest.raises(
            ValueError, match="variable_names must not contain duplicate values."
        ):
            IndexIn(indices=[1, 2, 3], name="I", variable_names=["x", "x"])

    def test_mismatched_variable_names_with_multiindex_raises(self):
        data = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["x", "y"])

        with pytest.raises(
            ValueError,
            match="The variable names must match the names of the index dimensions.",
        ):
            IndexIn(indices=data, name="I", variable_names=["x", "z"])

    def test_multiindex_with_names_and_no_variable_names(self):
        data = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["x", "y"])
        v = IndexIn(indices=data, name="I", variable_names=None)

        assert v.variable_names == ["x", "y"]
        assert v.indices.names == ["x", "y"]

    def test_multiindex_with_no_names_and_variable_names(self):
        data = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
        v = IndexIn(indices=data, name="I", variable_names=["x", "y"])

        assert v.variable_names == ["x", "y"]
        assert v.indices.names == ["x", "y"]

    def test_empty_variable_names_with_multiindex_with_no_names(self):
        data = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
        v = IndexIn(
            indices=data,
            name="I",
            variable_names=None,
        )

        assert v.variable_names == ["I_0", "I_1"]
        assert v.indices.names == ["I_0", "I_1"]

    def test_mismatched_variable_name_with_index_raises(self):
        data = pd.Index([1, 2, 3], name="x")

        with pytest.raises(
            ValueError,
            match="The variable name must match the name of underlying pd.Index",
        ):
            IndexIn(indices=data, name="I", variable_names=["y"])

    def test_index_with_name_and_no_variable_name(self):
        data = pd.Index([1, 2, 3], name="x")
        v = IndexIn(indices=data, name="I", variable_names=None)

        assert v.variable_names == ["x"]
        assert v.indices.name == "x"

    def test_index_with_no_name_and_variable_name(self):
        data = pd.Index([1, 2, 3])
        v = IndexIn(indices=data, name="I", variable_names=["x"])

        assert v.variable_names == ["x"]
        assert v.indices.name == "x"

    def test_index_with_no_name_and_no_variable_name(self):
        data = pd.Index([1, 2, 3])
        v = IndexIn(indices=data, name="I", variable_names=None)

        assert v.variable_names == ["I"]
        assert v.indices.name == "I"
