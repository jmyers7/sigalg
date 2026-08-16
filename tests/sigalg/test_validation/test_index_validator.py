import pandas as pd
import pytest
from sigalg.validation.index_validator import IndexValidator


def test_list_with_duplicates_raises():
    with pytest.raises(ValueError, match="index must not contain duplicate values"):
        IndexValidator(indices=[1, 1], name="I", default_name="I")


def test_list_with_duplicate_tuples_raises():
    with pytest.raises(ValueError, match="index must not contain duplicate values"):
        IndexValidator(indices=[(1, 2), (1, 2)], name="I", default_name="I")


def test_pd_index_with_duplicates_raises():
    with pytest.raises(ValueError, match="index must not contain duplicate values"):
        IndexValidator(indices=pd.Index([1, 1]), name="I", default_name="I")


def test_pd_multiindex_with_duplicates_raises():
    with pytest.raises(ValueError, match="index must not contain duplicate values"):
        IndexValidator(
            indices=pd.MultiIndex.from_tuples([(1, 2), (1, 2)]),
            name="I",
            default_name="I",
        )


def test_list_with_mixed_tuples_and_non_tuples_raises():
    with pytest.raises(
        ValueError, match="If the list contains tuples, all elements must be tuples"
    ):
        IndexValidator(indices=[(1, 2), 3], name="I", default_name="I")


def test_list_with_inconsistent_tuple_length_raises():
    with pytest.raises(ValueError, match="All tuples must have the same length"):
        IndexValidator(indices=[(1, 2), (3, 4, 5)], name="I", default_name="I")


def test_variable_names_non_list_raises():
    with pytest.raises(
        TypeError, match="variable_names must be a list of Hashable or `None`."
    ):
        IndexValidator(
            indices=[1, 2], name="I", variable_names="not a list", default_name="I"
        )


def test_variable_names_contains_non_hashable_raises():
    with pytest.raises(
        TypeError, match="All elements in variable_names must be Hashable."
    ):
        IndexValidator(
            indices=[1, 2], name="I", variable_names=["valid", {}], default_name="I"
        )


def test_variable_names_with_duplicates_raises():
    with pytest.raises(
        ValueError, match="variable_names must not contain duplicate values."
    ):
        IndexValidator(
            indices=[1, 2], name="I", variable_names=["dup", "dup"], default_name="I"
        )


def test_too_many_variable_names_with_multiindex_raises():
    indices = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    with pytest.raises(
        ValueError,
        match="Length of names must match number of levels in MultiIndex",
    ):
        IndexValidator(
            indices=indices,
            name="I",
            variable_names=["num", "letter", "extra"],
            default_name="I",
        )


def test_too_many_variable_names_with_index_raises():
    indices = pd.Index([1, 2, 3])
    with pytest.raises(
        ValueError,
        match="Length of new names must be 1",
    ):
        IndexValidator(
            indices=indices, name="I", variable_names=["one", "two"], default_name="I"
        )
