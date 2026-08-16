from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def align_index(given: pd.Index | pd.Series | pd.DataFrame, by: pd.Index) -> pd.Index:
    """Align an index by the contents and level names of another.

    The method assumes:

    1. The level names of the two indices are not all `None`.

    2. The levels of each index do not contain duplicate names.

    3. The indices themselves do not contain duplicates.

    The method will return through a fast path first if the indices are already aligned content-wise and along level names. Failing that, the method will first align level names and check whether the contents match. If so, it returns. If not, it will then check whether the contents of the indices are equal as sets and attempt to reorder. If any of these steps fail, a `ValueError` is raised.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core._utils.index_helpers import align_index

    Fast paths.

    >>> I = pd.Index([1, 2], name="x")
    >>> J = pd.Index([1, 2], name="x")
    >>> align_index(J, by=I) is J
    True
    >>> s = pd.Series(["u", "v"], index=J)
    >>> align_index(s, by=I) is s
    True

    Check level names.

    >>> I = pd.Index([1, 2], name="x")
    >>> J = pd.Index([1, 2], name="y")
    >>> align_index(J, by=I)
    Traceback (most recent call last):
        ...
    ValueError: The level names of the two indices do not match.
    >>> s = pd.Series(["u", "v"], index=J)
    >>> align_index(s, by=I)
    Traceback (most recent call last):
        ...
    ValueError: The level names of the two indices do not match.

    Align level names.

    >>> I = pd.MultiIndex.from_tuples([(1, 2), (3, 4)], names=["a", "b"])
    >>> J = pd.MultiIndex.from_tuples([(2, 1), (4, 3)], names=["b", "a"])
    >>> align_index(J, by=I)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 2),
                (3, 4)],
               names=['a', 'b'])
    >>> s = pd.Series(["u", "v"], index=J)
    >>> align_index(s, by=I)
    a  b
    1  2    u
    3  4    v
    dtype: str

    Put level names in order, and align contents.

    >>> I = pd.MultiIndex.from_tuples([(1, 2), (3, 4)], names=["a", "b"])
    >>> J = pd.MultiIndex.from_tuples([(4, 3), (2, 1)], names=["b", "a"])
    >>> align_index(J, by=I)  # doctest: +NORMALIZE_WHITESPACE
    MultiIndex([(1, 2),
                (3, 4)],
               names=['a', 'b'])
    >>> s = pd.Series(["v", "u"], index=J)
    >>> align_index(s, by=I)
    a  b
    1  2    u
    3  4    v
    dtype: str

    Check contents.

    >>> I = pd.MultiIndex.from_tuples([(1, 2), (3, 4)], names=["a", "b"])
    >>> J = pd.MultiIndex.from_tuples([(2, 1), (5, 4)], names=["b", "a"])
    >>> align_index(J, by=I)
    Traceback (most recent call last):
        ...
    ValueError: The contents of the two indices do not coincide (as sets).
    >>> s = pd.Series(["u", "v"], index=J)
    >>> align_index(s, by=I)
    Traceback (most recent call last):
        ...
    ValueError: The contents of the two indices do not coincide (as sets).
    """
    import pandas as pd

    from .utils import pandas_all_equal

    if isinstance(given, pd.Series | pd.DataFrame):
        given_index = given.index
    else:
        given_index = given

    if pandas_all_equal(given_index, by):
        return given

    if set(given_index.names) != set(by.names):
        raise ValueError("The level names of the two indices do not match.")
    elif isinstance(given_index, pd.MultiIndex):
        aligned = given.reorder_levels(by.names)
    else:
        aligned = given.copy()

    if isinstance(aligned, pd.Index):
        if aligned.equals(by):
            return aligned
        else:
            aligned_index = aligned
    elif isinstance(aligned, pd.Series | pd.DataFrame):
        if aligned.index.equals(by):
            return aligned
        else:
            aligned_index = aligned.index

    if set(aligned_index) != set(by):
        raise ValueError("The contents of the two indices do not coincide (as sets).")
    else:
        return (
            aligned.reindex(by)[0]
            if isinstance(aligned, pd.Index)
            else aligned.reindex(by)
        )
