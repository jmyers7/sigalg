import re
from collections import Counter

import numpy as np
import pandas as pd


def to_df(  # noqa: D103
    data: pd.Series | pd.DataFrame,
    suffix: str | None = None,
    subscript_index_flag: bool = False,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        result = data.copy()
    else:
        result = data.to_frame()

    if suffix is not None:
        result = result.add_suffix(suffix)

    if subscript_index_flag:
        result.index.names = [f"{name}_d" for name in result.index.names]
    return result


def add_suffix(lst: list, suffix: str) -> list:  # noqa: D103
    return [f"{name}{suffix}" for name in lst]


def add_subscript(lst: list, subscript: str) -> list:  # noqa: D103
    return [f"{name}_{subscript}" for name in lst]


def remove_subscript(lst: list) -> list:
    return [name.split("_")[0] for name in lst]


def flatten(t):  # noqa: D103
    return tuple(
        x for item in t for x in (item if isinstance(item, tuple) else (item,))
    )


def subscript_var_names(lists, grouped: bool = False):  # noqa: D103
    names = [x for names in lists for x in names]
    if set(Counter(names).values()) == {1}:
        return names if not grouped else lists

    def base(s):
        m = re.fullmatch(r"(.+)_(\d+)", s)
        return (s, None) if not m else (m.group(1), int(m.group(2)))

    tuples = [base(s) for lst in lists for s in lst]
    bases = [t[0] for t in tuples]
    common_bases = {base for base, count in Counter(bases).items() if count >= 2}

    for base in common_bases:
        idx = 0
        for i, t in enumerate(tuples):
            if t[0] == base:
                tuples[i] = (base, idx)
                idx += 1

    result = [f"{t[0]}_{t[1]}" if t[1] is not None else t[0] for t in tuples]

    if grouped:
        grouped_result = []
        for lst in lists:
            grouped_result.append(result[: len(lst)])
            del result[: len(lst)]
        return grouped_result
    else:
        return result


def pandas_all_equal(
    first: pd.Index | pd.Series | pd.DataFrame,
    second: pd.Index | pd.Series | pd.DataFrame,
    check_series_names: bool = True,
) -> bool:
    """Check whether two pandas objects of the same type are exactly equal.

    Examples
    --------
    Values and names of indices must both match:

    >>> i1 = pd.Index([1, 2], name="a")
    >>> i2 = pd.Index([1, 2], name="b")
    >>> pandas_all_equal(i1, i2)
    False
    >>> i3 = pd.Index([1, 2], name="a")
    >>> i4 = pd.Index([1, 2], name="a")
    >>> pandas_all_equal(i3, i4)
    True

    Level names of multi-indices must match and be in the same order.

    >>> mi1 = pd.MultiIndex.from_tuples([(1, 2), (3, 4)], names=["x", "y"])
    >>> mi2 = pd.MultiIndex.from_tuples([(1, 2), (3, 4)], names=["y", "x"])
    >>> pandas_all_equal(mi1, mi2)
    False
    >>> mi3 = pd.MultiIndex.from_arrays([(1, 2), (3, 4)], names=["x", "y"])
    >>> mi4 = pd.MultiIndex.from_arrays([(1, 2), (3, 4)], names=["x", "y"])
    >>> pandas_all_equal(mi3, mi4)
    True

    Series are equal if they have same values, the same names, and the same indices (the latter being judged equal by the above criteria).

    >>> s1 = pd.Series([1, 2], index=mi1, name="s")
    >>> s2 = pd.Series([1, 2], index=mi2, name="s")
    >>> pandas_all_equal(s1, s2)
    False
    >>> s3 = pd.Series([1, 2], index=mi3, name="s")
    >>> s4 = pd.Series([1, 2], index=mi4, name="s")
    >>> pandas_all_equal(s3, s4)
    True
    >>> s3 = pd.Series([1, 2], index=mi3, name="s")
    >>> s4 = pd.Series([1, 2], index=mi4, name="s")
    >>> pandas_all_equal(s3, s4)
    True
    >>> s5 = pd.Series([1, 2], index=mi3, name="s")
    >>> s6 = pd.Series([1, 2], index=mi4, name="t")
    >>> pandas_all_equal(s5, s6)
    False

    Data frames are equal if they have the same values and the same indices and column indices (the latter two being judged by the above criteria).

    >>> df1 = pd.DataFrame([(1, 2), (3, 4)], index=mi1, columns=i3)
    >>> df2 = pd.DataFrame([(1, 2), (3, 4)], index=mi2, columns=i4)
    >>> pandas_all_equal(df1, df2)
    False
    >>> df3 = pd.DataFrame([(1, 2), (3, 4)], index=mi3, columns=i1)
    >>> df4 = pd.DataFrame([(1, 2), (3, 4)], index=mi4, columns=i2)
    >>> pandas_all_equal(df3, df4)
    False
    >>> df5 = pd.DataFrame([(1, 2), (3, 4)], index=mi3, columns=i3)
    >>> df6 = pd.DataFrame([(1, 2), (3, 4)], index=mi4, columns=i4)
    >>> pandas_all_equal(df5, df6)
    True
    """
    if isinstance(first, pd.Index) and isinstance(second, pd.Index):
        return first.names == second.names and np.array_equal(first, second)

    elif isinstance(first, pd.Series) and isinstance(second, pd.Series):
        if pandas_all_equal(first.index, second.index) and np.array_equal(
            first, second
        ):
            if check_series_names:
                return first.name == second.name
            else:
                return True
        else:
            return False

    elif isinstance(first, pd.DataFrame) and isinstance(second, pd.DataFrame):
        return (
            pandas_all_equal(first.index, second.index)
            and pandas_all_equal(first.columns, second.columns)
            and np.array_equal(first, second)
        )

    else:
        return False
