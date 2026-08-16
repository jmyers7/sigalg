from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable

    import pandas as pd

    PandasLike = pd.Series | pd.DataFrame


def sig_alg_func_to_measurable_func(
    self_data: PandasLike,
    sig_alg_data: PandasLike,
    parameter_names: list[Hashable],
) -> PandasLike:
    """Convert (parametrized) function data defined on atom identifiers of a sigma-algebra into (parametrized) function data defined on the measurable domain.

    * If `self_data` carries a multi-index, the level names are assumed to be equal to either `parameter_names + sig_alg_data.name` or `parameter_names + list(sig_alg_data.columns)`, in that order, depending on whether `sig_alg_data` is a `pd.Series` or `pd.DataFrame`.

    * If `self_data` carries a plain index, it is assumed `sig_alg_data` is a `pd.Series` and the name of the index is `sig_alg_data.name` and `parameter_names` is empty.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core._utils import sig_alg_func_to_measurable_func

    Create data for a 2-dimensional sigma-algebra over a 2-dimensional domain.

    >>> domain_data = pd.MultiIndex.from_tuples(
    ...     [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")], names=["x_0", "x_1"]
    ... )
    >>> sig_alg_data = pd.DataFrame({"u_0": [0, 0, 1, 2], "u_1": [1, 1, 2, 3]}, index=domain_data)
    >>> print(sig_alg_data)  # doctest: +NORMALIZE_WHITESPACE
             u_0  u_1
    x_0 x_1
    a   b      0    1
    c   d      0    1
    e   f      1    2
    g   h      2    3

    Create data for a 2-dimensional vector defined on sigma-algebra identifiers.

    >>> self_data = pd.DataFrame(
    ...     [(3, 4), (5, 6), (7, 8)],
    ...     index=pd.MultiIndex.from_tuples([(0, 1), (1, 2), (2, 3)], names=["u_0", "u_1"]),
    ...     columns=pd.Index([1, 2], name="i"),
    ... )
    >>> print(self_data)  # doctest: +NORMALIZE_WHITESPACE
    i        1  2
    u_0 u_1
    0   1    3  4
    1   2    5  6
    2   3    7  8

    Convert to the data of a measurable vector on the domain.

    >>> data = sig_alg_func_to_measurable_func(
    ...     self_data=self_data,
    ...     sig_alg_data=sig_alg_data,
    ...     parameter_names=[],
    ... )
    >>> print(data)  # doctest: +NORMALIZE_WHITESPACE
    i        1  2
    x_0 x_1
    a   b    3  4
    c   d    3  4
    e   f    5  6
    g   h    7  8

    Create data for a parametrized function defined on sigma-algebra identifiers.

    >>> param_sig_alg_space = pd.MultiIndex.from_tuples(
    ...     [
    ...         (0, 0, 0, 1),
    ...         (0, 0, 1, 2),
    ...         (0, 0, 2, 3),
    ...         (0, 1, 0, 1),
    ...         (0, 1, 1, 2),
    ...         (0, 1, 2, 3),
    ...         (1, 0, 0, 1),
    ...         (1, 0, 1, 2),
    ...         (1, 0, 2, 3),
    ...         (1, 1, 0, 1),
    ...         (1, 1, 1, 2),
    ...         (1, 1, 2, 3),
    ...     ],
    ...     names=["theta_0", "theta_1", "u_0", "u_1"],
    ... )
    >>> self_data = pd.Series(
    ...     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=param_sig_alg_space
    ... )
    >>> print(self_data)  # doctest: +NORMALIZE_WHITESPACE
    theta_0  theta_1  u_0  u_1
    0        0        0    1       1
                      1    2       2
                      2    3       3
             1        0    1       4
                      1    2       5
                      2    3       6
    1        0        0    1       7
                      1    2       8
                      2    3       9
             1        0    1      10
                      1    2      11
                      2    3      12
    dtype: int64

    Convert to the data of a parametrized measurable function on the domain.

    >>> data = sig_alg_func_to_measurable_func(
    ...     self_data=self_data,
    ...     sig_alg_data=sig_alg_data,
    ...     parameter_names=["theta_0", "theta_1"],
    ... )
    >>> print(data)  # doctest: +NORMALIZE_WHITESPACE
    theta_0  theta_1  x_0  x_1
    0        0        a    b       1
                      c    d       1
                      e    f       2
                      g    h       3
             1        a    b       4
                      c    d       4
                      e    f       5
                      g    h       6
    1        0        a    b       7
                      c    d       7
                      e    f       8
                      g    h       9
             1        a    b      10
                      c    d      10
                      e    f      11
                      g    h      12
    dtype: int64
    """
    import pandas as pd

    from .utils import add_subscript, to_df

    sig_alg_data = to_df(sig_alg_data)
    self_data = to_df(self_data)
    sig_alg_variable_names = list(sig_alg_data.columns)
    domain_variable_names = sig_alg_data.index.names
    function_index = self_data.columns

    sig_alg_data.columns = add_subscript(sig_alg_variable_names, "ID")
    self_data.index.names = add_subscript(parameter_names, "param") + add_subscript(
        sig_alg_variable_names, "ID"
    )

    data = (
        pd.merge(left=sig_alg_data.reset_index(), right=self_data.reset_index())
        .set_index(add_subscript(parameter_names, "param") + domain_variable_names)[
            function_index
        ]
        .sort_index()
        .squeeze(axis=1)
    )

    data.index.names = parameter_names + domain_variable_names

    if isinstance(data, pd.DataFrame):
        data.columns = function_index
    else:
        data.name = None

    return data


def compose_funcs(inner_data: PandasLike, outer_data: PandasLike) -> PandasLike:
    """Compute the data for the composition of two functions.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core._utils import compose_funcs

    Create data representing a function with 2-dimensional outputs, representing the "inner" function of the composition.

    >>> inner_data = pd.DataFrame(
    ...     [(1, 2), (3, 4), (5, 6)],
    ...     index=pd.Index(["a", "b", "c"], name="x"),
    ...     columns=pd.Index([1, 2], name="i"),
    ... )
    >>> print(inner_data)  # doctest: +NORMALIZE_WHITESPACE
    i  1  2
    x
    a  1  2
    b  3  4
    c  5  6

    Create a second set of data representing a function with 2-dimensional outputs, representing the "outer" function of the composition. Notice that there is overlap between the index of these data (the domain of the inner function) with the columns of the previous data (the range of the inner funciton.)

    >>> outer_data = pd.DataFrame(
    ...     [(1, 2), (2, 2), (3, 2), (4, 2)],
    ...     index=pd.MultiIndex.from_tuples(
    ...         [(1, 2), (3, 4), (7, 8), (9, 0)], names=["y_0", "y_1"]
    ...     ),
    ...     columns=pd.Index([2, 3], name="j"),
    ... )
    >>> print(outer_data)  # doctest: +NORMALIZE_WHITESPACE
    j        2  3
    y_0 y_1
    1   2    1  2
    3   4    2  2
    7   8    3  2
    9   0    4  2

    Compute the composed data.

    >>> composed_data = compose_funcs(inner_data, outer_data)
    >>> print(composed_data)  # doctest: +NORMALIZE_WHITESPACE
       2  3
    x
    a  1  2
    b  2  2
    """
    import pandas as pd

    from .utils import to_df

    inner_data = to_df(inner_data)
    outer_data = to_df(outer_data)
    original_columns = list(outer_data.columns)
    outer_data = outer_data.add_suffix("_outer")

    data = pd.merge(
        left=inner_data,
        right=outer_data,
        left_on=list(inner_data.columns),
        right_index=True,
    )[list(outer_data.columns)].squeeze(axis=1)

    if isinstance(data, pd.Series):
        data = data.rename()
    else:
        data.columns = original_columns

    return data


def compute_atom_data(self_data: PandasLike, sig_alg_data: PandasLike) -> PandasLike:
    """Compute the atom outputs of a (parametrized) measurable function/vector.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core._utils import compute_atom_data

    Define data for a parametrized domain.

    >>> parametrized_domain_data = pd.MultiIndex.from_product(
    ...     [[0, 1], [0, 1, 2]], names=["theta", "x"]
    ... )

    Define data for a parametrized measurable function.

    >>> self_data = pd.Series([1, 2, 2, 0, -3, -3], index=parametrized_domain_data, name="f")
    >>> print(self_data)  # doctest: +NORMALIZE_WHITESPACE
    theta  x
    0      0    1
           1    2
           2    2
    1      0    0
           1   -3
           2   -3
    Name: f, dtype: int64

    Define data for a sigma-algebra.

    >>> sig_alg_data = pd.Series([0, 1, 2], index=pd.Index([0, 1, 2], name="x"), name="u")
    >>> print(sig_alg_data)  # doctest: +NORMALIZE_WHITESPACE
    x
    0    0
    1    1
    2    2
    Name: u, dtype: int64

    Compute the parametrized atom data.

    >>> print(compute_atom_data(self_data=self_data, sig_alg_data=sig_alg_data))  # doctest: +NORMALIZE_WHITESPACE
    theta  u
    0      0    1
           1    2
           2    2
    1      0    0
           1   -3
           2   -3
    Name: f, dtype: int64
    """
    import pandas as pd

    from .utils import to_df

    if isinstance(self_data, pd.Series):
        original_name = self_data.name
    else:
        original_cols = self_data.columns

    self_data = to_df(self_data)
    sig_alg_data = to_df(sig_alg_data)

    parameter_names = [
        name for name in self_data.index.names if name not in sig_alg_data.index.names
    ]

    if parameter_names:
        atom_data = (
            pd.merge(
                left=self_data, right=sig_alg_data, left_index=True, right_index=True
            )
            .groupby(level=parameter_names)
            .apply(lambda g: g.drop_duplicates().droplevel(parameter_names))
            .set_index(keys=list(sig_alg_data.columns), append=True)
            .droplevel(list(sig_alg_data.index.names))
            .squeeze(axis=1)
        )

    else:
        atom_data = (
            pd.concat([self_data, sig_alg_data], axis=1)
            .drop_duplicates()
            .set_index(list(sig_alg_data.columns))
            .squeeze(axis=1)
        )

    if isinstance(atom_data, pd.Series):
        atom_data.name = original_name
    else:
        atom_data.columns = original_cols

    return atom_data
