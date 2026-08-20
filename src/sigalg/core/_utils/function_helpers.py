from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable

    import pandas as pd

    PandasLike = pd.Series | pd.DataFrame


def ascend_from_atom_space(
    self_data: PandasLike,
    sig_alg_data: PandasLike,
    parameter_names: list[Hashable] | None = None,
) -> PandasLike:
    """Convert (parametrized) function data defined on atom identifiers of a sigma-algebra into (parametrized) function data defined on the measurable domain.

    * If `self_data` carries a multi-index, the level names are assumed to be equal to either `parameter_names + sig_alg_data.name` or `parameter_names + list(sig_alg_data.columns)`, in that order, depending on whether `sig_alg_data` is a `pd.Series` or `pd.DataFrame`.

    * If `self_data` carries a plain index, it is assumed `sig_alg_data` is a `pd.Series` and the name of the index is `sig_alg_data.name` and `parameter_names` is empty.

    Examples
    --------
    >>> import sigalg as sa
    >>> from sigalg.core._utils import sig_alg_func_to_measurable_func

    Generate a function with 2-dimensional outputs defined on the atom space of a sigma-algebra with 2-dimensional atom identifiers.

    >>> X = sa.Domain(
    ...     [("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")], variable_names=["x_0", "x_1"]
    ... )
    >>> F = sa.SigmaAlgebra(domain=X, mapping=dict(zip(X, [(0, 1), (0, 1), (1, 2), (2, 3)])))
    >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
    i        0  1
    x_0 x_1
    a   b    0  1
    c   d    0  1
    e   f    1  2
    g   h    2  3
    >>> I = sa.Index([1, 2])
    >>> f = sa.Function(
    ...     domain=F.atom_space,
    ...     mapping=dict(zip(F.atom_space, [(3, 4), (5, 6), (7, 8)])),
    ...     index=I,
    ... )
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Function 'f':
    i        1  2
    F_0 F_1
    0   1    3  4
    1   2    5  6
    2   3    7  8

    Convert to the data of a measurable vector on the domain.

    >>> data = sig_alg_func_to_measurable_func(
    ...     self_data=f.data,
    ...     sig_alg_data=F.data,
    ... )
    >>> print(data)  # doctest: +NORMALIZE_WHITESPACE
    i        1  2
    x_0 x_1
    a   b    3  4
    c   d    3  4
    e   f    5  6
    g   h    7  8

    Create a parametrized function defined on the atom space of the sigma-algebra.

    >>> Theta = sa.Domain.cartesian_power(
    ...     [0, 1], n=2, variable_names=["theta_0", "theta_1"], name="Theta"
    ... )
    >>> g = sa.Function(
    ...     domain=Theta @ F.atom_space,
    ...     mapping={
    ...         (0, 0, 0, 1): 1,
    ...         (0, 0, 1, 2): 2,
    ...         (0, 0, 2, 3): 3,
    ...         (0, 1, 0, 1): 4,
    ...         (0, 1, 1, 2): 5,
    ...         (0, 1, 2, 3): 6,
    ...         (1, 0, 0, 1): 7,
    ...         (1, 0, 1, 2): 8,
    ...         (1, 0, 2, 3): 9,
    ...         (1, 1, 0, 1): 10,
    ...         (1, 1, 1, 2): 11,
    ...         (1, 1, 2, 3): 12,
    ...     },
    ...     name="g",
    ... )
    >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
    Function 'g':
                              g
    theta_0 theta_1 F_0 F_1
    0       0       0   1     1
                    1   2     2
                    2   3     3
            1       0   1     4
                    1   2     5
                    2   3     6
    1       0       0   1     7
                    1   2     8
                    2   3     9
            1       0   1    10
                    1   2    11
                    2   3    12

    Convert to the data of a parametrized measurable function on the domain.

    >>> data = sig_alg_func_to_measurable_func(
    ...     self_data=g.data,
    ...     sig_alg_data=F.data,
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

    if parameter_names is None:
        parameter_names = []

    sig_alg_data = to_df(sig_alg_data)
    self_data = to_df(self_data)
    self_index = self_data.columns

    sig_alg_variable_names = [
        name for name in self_data.index.names if name not in parameter_names
    ]
    domain_variable_names = sig_alg_data.index.names

    sig_alg_data.columns = add_subscript(sig_alg_variable_names, "ID")

    self_data.index.names = [
        f"{name}_param" if name in parameter_names else f"{name}_ID"
        for name in self_data.index.names
    ]

    data = (
        pd.merge(left=sig_alg_data.reset_index(), right=self_data.reset_index())
        .set_index(add_subscript(parameter_names, "param") + domain_variable_names)[
            self_index
        ]
        .sort_index()
        .squeeze(axis=1)
    )

    data.index.names = [
        f"{name}_0" if name in domain_variable_names else name
        for name in parameter_names
    ] + domain_variable_names

    if isinstance(data, pd.DataFrame):
        data.columns = self_index
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


def compute_expectation(
    rv_atom_data: PandasLike,
    given_data: PandasLike,
    given_variable_names: list[Hashable],
    atom_data: PandasLike,
    measure_data: pd.Series,
    measure_data_on_given: pd.Series,
) -> PandasLike:
    """Compute the data for a conditional expectation.

    Examples
    --------
    >>> import numpy as np
    >>> import sigalg as sa
    >>> from sigalg.core._utils import compute_expectation
    >>> rng = np.random.default_rng(42)
    >>> Omega = sa.SampleSpace.from_sequence(size=10)
    >>> F = sa.SigmaAlgebra.from_rand(domain=Omega, num_atoms=5, random_state=rng)
    >>> G = sa.SigmaAlgebra.from_rand(super=F, num_atoms=3, random_state=rng, name="G")
    >>> P = sa.ProbabilityMeasure.from_rand(domain=F, num_null_atoms=2, random_state=rng)
    >>> X = sa.RandomVariable.from_rand(
    ...     domain=Omega, sig_alg=F, measure=P, diff_values=1, random_state=rng
    ... )
    >>> exp = compute_expectation(
    ...     rv_atom_data=X.atom_data(),
    ...     given_data=G.data,
    ...     given_variable_names=G.variable_names,
    ...     atom_data=G.up_lattice.get_atom_data(F),
    ...     measure_data=P.data,
    ...     measure_data_on_given=(P | G).data,
    ... )
    >>> print(exp)  # doctest: +NORMALIZE_WHITESPACE
    s
    0    4.0
    1    4.0
    2    1.0
    3    4.0
    4    6.0
    5    6.0
    6    4.0
    7    4.0
    8    4.0
    9    4.0
    dtype: float64
    """
    import pandas as pd

    from .utils import add_subscript, to_df

    rv_atom_data = to_df(rv_atom_data)
    rv_cols = list(rv_atom_data.columns)

    atom_data = to_df(atom_data)
    sig_alg_cols = add_subscript(given_variable_names, "ID")
    atom_data.columns = sig_alg_cols

    measure_data_on_given = measure_data_on_given.rename("measure")
    measure_data_on_given.index.names = sig_alg_cols

    given_data = to_df(given_data)
    given_data.columns = sig_alg_cols

    rv_times_prob = rv_atom_data.multiply(measure_data, axis=0)
    combined_data = pd.concat([rv_times_prob, atom_data], axis=1)

    merged_data = pd.merge(
        left=combined_data, right=measure_data_on_given.reset_index()
    )
    merged_data[rv_cols] = (
        merged_data[rv_cols].divide(merged_data["measure"], axis=0).fillna(0.0)
    )
    merged_data.drop(columns="measure", inplace=True)
    merged_data = merged_data.groupby(sig_alg_cols).sum()

    merged_data.index.names = given_variable_names

    return ascend_from_atom_space(
        self_data=merged_data,
        sig_alg_data=given_data,
    )


def compute_radon_nikodym(
    measure_data: pd.Series,
    base_measure_data: pd.Series,
    sig_alg_data: pd.Series | pd.DataFrame,
) -> pd.Series:
    """Pass."""
    data = (measure_data / base_measure_data).fillna(0.0)
    data.name = None
    return ascend_from_atom_space(self_data=data, sig_alg_data=sig_alg_data)
