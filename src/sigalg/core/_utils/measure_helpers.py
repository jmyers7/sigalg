from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numbers import Real

    import pandas as pd


def restrict_measure_to_sub_sig_alg(
    self_data: pd.Series,
    super_data: pd.Series | pd.DataFrame,
    sub_data: pd.Series | pd.DataFrame,
) -> pd.Series:
    """Restrict the data of a measure to a sub-sigma-algebra.

    Examples
    --------
    >>> import sigalg as sa
    >>> from sigalg.core._utils import restrict_measure_to_sub_sig_alg
    >>> X = sa.Domain.from_sequence(size=5)
    >>> F = sa.SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: (0, 1),
    ...         1: (0, 1),
    ...         2: (1, 1),
    ...         3: (2, 1),
    ...         4: (2, 1),
    ...     },
    ...     variable_names=["u", "v"],
    ... )
    >>> G = sa.SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: (0, 1),
    ...         1: (0, 1),
    ...         2: (0, 1),
    ...         3: (1, 1),
    ...         4: (1, 1),
    ...     },
    ...     name="G",
    ...     variable_names=["a", "b"],
    ... )
    >>> mu = sa.Measure.from_rand(domain=F, random_state=42)
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'mu':
         mu
    u v
    0 1   7
    1 1   1
    2 1   8
    >>> restricted_data = restrict_measure_to_sub_sig_alg(
    ...     self_data=mu.data,
    ...     super_data=F.data,
    ...     sub_data=G.data,
    ... )
    >>> print(restricted_data)  # doctest: +NORMALIZE_WHITESPACE
    a  b
    0  1    8
    1  1    8
    Name: measure, dtype: int64
    """
    import pandas as pd

    from .utils import add_subscript, remove_subscript, to_df

    sub_data = to_df(sub_data).add_suffix("_sub")
    super_data = to_df(super_data).add_suffix("_super")
    self_data = self_data.copy()

    sig_alg_data = pd.concat([sub_data, super_data], axis=1)
    self_data.index.names = add_subscript(self_data.index.names, "super")

    data = (
        pd.merge(left=sig_alg_data, right=self_data.rename("measure").reset_index())
        .drop_duplicates(list(sub_data.columns) + ["measure"])
        .groupby(list(sub_data.columns))["measure"]
        .sum()
    )
    data.index.names = remove_subscript(data.index.names)

    return data


def get_measure_of_set(
    indicator_data: pd.Series,
    sig_alg_data: pd.Series | pd.DataFrame,
    measure_data: pd.Series,
) -> Real:
    """Get the measure of a measurable set in a sigma-algebra.

    Examples
    --------
    >>> import sigalg as sa
    >>> from sigalg.core._utils import get_measure_of_set
    >>> X = sa.Domain.from_sequence(size=6)
    >>> F = sa.SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: (1, 2),
    ...         1: (1, 2),
    ...         2: (0, 2),
    ...         3: (2, 4),
    ...         4: (2, 4),
    ...         5: (2, 4),
    ...     },
    ... )
    >>> mu = sa.Measure(
    ...     domain=F,
    ...     mapping={
    ...         (1, 2): 2,
    ...         (0, 2): 4,
    ...         (2, 4): 6,
    ...     },
    ... )
    >>> U = F.get_set([0, 1, 2], name="U")
    >>> get_measure_of_set(
    ...     indicator_data=U.indicator_data, sig_alg_data=F.data, measure_data=mu.data
    ... )
    6
    """
    from numbers import Real

    import pandas as pd

    from .utils import to_df

    sig_alg_data = to_df(sig_alg_data)

    indicator_atom_data = (
        pd.concat([sig_alg_data, indicator_data], axis=1)
        .drop_duplicates()
        .set_index(list(sig_alg_data.columns))
        .squeeze(axis=1)
    )
    return (measure_data * indicator_atom_data).sum().astype(Real)
