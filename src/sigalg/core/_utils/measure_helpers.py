from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numbers import Real

    import pandas as pd


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
