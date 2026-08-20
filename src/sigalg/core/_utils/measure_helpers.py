from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def reindex_measure(
    measure_data: pd.Series, atom_data: pd.Series | pd.DataFrame
) -> pd.Series:
    """Reindex the data of a measure given the data of an equal sigma-algebra.

    Examples
    --------
    >>> import sigalg as sa
    >>> from sigalg.core._utils import reindex_measure
    >>> Omega = sa.SampleSpace.from_sequence(size=7)
    >>> F = sa.SigmaAlgebra(
    ...     domain=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 2,
    ...         3: 2,
    ...         4: 3,
    ...         5: 4,
    ...         6: 4,
    ...     },
    ... )
    >>> G = sa.SigmaAlgebra(
    ...     domain=Omega,
    ...     mapping={
    ...         0: "a",
    ...         1: "b",
    ...         2: "c",
    ...         3: "c",
    ...         4: "d",
    ...         5: "e",
    ...         6: "e",
    ...     },
    ...     name="G",
    ... )
    >>> mu = sa.Measure(
    ...     domain=G,
    ...     mapping={
    ...         "a": 1,
    ...         "b": 2,
    ...         "c": 3,
    ...         "d": 4,
    ...         "e": 0,
    ...     },
    ... )
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'mu':
        mu
    G
    a   1
    b   2
    c   3
    d   4
    e   0
    >>> atom_data = G.down_lattice.get_atom_data(F)
    >>> print(atom_data)  # doctest: +NORMALIZE_WHITESPACE
    G
    a    0
    b    1
    c    2
    d    3
    e    4
    Name: F, dtype: int64
    >>> new_data = reindex_measure(measure_data=mu.data, atom_data=atom_data)
    >>> print(new_data)  # doctest: +NORMALIZE_WHITESPACE
    F
    0    1
    1    2
    2    3
    3    4
    4    0
    Name: mu, dtype: int64
    """
    import pandas as pd

    from .utils import to_df

    atom_data = to_df(atom_data)

    data = (
        pd.concat([atom_data, measure_data], axis=1)
        .set_index(list(atom_data.columns))
        .squeeze(axis=1)
    )

    return data


def compute_conditional_prob_measure(
    measure_data,
    restricted_measure_data,
    atom_data,
    given_data,
    given_variable_names,
    return_raw_data: bool = False,
    ascend: bool = False,
):
    """Pass."""
    import pandas as pd

    from .function_helpers import ascend_from_atom_space
    from .utils import to_df

    restricted_measure_name = restricted_measure_data.name
    restricted_measure_data = to_df(restricted_measure_data).copy()
    restricted_measure_data.index.names = given_variable_names

    atom_data = to_df(atom_data)
    atom_data.columns = given_variable_names

    cross_data = pd.merge(
        left=restricted_measure_data.reset_index(),
        right=measure_data.index.to_frame(),
        how="cross",
    )

    self_and_sub_data = pd.merge(
        left=measure_data.reset_index(), right=atom_data.reset_index()
    )

    prob_data = pd.merge(
        left=self_and_sub_data, right=restricted_measure_data.reset_index()
    )
    prob_data["probs"] = (
        prob_data[measure_data.name] / prob_data[restricted_measure_name]
    )
    prob_data = prob_data[given_variable_names + measure_data.index.names + ["probs"]]

    data = pd.merge(left=cross_data, right=prob_data, how="outer").set_index(
        given_variable_names + measure_data.index.names
    )

    if return_raw_data:
        return data.rename(columns={restricted_measure_name: "restricted_probs"})

    mask = data["probs"].isna() & (data[restricted_measure_name] < 1e-10)
    data.loc[mask, "probs"] = 1 / len(measure_data)
    data = data.fillna(0.0, inplace=True)["probs"].sort_index()

    if ascend:
        data = ascend_from_atom_space(
            self_data=data.reorder_levels(
                measure_data.index.names + given_variable_names
            ),
            sig_alg_data=given_data,
            parameter_names=measure_data.index.names,
        )

        data = data.reorder_levels(
            given_data.index.names + measure_data.index.names
        ).sort_index()

    return data
