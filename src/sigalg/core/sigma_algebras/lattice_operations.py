"""Module for lattice operations on sigma algebras.

Functions
---------
join
    Compute the join (least upper bound) of a list of sigma algebras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .sigma_algebra import SigmaAlgebra


def join(sigma_algebras: list[SigmaAlgebra]) -> SigmaAlgebra:
    """Compute the join (least upper bound) of a list of sigma algebras.

    Parameters
    ----------
    sigma_algebras : list[SigmaAlgebra]
        A list of SigmaAlgebra instances to join.

    Raises
    ------
    TypeError
        If the input is not a list of SigmaAlgebra instances.
    ValueError
        If the list is empty or if the SigmaAlgebra instances do not share the same sample space.
    """
    from .sigma_algebra import SigmaAlgebra

    if not isinstance(sigma_algebras, list):
        raise TypeError("Expected a list of SigmaAlgebra instances")
    if not all(isinstance(alg, SigmaAlgebra) for alg in sigma_algebras):
        raise TypeError("All elements of the list must be SigmaAlgebra instances")
    if len(sigma_algebras) == 0:
        raise ValueError(
            "The meet of an empty list of sigma algebras is the trivial algebra on the sample space"
        )
    if len(sigma_algebras) == 1:
        return sigma_algebras[0]
    sample_space = sigma_algebras[0].sample_space
    if not all(alg.sample_space == sample_space for alg in sigma_algebras):
        raise ValueError("All SigmaAlgebra instances must have the same sample space")

    for alg in sigma_algebras:
        alg.data.rename(alg.name, inplace=True)
    df = pd.concat([alg.data for alg in sigma_algebras], axis=1)

    sample_id_to_atom_id = {}
    for id, (_, grp) in enumerate(df.groupby(list(df.columns))):
        atom = grp.index.to_list()
        for sample_id in atom:
            sample_id_to_atom_id[sample_id] = id

    return SigmaAlgebra(
        sample_id_to_atom_id=sample_id_to_atom_id,
        sample_space=sample_space,
        name="join",
    )
