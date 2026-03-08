"""Module for comparing and combining sigma algebras."""
from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .sigma_algebra import SigmaAlgebra

# TODO: Update docstrings
def is_subalgebra(sub_algebra: SigmaAlgebra, super_algebra: SigmaAlgebra) -> bool:
    """Check if one sigma algebra is a subalgebra of another.

    Parameters
    ----------
    sub_algebra : SigmaAlgebra
        The candidate subalgebra.
    super_algebra : SigmaAlgebra
        The candidate superalgebra.

    Returns
    -------
    is_subalgebra : bool
        True if `sub_algebra` is a subalgebra of `super_algebra`, False otherwise.
    """
    sub_atoms = sub_algebra.atom_id_to_event.values()
    super_atoms = super_algebra.atom_id_to_event.values()
    for super_atom in super_atoms:
        if not any(super_atom <= sub_atom for sub_atom in sub_atoms):
            return False
    return True

# TODO: Update docstrings
def is_refinement(coarser_algebra: SigmaAlgebra, finer_algebra: SigmaAlgebra) -> bool:
    """Check if one sigma algebra is a refinement of another.

    Parameters
    ----------
    coarser_algebra : SigmaAlgebra
        The candidate coarser algebra.
    finer_algebra : SigmaAlgebra
        The candidate finer algebra.

    Returns
    -------
    is_refinement : bool
        True if `finer_algebra` is a refinement of `coarser_algebra`, False otherwise.
    """
    return is_subalgebra(sub_algebra=coarser_algebra, super_algebra=finer_algebra)

# TODO: Update docstrings
def join(
    sigma_algebras: list[SigmaAlgebra], name: Hashable | None = "join"
) -> SigmaAlgebra:
    """Compute the join (least upper bound) of a list of sigma algebras.

    Parameters
    ----------
    sigma_algebras : list[SigmaAlgebra]
        A list of SigmaAlgebra instances to join.
    name : Hashable | None, default="join"
        Name identifier for the resulting sigma algebra.

    Raises
    ------
    TypeError
        If the input is not a list of SigmaAlgebra instances.
    ValueError
        If the list is empty or if the SigmaAlgebra instances do not share the same sample space.
    """
    from .sigma_algebra import SigmaAlgebra

    if name is not None and not isinstance(name, Hashable):
        raise TypeError("name must be a Hashable or None")
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

    sample_id_to_atom_id = df.apply(lambda row: tuple(row), axis=1).to_dict()

    return SigmaAlgebra(sample_space=sample_space, name=name).from_dict(
        sample_id_to_atom_id
    )
