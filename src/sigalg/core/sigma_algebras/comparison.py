from __future__ import annotations  # noqa: D100

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sigma_algebra import SigmaAlgebra


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
