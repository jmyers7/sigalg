from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sigma_algebra import SigmaAlgebra


def is_subalgebra(sub_algebra: SigmaAlgebra, super_algebra: SigmaAlgebra) -> bool:
    sub_atoms = sub_algebra.atom_id_to_event.values()
    super_atoms = super_algebra.atom_id_to_event.values()
    for super_atom in super_atoms:
        if not any(super_atom <= sub_atom for sub_atom in sub_atoms):
            return False
    return True


def is_refinement(coarser_algebra: SigmaAlgebra, finer_algebra: SigmaAlgebra) -> bool:
    return is_subalgebra(sub_algebra=coarser_algebra, super_algebra=finer_algebra)
