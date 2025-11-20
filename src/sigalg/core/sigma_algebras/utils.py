from .sigma_algebra import SigmaAlgebra


def is_sub_algebra(sub: SigmaAlgebra, super: SigmaAlgebra) -> bool:
    sub_atoms = sub.to_events().values()
    super_atoms = super.to_events().values()
    for super_atom in super_atoms:
        if not any(super_atom <= sub_atom for sub_atom in sub_atoms):
            return False
    return True
