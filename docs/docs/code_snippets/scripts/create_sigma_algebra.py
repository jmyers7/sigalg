"""Create a sigma-algebra from a dictionary of atom IDs."""

from sigalg.core import SampleSpace, SigmaAlgebra

Omega = SampleSpace().from_sequence(size=5)  # (1)!

sample_id_to_atom_id = {  # (2)!
    0: 0,
    1: 1,
    2: 0,
    3: 1,
    4: 2,
}

F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id)  # (3)!

print(F)  # (4)!
