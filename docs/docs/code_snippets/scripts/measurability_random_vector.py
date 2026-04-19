"""Check measurability of random vectors."""

from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra

Omega = SampleSpace().from_sequence(size=4)  # (1)!

outputs_X = dict(zip(Omega, [(1, 2), (3, 4), (3, 4), (3, 4)]))  # (2)!
outputs_Y = dict(zip(Omega, [(1, 2), (3, 4), (5, 6), (7, 8)]))
X = RandomVector(domain=Omega, name="X").from_dict(outputs_X)
Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)

atom_ids = dict(zip(Omega, [0, 1, 1, 2]))  # (3)!
F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)

print(f"Is 'X' F-measurable? {X.is_measurable(F)}")  # (4)!
print(f"Is 'Y' F-measurable? {Y.is_measurable(F)}")
