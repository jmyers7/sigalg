"""Create an event space from a sample space and a sigma-algebra."""

from sigalg.core import EventSpace, SampleSpace, SigmaAlgebra

Omega = SampleSpace().from_sequence(size=4)  # (1)!

atom_ids = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
}
F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)  # (2)!

event_space = EventSpace(sample_space=Omega, sigma_algebra=F)  # (3)!

print(event_space.sample_space)
print("\n", event_space.sigma_algebra)  # (4)!

new_atom_ids = {
    0: 0,
    1: 1,
    2: 1,
    3: 0,
}
G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(new_atom_ids)  # (5)!

event_space.sigma_algebra = G  # (6)!

print("\n", event_space.sigma_algebra)
