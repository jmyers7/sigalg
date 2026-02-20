"""Event spaces inherit methods from SampleSpace and SigmaAlgebra."""

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

A = event_space.get_event([0, 3])  # (4)!
B = event_space.get_event([0, 1, 2])

print("Is the event A measurable?", event_space.is_measurable(A))  # (5)!
print("\nIs the event B measurable?", event_space.is_measurable(B))
