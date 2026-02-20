"""Extracting events from a sample space."""

from sigalg.core import SampleSpace

Omega = SampleSpace().from_sequence(size=5, initial_index=1, prefix="omega")  # (1)!

A = Omega.get_event(["omega_1", "omega_2", "omega_3"], name="A")  # (2)!
B = Omega[2:5, "B"]  # (3)!
C = Omega[[0, 3], "C"]  # (4)!

print(Omega)
print("\n", A)
print("\n", B)
print("\n", C)
