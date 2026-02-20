"""Accessing the underlying data in a sample space."""

from sigalg.core import SampleSpace

Omega = SampleSpace().from_sequence(size=5, prefix="s")  # (1)!

data = Omega.data  # (2)!

print(Omega)
print("\nThe `data` attribute of the sample space:\n", Omega.data)
