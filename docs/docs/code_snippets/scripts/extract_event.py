"""Extracting events from a sample space."""

from sigalg.core import SampleSpace

# Create a sample space Omega = [omega_1, omega_2, omega_3, omega_4, omega_5]
Omega = SampleSpace().from_sequence(size=5, initial_index=1, prefix="omega")

# Extract an event A using the `get_event` method
A = Omega.get_event(["omega_1", "omega_2", "omega_3"], name="A")

# Extract an event B by (positional-based) slicing
B = Omega[2:5, "B"]

# Extract an event C by (positional-based) indexing
C = Omega[[0, 3], "C"]

print(Omega)
print("\n", A)
print("\n", B)
print("\n", C)
