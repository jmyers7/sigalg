"""Extracting events from a sample space, and performing set operations on them."""

from sigalg.core import SampleSpace

# Create a sample space Omega = [0, 1, 2, 3, 4]
Omega = SampleSpace().from_sequence(size=5)

# Extra two events A and B
A = Omega.get_event([0, 1, 2], name="A")
B = Omega.get_event([2, 3, 4], name="B")

# Perform set operations on events
union = A | B
intersection = A & B
difference = A - B

print(Omega)
print("\n", A)
print("\n", B)
print("\n", union)
print("\n", intersection)
print("\n", difference)
