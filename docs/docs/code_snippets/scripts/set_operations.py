"""Perform set-theoretic operations on events."""

from sigalg.core import SampleSpace

Omega = SampleSpace().from_sequence(size=5)  # (1)!

A = Omega.get_event([0, 1, 2], name="A")
B = Omega.get_event([2, 3, 4], name="B")

intersection = A & B
union = A | B
difference = A - B
complement = ~A

print(A)
print("\n", B)
print("\n", intersection)
print("\n", union)
print("\n", difference)
print("\n", complement)
