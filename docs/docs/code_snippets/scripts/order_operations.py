"""Perform order-theoretic operations on events."""

from sigalg.core import SampleSpace

Omega = SampleSpace().from_sequence(size=5)  # (1)!

A = Omega.get_event([0, 1, 2], name="A")
B = Omega.get_event([0, 1, 2, 3], name="B")
C = Omega.get_event([0, 1, 3], name="C")

print(A)
print("\n", B)
print("\n", C)
print("\nIs A a subset of B?", A <= B)
print("Is A a subset of C?", A <= C)
print("Is B a superset of A?", B >= A)
print("Is C a superset of A?", C >= A)
