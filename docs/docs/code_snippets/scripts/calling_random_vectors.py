"""Demonstrate calling a random vector."""

from sigalg.core import RandomVector, SampleSpace

Omega = SampleSpace().from_sequence(size=3)  # (1)!
outputs = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))  # (2)!
X = RandomVector(domain=Omega).from_dict(outputs)
print(X, "\n")

print(f"Call a random vector on a single sample point:\n{X(0)}\n")  # (3)!

A = Omega.get_event([0, 2])  # (4)!
X_A = X(A)
print(f"The restriction of 'X' to the event 'A = [0, 2]':\n{X_A}\n")

B = [0, 1]  # (5)!
X_B = X(B)
print(f"The restriction of 'X' to the event 'B = [0, 1]':\n{X_B}")
