"""Accessing the components of a random vector."""

from sigalg.core import RandomVector, SampleSpace

Omega = SampleSpace().from_sequence(size=2)  # (1)!
outputs = dict(zip(Omega, [(1, 2, 3), (4, 5, 6)]))  # (2)!
X = RandomVector(domain=Omega).from_dict(outputs)
print(X, "\n")

X_1 = X.get_component_rv("X_1")  # (3)!
print(X_1, "\n")

X_sub = X.get_sub_vector(["X_0", "X_2"])  # (4)!
print(X_sub)
