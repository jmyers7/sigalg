"""Demonstrate algebra and functions of random vectors."""

from numbers import Real

import numpy as np

from sigalg.core import FeatureVector, RandomVector, SampleSpace


def pointwise_func(rv: RandomVector) -> RandomVector:  # (1)!
    """Add 2 to each entry of a random vector."""
    return rv + 2


def feature_func(vec: FeatureVector) -> Real:  # (2)!
    """Sum the entries of a feature vector and add 1."""
    return vec.sum() + 1


Omega = SampleSpace().from_sequence(size=2)  # (3)!
X = RandomVector(domain=Omega).from_dict(
    {
        0: (1, 2),
        1: (0, -1),
    }
)
Y = RandomVector(domain=Omega, name="Y").from_dict(
    {
        0: (2, 1),
        1: (2, 3),
    }
)
Z = RandomVector(domain=Omega, name="Z").from_dict(
    {
        0: (0, np.pi),
        1: (np.pi / 2, 3 * np.pi / 2),
    }
)

print(X, "\n")
print(Y, "\n")
print(Z, "\n")

print(X + Y, "\n")  # (4)!
print(X - Y, "\n")
print(X * Y, "\n")
print(X / Y, "\n")
print(X**Y, "\n")
print(2 * X - 3 * Y, "\n")

print(pointwise_func(X), "\n")  # (5)!
print(X.apply_to_features(feature_func), "\n")

print(np.sin(Z).round())  # (6)!
