"""Marker class for a 1-dimensional random vector."""

from .random_vector import RandomVector


class RandomVariable(RandomVector):
    """Marker class for a 1-dimensional random vector."""

    _repr_name = "Random variable"
