"""Marker class for a 1-dimensional random vector."""

from .random_vector import RandomVector


class RandomVariable(RandomVector):
    """Marker class for a 1-dimensional random vector."""

    _repr_name = "Random variable"

    def to_random_vector(self) -> RandomVector:
        """Pass."""
        from ..indices.index import Index

        self.__class__ = RandomVector
        self._data = self.data.to_frame()
        self._index = Index(indices=self.data.columns)

        return self
