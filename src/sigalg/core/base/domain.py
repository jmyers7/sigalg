"""Empty marker class for function domains."""

from collections.abc import Hashable

from .index import Index


class Domain(Index):
    """Empty marker class for function domains."""

    def __init__(self, name: Hashable = "D") -> None:
        super().__init__(name=name)
