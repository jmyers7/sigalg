"""Pass."""

from collections.abc import Hashable

from .index import Index


class Domain(Index):
    """Pass."""

    def __init__(self, name: Hashable | None = "D") -> None:
        super().__init__(name=name)
