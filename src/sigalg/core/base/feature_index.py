from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from .index import Index


class FeatureIndex(Index):

    # --------------------- constructor --------------------- #

    def __init__(self, indices: list[Hashable], values_name: str = "feature") -> None:
        super().__init__(indices=indices, values_name=values_name)

    # --------------------- factory methods --------------------- #

    @classmethod
    def generate_default(
        cls,
        initial_index: int = 0,
        size: int = 10,
        prefix: str = "X",
        values_name: str = "feature",
    ) -> FeatureIndex:
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if not isinstance(values_name, str):
            raise TypeError("'values_name' must be a string.")
        if not isinstance(prefix, str):
            raise TypeError("'prefix' must be a string.")

        if size == 1:
            indices = [prefix]
        else:
            indices = [
                f"{prefix}{i}" for i in range(initial_index, initial_index + size)
            ]
        return cls(indices=indices, values_name=values_name)

    # --------------------- data access methods --------------------- #

    def _getitem_hook(self, key: Any) -> FeatureIndex:
        result = self.values[key].to_list()
        return FeatureIndex(indices=result, values_name=self.values_name)

    # --------------------- equality --------------------- #

    def __eq__(self, other: FeatureIndex) -> bool:
        return isinstance(other, FeatureIndex) and super().__eq__(other)
