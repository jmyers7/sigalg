from __future__ import annotations

from collections.abc import Hashable

import pandas as pd


class FeatureIndex:

    # --------------------- constructor --------------------- #

    def __init__(self, indices: list[Hashable], values_name: str = "feature") -> None:
        self._validate_parameters(indices, values_name)
        self._values = pd.Index(data=indices, name=values_name)

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Index:
        return self._values.copy()

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

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> iter:
        return iter(self._values)

    # --------------------- equality --------------------- #

    def __eq__(self, other: FeatureIndex) -> bool:
        return isinstance(other, FeatureIndex) and self.values.equals(other.values)

    # --------------------- validation methods --------------------- #

    def _validate_parameters(self, indices: list[Hashable], values_name: str) -> None:
        if not isinstance(indices, list) or not all(
            isinstance(idx, Hashable) for idx in indices
        ):
            raise TypeError("indices must be provided as a list.")
        if not isinstance(values_name, str):
            raise TypeError("values_name must be a string.")
        if len(indices) == 0:
            raise ValueError("indices list cannot be empty.")
        if len(indices) != len(set(indices)):
            raise ValueError("indices must be unique (no duplicates allowed).")
