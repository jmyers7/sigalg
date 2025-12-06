from __future__ import annotations

from collections.abc import Hashable

import pandas as pd


class FeatureIndex:

    # --------------------- constructor --------------------- #

    def __init__(self, indices: list[Hashable], name: str = "X") -> None:
        self._validate_parameters(indices, name)
        self._values = pd.Index(data=indices, name="features")
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Index:
        return self._values.copy()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    # --------------------- factory methods --------------------- #

    @classmethod
    def generate_default(
        cls,
        initial_index: int = 0,
        size: int = 10,
        prefix: str = "X",
        name: str = "X",
    ) -> FeatureIndex:
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")

        if size == 1:
            indices = [prefix]
        else:
            indices = [
                f"{prefix}{i}" for i in range(initial_index, initial_index + size)
            ]
        return cls(indices=indices, name=name)

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> iter:
        return iter(self._values)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Feature index {self.name}:\n{self._values.to_list()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: FeatureIndex) -> bool:
        return (
            isinstance(other, FeatureIndex)
            and self.values.equals(other.values)
            and self.name == other.name
        )

    # --------------------- validation methods --------------------- #

    def _validate_parameters(self, indices: list[Hashable], name: str) -> None:
        if not isinstance(indices, list) or not all(
            isinstance(idx, Hashable) for idx in indices
        ):
            raise TypeError("indices must be provided as a list.")
        if len(indices) == 0:
            raise ValueError("indices list cannot be empty.")
        if len(indices) != len(set(indices)):
            raise ValueError("indices must be unique (no duplicates allowed).")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
