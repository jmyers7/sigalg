from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd


class Index:

    # --------------------- constructor --------------------- #

    def __init__(self, indices: list[Hashable], values_name: str, **kwargs) -> None:
        self._validate_parameters(indices, values_name)
        self.values = pd.Index(data=indices, name=values_name)

    # --------------------- data access methods --------------------- #

    def __getitem__(self, key: Any) -> Any:
        return self._getitem_hook(key=key)

    def _getitem_hook(self, key: Any) -> Any:
        return self.values[key]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> iter:
        return iter(self.values)

    # --------------------- equality --------------------- #

    def __eq__(self, other: Index) -> bool:
        return isinstance(other, Index) and self.values.equals(other.values)

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(indices: list[Hashable], values_name: str) -> None:
        if not isinstance(indices, list):
            raise TypeError("indices must be a list of Hashable items.")
        for idx in indices:
            if not isinstance(idx, Hashable):
                raise TypeError("All indices must be Hashable items.")
        if len(indices) != len(set(indices)):
            raise ValueError("Index indices must be unique.")
        if not isinstance(values_name, str):
            raise TypeError("values_name must be a string.")
