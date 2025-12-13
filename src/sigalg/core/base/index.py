from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd


class Index:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        indices: list[Hashable] | None = None,
        values: pd.Index | None = None,
        name: str | None = None,
        values_name: str | None = None,
        **kwargs,
    ) -> None:
        self._validate_parameters(
            indices=indices, values=values, name=name, values_name=values_name
        )

        if values is not None:
            self.values = values
            self.indices = values.to_list()
            self.values_name = values.name
        elif indices is not None:
            self.values = pd.Index(data=indices, name=values_name)
            self.indices = indices
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

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
    def _validate_parameters(
        indices: list[Hashable] | None = None,
        values: pd.Index | None = None,
        name: str | None = None,
        values_name: str | None = None,
    ) -> None:
        if indices is not None and values is not None:
            raise ValueError("Cannot specify both 'indices' and 'values'.")
        if indices is None and values is None:
            raise ValueError("Must specify either 'indices' or 'values'.")
        if indices is not None:
            if not isinstance(indices, list):
                raise TypeError("indices must be a list of Hashable items.")
            for idx in indices:
                if not isinstance(idx, Hashable):
                    raise TypeError("All items in 'indices' must be Hashable.")
            if len(indices) != len(set(indices)):
                raise ValueError("All items in 'indices' must be unique.")
        if values is not None:
            if not isinstance(values, pd.Index):
                raise TypeError("values must be a pandas Index.")
            if len(values) != values.nunique():
                raise ValueError("All items in 'values' must be unique.")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string.")
        if values_name is not None and not isinstance(values_name, str):
            raise TypeError("values_name must be a string.")
