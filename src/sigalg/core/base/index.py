from collections.abc import Hashable
from typing import Any

import pandas as pd


class Index:

    # --------------------- constructor --------------------- #

    def __init__(
        self, indices: list[Hashable], values_name: str | None = None, **kwargs
    ) -> None:
        self.values = pd.Index(data=indices, name=values_name)

    # --------------------- properties --------------------- #

    @property
    def values_name(self) -> str:
        return self.values.name

    @values_name.setter
    def values_name(self, values_name: str) -> None:
        if not isinstance(values_name, str):
            raise TypeError("values_name must be a string.")
        self.values.name = values_name

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
