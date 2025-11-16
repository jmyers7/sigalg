from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .event import Event


class SampleSpace(Sequence):

    # --------------------- constructor --------------------- #

    def __init__(self, indices: list[Hashable]) -> None:
        self._validate_parameters(indices)
        self._index = pd.Index(indices)

    # --------------------- properties --------------------- #

    @property
    def index(self) -> pd.Index:
        return self._index

    # --------------------- methods --------------------- #

    def get_event(self, event_indices: list[Hashable]) -> Event:
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of Hashable items.")
        for idx in event_indices:
            if idx not in self._index:
                raise ValueError(f"Index '{idx}' not found in sample space.")
        return self[event_indices]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, key: Hashable | list[Hashable]) -> Hashable | Event:
        from .event import Event

        if isinstance(key, list):
            return Event(sample_space=self, event_indices=key)
        elif isinstance(key, int):
            return self._index[key]
        else:
            if key not in self._index:
                raise KeyError(f"Index '{key}' not found in sample space.")
            return self._index.get_loc(key)

    def __iter__(self) -> iter:
        return iter(self._index)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"SampleSpace({list(self._index)})"

    # --------------------- equality & hashing --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        return isinstance(other, SampleSpace) and self.index.equals(other.index)

    def __hash__(self) -> int:
        if not hasattr(self, "_cached_hash"):
            self._cached_hash = hash(tuple(self._index))
        return self._cached_hash

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(indices: list[Hashable]) -> None:
        if not isinstance(indices, list):
            raise TypeError("Sample space indices must be provided as a list.")
        if len(indices) == 0:
            raise ValueError("Sample space cannot be empty.")
        if len(indices) != len(set(indices)):
            raise ValueError(
                "Sample space indices must be unique (no duplicates allowed)."
            )
        try:
            set(indices)
        except TypeError as e:
            raise TypeError("All sample space indices must be hashable.") from e
