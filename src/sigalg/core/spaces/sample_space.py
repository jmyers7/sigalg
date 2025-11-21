from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from . import ProbabilitySpace
    from .event import Event


class SampleSpace:

    # --------------------- constructor --------------------- #

    def __init__(self, indices: list[Hashable]) -> None:
        self._validate_parameters(indices)
        self._values = pd.Index(indices)

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Index:
        return self._values.copy()

    # --------------------- data access methods --------------------- #

    def get_event(self, event_indices: list[Hashable]) -> Event:
        from .event import Event

        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of Hashable items.")
        for idx in event_indices:
            if idx not in self._values:
                raise ValueError(f"Index '{idx}' not found in sample space.")
        return Event(sample_space=self, event_indices=event_indices)

    def get_event_at(self, event_positions: list[int] | slice) -> Event:
        from .event import Event

        if not isinstance(event_positions, (list, slice)):
            raise TypeError("event_positions must be a list of integers or a slice.")
        event_indices = self._values[event_positions].tolist()
        for idx in event_indices:
            if idx not in self._values:
                raise ValueError(f"Index '{idx}' not found in sample space.")
        return Event(sample_space=self, event_indices=event_indices)

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> iter:
        return iter(self._values)

    # --------------------- probability methods --------------------- #

    def add_probability_measure(
        self, probabilities: dict[Hashable, Real]
    ) -> ProbabilitySpace:
        from . import ProbabilitySpace

        return ProbabilitySpace(sample_space=self, probabilities=probabilities)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"SampleSpace({list(self._values)})"

    # --------------------- equality & hashing --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        return isinstance(other, SampleSpace) and self.values.equals(other.values)

    def __hash__(self) -> int:
        if not hasattr(self, "_cached_hash"):
            self._cached_hash = hash(tuple(self._values))
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


class SampleSpaceMethods:
    # --------------------- methods --------------------- #

    def get_event(self, event_indices: list[Hashable]) -> Event:
        return self.sample_space.get_event(event_indices)

    def get_event_at(self, event_positions: list[int] | slice) -> Event:
        return self.sample_space.get_event_at(event_positions)

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self.sample_space)
