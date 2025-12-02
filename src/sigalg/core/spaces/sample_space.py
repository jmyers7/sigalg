from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from . import ProbabilitySpace
    from .event import Event


class SampleSpace:

    # --------------------- constructor --------------------- #

    def __init__(self, indices: list[Hashable], name: str = "Omega") -> None:
        self._validate_parameters(indices, name)
        self._values = pd.Index(data=indices, name=name)
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Index:
        return self._values.copy()

    @property
    def name(self) -> str:
        return self._name

    # --------------------- setter methods --------------------- #

    @name.setter
    def name(self, name: str) -> None:
        self._validate_parameters(self._values.tolist(), name)
        self._name = name
        self._values.name = name

    # --------------------- data access methods --------------------- #

    def get_event(self, event_indices: list[Hashable], name: str = "A") -> Event:
        from .event import Event

        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of Hashable items.")
        for idx in event_indices:
            if idx not in self._values:
                raise ValueError(f"Index '{idx}' not found in sample space.")
        return Event(sample_space=self, event_indices=event_indices, name=name)

    @property
    def get_event_at(self):
        return self._EventIndexer(self)

    class _EventIndexer:
        def __init__(self, sample_space) -> None:
            self._sample_space = sample_space

        def __getitem__(self, key) -> Event:
            from .event import Event

            if isinstance(key, tuple) and len(key) == 2:
                index_key, name = key
            else:
                index_key = key
                name = "A"

            if isinstance(index_key, (int, slice)):
                event_indices = self._sample_space._values[index_key]
                if isinstance(event_indices, pd.Index):
                    event_indices = event_indices.tolist()
                else:
                    event_indices = [event_indices]
            elif isinstance(index_key, list):
                event_indices = self._sample_space._values[index_key].tolist()
            else:
                raise TypeError("Index must be an integer, list of integers, or slice.")

            for idx in event_indices:
                if idx not in self._sample_space._values:
                    raise ValueError(f"Index '{idx}' not found in sample space.")

            return Event(
                sample_space=self._sample_space, event_indices=event_indices, name=name
            )

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> iter:
        return iter(self._values)

    # --------------------- probability methods --------------------- #

    def add_probability_measure(
        self,
        probability_measure: ProbabilityMeasure | None = None,
        probabilities: dict[Hashable, Real] | None = None,
    ) -> ProbabilitySpace:
        from . import ProbabilitySpace

        return ProbabilitySpace(
            sample_space=self,
            probability_measure=probability_measure,
            probabilities=probabilities,
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Sample space {self.name}:\n{self._values.to_list()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        return (
            isinstance(other, SampleSpace)
            and self.values.equals(other.values)
            and self.name == other.name
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(indices: list[Hashable], name: str) -> None:
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
        if not isinstance(name, str):
            raise ValueError("'name' must be a string.")


class SampleSpaceMethods:
    def get_event(self, event_indices: list[Hashable], name: str = "A") -> Event:
        return self.sample_space.get_event(event_indices, name)

    @property
    def get_event_at(self) -> Event:
        return self.sample_space.get_event_at
