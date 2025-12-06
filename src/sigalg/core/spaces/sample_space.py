from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from . import ProbabilitySpace
    from .event import Event
    from .event_space import EventSpace


class SampleSpace:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        indices: list[Hashable],
        name: str = "Omega",
        values_name: str = "sample",
    ) -> None:
        self._validate_parameters(indices, name=name, values_name=values_name)
        self._values = pd.Index(data=indices, name=values_name)
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
        prefix: str = "omega",
        name: str = "Omega",
        values_name: str = "sample",
    ) -> SampleSpace:
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if not isinstance(values_name, str):
            raise TypeError("'values_name' must be a string.")
        if not isinstance(name, str):
            raise TypeError("'name' must be a string.")
        if not isinstance(prefix, str):
            raise TypeError("'prefix' must be a string.")

        if size == 1:
            indices = [prefix]
        else:
            indices = [
                f"{prefix}{i}" for i in range(initial_index, initial_index + size)
            ]
        return cls(indices=indices, name=name, values_name=values_name)

    # --------------------- conversion methods --------------------- #

    def make_probability_space(
        self,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilitySpace:
        from . import ProbabilitySpace

        return ProbabilitySpace(
            sample_space=self,
            sigma_algebra=sigma_algebra,
            probability_measure=probability_measure,
        )

    def make_event_space(self, sigma_algebra: SigmaAlgebra | None = None) -> EventSpace:
        from .event_space import EventSpace

        return EventSpace(sample_space=self, sigma_algebra=sigma_algebra)

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

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Sample space '{self.name}':\n{self._values.to_list()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        return (
            isinstance(other, SampleSpace)
            and self.values.equals(other.values)
            and self.name == other.name
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        indices: list[Hashable], name: str, values_name: str
    ) -> None:
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
        if not isinstance(name, str):
            raise TypeError("name must be a string.")


class SampleSpaceMethods:
    def get_event(self, event_indices: list[Hashable], name: str = "A") -> Event:
        return self.sample_space.get_event(event_indices, name)

    @property
    def get_event_at(self) -> Event:
        return self.sample_space.get_event_at
