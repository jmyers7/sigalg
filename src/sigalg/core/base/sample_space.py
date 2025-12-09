from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from . import ProbabilitySpace
    from .event import Event
    from .event_space import EventSpace


class SampleSpace(Index):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        indices: list[Hashable],
        name: str = "Omega",
        values_name: str = "sample",
    ) -> None:
        super().__init__(indices=indices, values_name=values_name)
        self._validate_sample_space_parameters(indices=indices, name=name)
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
            if idx not in self.values:
                raise ValueError(f"Index '{idx}' not found in sample space.")
        return Event(sample_space=self, event_indices=event_indices, name=name)

    def _getitem_hook(self, key):
        from .event import Event

        if isinstance(key, tuple) and len(key) == 2:
            item_idx, name = key
        else:
            item_idx = key
            name = "A"
        event_indices = self.values[item_idx].to_list()
        return Event(sample_space=self, event_indices=event_indices, name=name)

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> iter:
        return iter(self.values)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Sample space '{self.name}':\n{self.values.to_list()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        return (
            isinstance(other, SampleSpace)
            and self.values.equals(other.values)
            and self.name == other.name
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_sample_space_parameters(
        indices: list[Hashable],
        name: str,
    ) -> None:
        if len(indices) == 0:
            raise ValueError("Sample space must contain at least one index.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")


class SampleSpaceMethods:
    def get_event(self, event_indices: list[Hashable], name: str = "A") -> Event:
        return self.sample_space.get_event(event_indices, name)

    def _getitem_hook(self, key):
        return self.sample_space._getitem_hook(key)
