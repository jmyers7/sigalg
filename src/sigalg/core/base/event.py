from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index
from .sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from .sample_space import SampleSpace


class Event(SampleSpaceMethods, Index):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        event_indices: list[Hashable],
        name: str = "A",
        values_name: str = "sample",
    ) -> None:
        self._validate_event_parameters(
            sample_space=sample_space,
            event_indices=event_indices,
        )
        pts = set(event_indices)
        ordered = [idx for idx in sample_space.values if idx in pts]
        super().__init__(indices=ordered, name=name, values_name=values_name)
        self.sample_space = sample_space

    # --------------------- set-theoretic operations --------------------- #

    def complement(self) -> Event:
        return ~self

    def intersection(self, other: Event) -> Event:
        return self & other

    def union(self, other: Event) -> Event:
        return self | other

    def difference(self, other: Event) -> Event:
        return self - other

    # --------------------- set-theoretic operators --------------------- #

    def __invert__(self) -> Event:
        space = self.sample_space.values
        pts = set(self.values)
        comp = [idx for idx in space if idx not in pts]
        return Event(self.sample_space, comp, name=f"{self.name} complement")

    def __or__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.values) | set(other.values)
        return Event(
            self.sample_space, list(pts), name=f"{self.name} union {other.name}"
        )

    def __and__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.values) & set(other.values)
        return Event(
            self.sample_space, list(pts), name=f"{self.name} intersect {other.name}"
        )

    def __sub__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.values) - set(other.values)
        return Event(
            self.sample_space, list(pts), name=f"{self.name} difference {other.name}"
        )

    # --------------------- sub/superset methods --------------------- #

    def __le__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.values).issubset(set(other.values))

    def __lt__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.values) < set(other.values)

    def __ge__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.values).issuperset(set(other.values))

    def __gt__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.values) > set(other.values)

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Event)
            and self.sample_space == other.sample_space
            and self.values.equals(other.values)
        )

    # --------------------- conversion methods --------------------- #

    def to_sample_space(self) -> SampleSpace:
        from ..base import SampleSpace

        return SampleSpace(self.values.to_list())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Event '{self.name}':\n{self.values.to_list()}"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_event_parameters(
        sample_space: SampleSpace,
        event_indices: list[Hashable],
    ) -> None:
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list.")
        for idx in event_indices:
            if idx not in sample_space.values:
                raise ValueError(f"Index '{idx}' not found in sample_space.")
