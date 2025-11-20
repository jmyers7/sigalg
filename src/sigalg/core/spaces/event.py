from __future__ import annotations

from collections.abc import Hashable

import pandas as pd

from .sample_space import SampleSpace


class Event:

    # --------------------- constructor --------------------- #

    def __init__(
        self, sample_space: SampleSpace, event_indices: list[Hashable]
    ) -> None:
        self._validate_parameters(sample_space, event_indices)
        pts = set(event_indices)
        ordered = [idx for idx in sample_space.values if idx in pts]
        self._sample_space = sample_space
        self._values = pd.Index(ordered)

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Index:
        return self._values

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    # --------------------- set-theoretic methods --------------------- #

    def complement(self) -> Event:
        return ~self

    def intersection(self, other: Event) -> Event:
        return self & other

    def union(self, other: Event) -> Event:
        return self | other

    def difference(self, other: Event) -> Event:
        return self - other

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> iter:
        return iter(self._values)

    def __getitem__(self, key) -> Hashable | Event:
        if isinstance(key, list):
            for k in key:
                if k not in self._values:
                    raise ValueError(f"Index '{k}' not found in this event.")
            return Event(self.sample_space, key)
        return self._values[key]

    # --------------------- set-theoretic operators --------------------- #

    def __invert__(self) -> Event:
        space = self.sample_space.values
        pts = set(self.values)
        comp = [idx for idx in space if idx not in pts]
        return Event(self.sample_space, comp)

    def __or__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.values) | set(other.values)
        return Event(self.sample_space, list(pts))

    def __and__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.values) & set(other.values)
        return Event(self.sample_space, list(pts))

    def __sub__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.values) - set(other.values)
        return Event(self.sample_space, list(pts))

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

    # --------------------- equality & hashing --------------------- #

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Event)
            and self.sample_space == other.sample_space
            and self.values.equals(other.values)
        )

    def __hash__(self) -> int:
        return hash((self.sample_space, tuple(self.values)))

    # --------------------- conversion methods --------------------- #

    def to_sample_space(self) -> SampleSpace:
        return SampleSpace(self.values.to_list())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Event({list(self._values)})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace, event_indices: list[Hashable]
    ) -> None:
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of Hashable items.")
        for idx in event_indices:
            if idx not in sample_space.values:
                raise ValueError(
                    f"event_indices contains index '{idx}' not in sample_space."
                )
