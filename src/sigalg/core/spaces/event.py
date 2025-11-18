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
        self._index = pd.Index(ordered)

    # --------------------- properties --------------------- #

    @property
    def index(self) -> pd.Index:
        return self._index

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
        return len(self._index)

    def __iter__(self) -> iter:
        return iter(self._index)

    def __getitem__(self, key) -> Hashable | Event:
        if isinstance(key, list):
            for k in key:
                if k not in self._index:
                    raise ValueError(f"Index '{k}' not found in this event.")
            return Event(self.sample_space, key)
        return self._index[key]

    # --------------------- set-theoretic operators --------------------- #

    def __invert__(self) -> Event:
        space = self.sample_space.values
        pts = set(self.index)
        comp = [idx for idx in space if idx not in pts]
        return Event(self.sample_space, comp)

    def __or__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.index) | set(other.index)
        return Event(self.sample_space, list(pts))

    def __and__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.index) & set(other.index)
        return Event(self.sample_space, list(pts))

    def __sub__(self, other: Event) -> Event:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.index) - set(other.index)
        return Event(self.sample_space, list(pts))

    # --------------------- sub/superset methods --------------------- #

    def __le__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.index).issubset(set(other.index))

    def __lt__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.index) < set(other.index)

    def __ge__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.index).issuperset(set(other.index))

    def __gt__(self, other: Event) -> bool:
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.index) > set(other.index)

    # --------------------- equality & hashing --------------------- #

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Event)
            and self.sample_space == other.sample_space
            and self.index.equals(other.index)
        )

    def __hash__(self) -> int:
        return hash((self.sample_space, tuple(self.index)))

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Event({list(self._index)})"

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
