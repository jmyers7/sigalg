from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..spaces import Event, ProbabilitySpace, SampleSpace


class SigmaAlgebra:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        space: SampleSpace | ProbabilitySpace,
        atom_ids: dict[Hashable, Hashable],
        name: str = "F",
    ) -> None:
        from ..spaces import ProbabilitySpace

        self._validate_parameters(space, atom_ids)
        if isinstance(space, ProbabilitySpace):
            self._sample_space = space.sample_space
            self._probability_space = space
        else:
            self._sample_space = space
            self._probability_space = None
        self._atom_ids = atom_ids
        self._name = name

        atom_id_to_sample_ids = {}
        for sample_id, atom_id in atom_ids.items():
            if atom_id not in atom_id_to_sample_ids:
                atom_id_to_sample_ids[atom_id] = []
            atom_id_to_sample_ids[atom_id].append(sample_id)
        self._atom_id_to_sample_ids = atom_id_to_sample_ids

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def probability_space(self) -> ProbabilitySpace | None:
        return self._probability_space

    @property
    def atom_ids(self) -> dict:
        return self._atom_ids.copy()

    @property
    def num_atoms(self) -> int:
        return len(self._atom_id_to_sample_ids)

    @property
    def name(self) -> str:
        return self._name

    # --------------------- methods --------------------- #

    def to_events(self) -> dict[Hashable, Event]:
        events = {}
        for atom_id, sample_ids in self._atom_id_to_sample_ids.items():
            event = self.sample_space.get_event(sample_ids)
            events[atom_id] = event
        return events

    def to_events_as_probability_space(self) -> dict[Hashable, Event]:
        events = {}
        for atom_id, sample_ids in self._atom_id_to_sample_ids.items():
            event = self.probability_space.get_event_as_probability_space(sample_ids)
            events[atom_id] = event
        return events

    def is_measurable(self, event: Event) -> bool:
        from ..spaces import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")
        if event.sample_space != self._sample_space:
            raise ValueError(
                "event must have the same sample_space as the sigma_algebra."
            )

        event_sample_ids = set(event.values)
        for event_sample_id in event_sample_ids:
            atom_id = self._atom_ids[event_sample_id]
            atom_sample_ids = set(self._atom_id_to_sample_ids[atom_id])
            if not event_sample_ids.issuperset(atom_sample_ids):
                return False
        return True

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        from ..spaces import Event

        if sample_id not in self._atom_ids:
            raise ValueError(f"Sample ID '{sample_id}' not in sample space.")
        atom_id = self._atom_ids[sample_id]
        sample_ids = self._atom_id_to_sample_ids[atom_id]
        return Event(sample_space=self._sample_space, event_indices=sample_ids)

    # --------------------- class methods --------------------- #

    @classmethod
    def power_set(cls, space: SampleSpace) -> SigmaAlgebra:
        atom_ids = {index: idx for idx, index in enumerate(space.values)}
        return cls(space=space, atom_ids=atom_ids)

    @classmethod
    def trivial(cls, space: SampleSpace) -> SigmaAlgebra:
        """Create the trivial sigma-algebra {∅, Ω}."""
        atom_ids = dict.fromkeys(space.values, 0)
        return cls(space=space, atom_ids=atom_ids)

    # --------------------- iter method --------------------- #

    def __iter__(self) -> iter:
        return iter(self.to_events().items())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        series = pd.Series(self._atom_ids, name="Atom IDs")
        return repr(series)

    # --------------------- equality & hashing --------------------- #

    def __eq__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return False
        return (
            self._sample_space == other._sample_space
            and self._atom_ids == other._atom_ids
        )

    def __hash__(self) -> int:
        return hash((self._sample_space, frozenset(self._atom_ids.items())))

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        space: SampleSpace | ProbabilitySpace, atom_ids: dict[Hashable, Hashable]
    ) -> None:
        from ..spaces import ProbabilitySpace, SampleSpace

        if not isinstance(space, SampleSpace) and not isinstance(
            space, ProbabilitySpace
        ):
            raise TypeError("space must be a SampleSpace or ProbabilitySpace instance.")
        if not isinstance(atom_ids, dict):
            raise TypeError(
                "atom_ids must be a dictionary mapping sample indices to atom IDs."
            )
        if set(atom_ids.keys()) != set(space.values):
            raise ValueError(
                "atom_ids must contain an entry for every sample index in sample_space."
            )
        try:
            frozenset(atom_ids.values())
        except TypeError as e:
            raise TypeError("All atom IDs must be hashable.") from e


class SigmaAlgebraMethods:
    # --------------------- properties --------------------- #

    @property
    def atom_ids(self) -> dict:
        return self.sigma_algebra.atom_ids

    @property
    def num_atoms(self) -> int:
        return self.sigma_algebra.num_atoms

    # --------------------- methods --------------------- #

    def to_events(self) -> dict[Hashable, Event]:
        return self.sigma_algebra.to_events()

    def is_measurable(self, event: Event) -> bool:
        return self.sigma_algebra.is_measurable(event)

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        return self.sigma_algebra.get_atom_containing(sample_id)
