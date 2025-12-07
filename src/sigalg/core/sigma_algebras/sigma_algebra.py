from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..spaces.event import Event
    from ..spaces.probability_space import ProbabilitySpace
    from ..spaces.sample_space import SampleSpace


class SigmaAlgebra:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_id_to_atom_id: dict[Hashable, Hashable],
        sample_space: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
        name: str = "F",
    ) -> None:
        self._validate_parameters(sample_id_to_atom_id, sample_space, probability_space)
        if probability_space is not None:
            sample_space = probability_space.sample_space
            self._probability_space = probability_space
        else:
            self._probability_space = None
        self._sample_space = sample_space
        self._sample_id_to_atom_id = sample_id_to_atom_id
        self._values = pd.Series(data=sample_id_to_atom_id, name=name)
        self._values.index.name = sample_space.name
        self._name = name

        atom_id_to_sample_list = {}
        for sample_id, atom_id in sample_id_to_atom_id.items():
            if atom_id not in atom_id_to_sample_list:
                atom_id_to_sample_list[atom_id] = []
            atom_id_to_sample_list[atom_id].append(sample_id)
        self._atom_id_to_sample_list = atom_id_to_sample_list

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def probability_space(self) -> ProbabilitySpace | None:
        return self._probability_space

    @property
    def sample_id_to_atom_id(self) -> dict:
        return self._sample_id_to_atom_id.copy()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name
        self._values.name = new_name

    @property
    def values(self) -> pd.Series:
        return self._values.copy()

    @property
    def num_atoms(self) -> int:
        return len(self._atom_id_to_sample_list)

    # --------------------- methods --------------------- #

    def to_events(self) -> dict[Hashable, Event]:
        events = {}
        for atom_id, sample_ids in self._atom_id_to_sample_list.items():
            event = self.sample_space.get_event(sample_ids, name=str(atom_id))
            events[atom_id] = event
        return events

    def to_events_as_probability_spaces(self) -> dict[Hashable, Event]:
        events = {}
        for atom_id, sample_ids in self._atom_id_to_sample_list.items():
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
            atom_id = self._sample_id_to_atom_id[event_sample_id]
            atom_sample_ids = set(self._atom_id_to_sample_list[atom_id])
            if not event_sample_ids.issuperset(atom_sample_ids):
                return False
        return True

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        from ..spaces import Event

        if sample_id not in self._sample_id_to_atom_id:
            raise ValueError(f"Sample ID '{sample_id}' not in sample space.")
        atom_id = self._sample_id_to_atom_id[sample_id]
        sample_ids = self._atom_id_to_sample_list[atom_id]
        return Event(sample_space=self._sample_space, event_indices=sample_ids)

    # --------------------- class methods --------------------- #

    @classmethod
    def power_set(
        cls,
        sample_space: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
    ) -> SigmaAlgebra:
        if probability_space is not None:
            sample_space = probability_space.sample_space
        sample_id_to_atom_id = {
            index: idx for idx, index in enumerate(sample_space.values)
        }
        return cls(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            probability_space=probability_space,
        )

    @classmethod
    def trivial(
        cls,
        sample_space: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
    ) -> SigmaAlgebra:
        if probability_space is not None:
            sample_space = probability_space.sample_space
        sample_id_to_atom_id = dict.fromkeys(sample_space.values, 0)
        return cls(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            probability_space=probability_space,
        )

    # --------------------- iter method --------------------- #

    def __iter__(self) -> iter:
        return iter(self.to_events().items())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Sigma algebra '{self.name}':\n{self.values.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return False
        if self._sample_space != other._sample_space:
            return False
        return self <= other and other <= self

    # --------------------- order relations --------------------- #

    def __le__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self._sample_space != other._sample_space:
            raise ValueError(
                "Sigma algebras must have the same sample space for comparison."
            )
        from .utils import is_sub_algebra

        return is_sub_algebra(sub=self, super=other)

    def __lt__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self._sample_space != other._sample_space:
            raise ValueError(
                "Sigma algebras must have the same sample space for comparison."
            )
        from .utils import is_sub_algebra

        return is_sub_algebra(sub=other, super=self)

    def __gt__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        return self >= other and self != other

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_id_to_atom_id: dict[Hashable, Hashable],
        sample_space: SampleSpace,
        probability_space: ProbabilitySpace,
    ) -> None:
        from ..spaces import ProbabilitySpace, SampleSpace

        if sample_space is None and probability_space is None:
            raise ValueError(
                "Either sample_space or probability_space must be provided."
            )
        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if probability_space is not None and not isinstance(
            probability_space, ProbabilitySpace
        ):
            raise TypeError("probability_space must be a ProbabilitySpace instance.")
        if (sample_space is not None and probability_space is not None) and (
            sample_space != probability_space.sample_space
        ):
            raise ValueError(
                "sample_space and probability_space.sample_space must be the same."
            )
        if not isinstance(sample_id_to_atom_id, dict):
            raise TypeError(
                "sample_id_to_atom_id must be a dictionary mapping sample indices to atom IDs."
            )
        if sample_space is not None and set(sample_id_to_atom_id.keys()) != set(
            sample_space.values
        ):
            raise ValueError(
                "sample_id_to_atom_id must contain an entry for every sample index in sample_space."
            )
        if probability_space is not None and set(sample_id_to_atom_id.keys()) != set(
            probability_space.sample_space.values
        ):
            raise ValueError(
                "sample_id_to_atom_id must contain an entry for every sample index in probability_space.sample_space."
            )


class SigmaAlgebraMethods:
    def is_measurable(self, event: Event) -> bool:
        return self.sigma_algebra.is_measurable(event)

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        return self.sigma_algebra.get_atom_containing(sample_id)
