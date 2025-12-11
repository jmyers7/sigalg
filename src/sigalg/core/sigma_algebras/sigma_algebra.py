from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.probability_space import ProbabilitySpace
    from ..base.sample_space import SampleSpace


class SigmaAlgebra:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_id_to_atom_id: dict[Hashable, Hashable],
        sample_space: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
        name: str = "F",
    ) -> None:
        self._validate_parameters(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            probability_space=probability_space,
        )
        if sample_space is None and probability_space is None:
            sample_space = self._generate_sample_space(sample_id_to_atom_id)
        elif probability_space is not None:
            sample_space = probability_space.sample_space
            self.probability_space = probability_space
        else:
            self.probability_space = None
        self.sample_space = sample_space
        self.sample_id_to_atom_id = sample_id_to_atom_id
        self.values = pd.Series(data=sample_id_to_atom_id, name=name)
        self.values.index.name = sample_space.name if sample_space is not None else None
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name
        self.values.name = new_name

    @property
    def num_atoms(self) -> int:
        return len(self.values.unique())

    @property
    def atom_ids(self) -> list[Hashable]:
        return list(self.sample_id_to_atom_id.values())

    @property
    def atom_id_to_sample_ids(self) -> dict[Hashable, list[Hashable]]:
        if not hasattr(self, "_atom_id_to_sample_ids"):
            atom_id_to_sample_ids = {}
            for sample_id, atom_id in self.sample_id_to_atom_id.items():
                if atom_id not in atom_id_to_sample_ids:
                    atom_id_to_sample_ids[atom_id] = []
                atom_id_to_sample_ids[atom_id].append(sample_id)
            self._atom_id_to_sample_ids = atom_id_to_sample_ids
        return self._atom_id_to_sample_ids

    @property
    def atom_id_to_event(self) -> dict[Hashable, Event]:
        if not hasattr(self, "_atom_id_to_event"):
            atom_id_to_event = {
                atom_id: self.sample_space.get_event(sample_ids, name=str(atom_id))
                for atom_id, sample_ids in self.atom_id_to_sample_ids.items()
            }
            self._atom_id_to_event = atom_id_to_event
        return self._atom_id_to_event

    @property
    def atom_id_to_cardinality(self) -> dict[Hashable, int]:
        if not hasattr(self, "_atom_id_to_cardinality"):
            self._atom_id_to_cardinality = {
                atom_id: len(event) for atom_id, event in self.atom_id_to_event.items()
            }
        return self._atom_id_to_cardinality

    @property
    def atom_id_to_probability_space(self) -> dict[Hashable, ProbabilitySpace]:
        if self.probability_space is None:
            raise ValueError("No probability space associated with this sigma algebra.")
        if not hasattr(self, "_atom_id_to_probability_space"):
            atom_id_to_probability_space = {
                atom_id: self.probability_space.get_event_as_probability_space(
                    sample_ids
                )
                for atom_id, sample_ids in self.atom_id_to_sample_ids.items()
            }
            self._atom_id_to_probability_space = atom_id_to_probability_space
        return self._atom_id_to_probability_space

    # --------------------- methods --------------------- #

    def to_atoms(self) -> list[Event]:
        return list(self.atom_id_to_event.values())

    def is_measurable(self, event: Event) -> bool:
        from ..base import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")
        if event.sample_space != self.sample_space:
            raise ValueError(
                "event must have the same sample_space as the sigma_algebra."
            )

        event_sample_ids = set(event.values)
        for event_sample_id in event_sample_ids:
            atom_id = self.sample_id_to_atom_id[event_sample_id]
            atom_sample_ids = set(self.atom_id_to_sample_ids[atom_id])
            if not event_sample_ids.issuperset(atom_sample_ids):
                return False
        return True

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        from ..base import Event

        if sample_id not in self.sample_id_to_atom_id:
            raise ValueError(f"Sample ID '{sample_id}' not in sample space.")
        atom_id = self.sample_id_to_atom_id[sample_id]
        sample_ids = self.atom_id_to_sample_ids[atom_id]
        return Event(sample_space=self.sample_space, event_indices=sample_ids)

    @staticmethod
    def _generate_sample_space(
        sample_id_to_atom_id: dict[Hashable, Hashable],
    ) -> SampleSpace:
        from ..base.sample_space import SampleSpace

        indices = list(sample_id_to_atom_id.keys())
        return SampleSpace(indices)

    # --------------------- factory methods --------------------- #

    @classmethod
    def power_set(
        cls,
        sample_space: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
        name: str = "power_set",
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
            name=name,
        )

    @classmethod
    def trivial(
        cls,
        sample_space: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
        name: str = "trivial",
    ) -> SigmaAlgebra:
        if probability_space is not None:
            sample_space = probability_space.sample_space
        sample_id_to_atom_id = dict.fromkeys(sample_space.values, 0)
        return cls(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            probability_space=probability_space,
            name=name,
        )

    # --------------------- iter method --------------------- #

    def __iter__(self) -> iter:
        return iter(self.atom_id_to_event.items())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Sigma algebra '{self.name}':\n{self.values.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return False
        if self.sample_space != other.sample_space:
            return False
        return self <= other and other <= self

    # --------------------- order relations --------------------- #

    def __le__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self.sample_space != other.sample_space:
            raise ValueError(
                "Sigma algebras must have the same sample space for comparison."
            )
        from .comparison import is_subalgebra

        return is_subalgebra(sub_algebra=self, super_algebra=other)

    def __lt__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other: SigmaAlgebra) -> bool:
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self.sample_space != other.sample_space:
            raise ValueError(
                "Sigma algebras must have the same sample space for comparison."
            )
        from .comparison import is_subalgebra

        return is_subalgebra(sub_algebra=other, super_algebra=self)

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
        from ..base import ProbabilitySpace, SampleSpace

        # if sample_space is None and probability_space is None:
        #     raise ValueError(
        #         "Either sample_space or probability_space must be provided."
        #     )
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
