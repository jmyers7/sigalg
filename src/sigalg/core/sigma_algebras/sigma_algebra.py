from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace


class SigmaAlgebra:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_id_to_atom_id: Mapping[Hashable, Hashable],
        sample_space: SampleSpace,
        name: Hashable = "F",
    ) -> None:

        v = SampleSpaceMappingIn(
            mapping=sample_id_to_atom_id,
            sample_space=sample_space,
            name=name,
        )

        self.sample_id_to_atom_id = v.mapping
        self.sample_space = v.sample_space
        self._name = v.name

        # caches for properties
        self._data: pd.Series | None = None
        self._num_atoms: int | None = None
        self._atom_ids: list[Hashable] | None = None
        self._atom_id_to_sample_ids: dict[Hashable, list[Hashable]] | None = None
        self._atom_id_to_event: dict[Hashable, Event] | None = None
        self._atom_id_to_cardinality: dict[Hashable, int] | None = None

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series:
        if self._data is None:
            self._data = pd.Series(
                data=list(self.sample_id_to_atom_id.values()),
                index=self.sample_space.data,
                name=self.name,
            )
        return self._data

    @data.setter
    def data(self, data: pd.Series) -> None:
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        if set(data.index) != set(self.sample_space.data):
            raise ValueError("data index must match sample space indices.")
        self._data = data

    @property
    def name(self) -> Hashable:
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable type.")
        self._name = name
        if self._data is not None:
            self._data.name = name

    @property
    def num_atoms(self) -> int:
        if self._num_atoms is None:
            self._num_atoms = self.data.nunique()
        return self._num_atoms

    @property
    def atom_ids(self) -> list[Hashable]:
        if self._atom_ids is None:
            self._atom_ids = list(self.data.unique())
        return self._atom_ids

    @property
    def atom_id_to_sample_ids(self) -> dict[Hashable, list[Hashable]]:
        if self._atom_id_to_sample_ids is None:
            atom_id_to_sample_ids = {}
            for sample_id, atom_id in self.sample_id_to_atom_id.items():
                if atom_id not in atom_id_to_sample_ids:
                    atom_id_to_sample_ids[atom_id] = []
                atom_id_to_sample_ids[atom_id].append(sample_id)
            self._atom_id_to_sample_ids = atom_id_to_sample_ids
        return self._atom_id_to_sample_ids

    @property
    def atom_id_to_event(self) -> dict[Hashable, Event]:
        if self._atom_id_to_event is None:
            atom_id_to_event = {
                atom_id: self.sample_space.get_event(sample_ids, name=atom_id)
                for atom_id, sample_ids in self.atom_id_to_sample_ids.items()
            }
            self._atom_id_to_event = atom_id_to_event
        return self._atom_id_to_event

    @property
    def atom_id_to_cardinality(self) -> dict[Hashable, int]:
        if self._atom_id_to_cardinality is None:
            self._atom_id_to_cardinality = {
                atom_id: len(event) for atom_id, event in self.atom_id_to_event.items()
            }
        return self._atom_id_to_cardinality

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

        event_sample_ids = set(event.data)
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
        return Event(sample_space=self.sample_space, indices=sample_ids)

    def __contains__(self, event: Event) -> bool:
        return self.is_measurable(event)

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_pandas(
        cls,
        data: pd.Series,
        name: Hashable = "F",
    ) -> SigmaAlgebra:
        from ..base.sample_space import SampleSpace

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        sample_space = SampleSpace.from_pandas(data.index, name="Omega")
        sample_id_to_atom_id = data.to_dict()
        sigma_algebra = cls(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            name=name,
        )
        sigma_algebra.data = data
        return sigma_algebra

    @classmethod
    def power_set(
        cls,
        sample_space: SampleSpace,
        name: Hashable = "power_set",
    ) -> SigmaAlgebra:
        sample_id_to_atom_id = {
            index: idx for idx, index in enumerate(sample_space.data)
        }
        return cls(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            name=name,
        )

    @classmethod
    def trivial(
        cls,
        sample_space: SampleSpace,
        name: Hashable = "trivial",
    ) -> SigmaAlgebra:
        sample_id_to_atom_id = dict.fromkeys(sample_space.data, 0)
        return cls(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            name=name,
        )

    # --------------------- iter method --------------------- #

    def __iter__(self) -> iter:
        return iter(self.atom_id_to_event.items())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Sigma algebra '{self.name}':\n{self.data.to_frame()}"

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


class SigmaAlgebraMethods:
    def is_measurable(self, event: Event) -> bool:
        return self.sigma_algebra.is_measurable(event)

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        return self.sigma_algebra.get_atom_containing(sample_id)
