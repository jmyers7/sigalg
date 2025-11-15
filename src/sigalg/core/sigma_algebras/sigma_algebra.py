import pandas as pd

from ..spaces import Event, SampleSpace


class SigmaAlgebra:

    def __init__(self, sample_space, atom_ids):
        self._validate_parameters(sample_space, atom_ids)
        self._sample_space = sample_space
        self._atom_ids = atom_ids

        atom_id_to_sample_ids = {}
        for sample_id, atom_id in atom_ids.items():
            if atom_id not in atom_id_to_sample_ids:
                atom_id_to_sample_ids[atom_id] = []
            atom_id_to_sample_ids[atom_id].append(sample_id)
        self._atom_id_to_sample_ids = atom_id_to_sample_ids

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def atom_ids(self) -> dict:
        return self._atom_ids

    def to_events(self):
        events = {}
        for atom_id, sample_ids in self._atom_id_to_sample_ids.items():
            event = Event(
                sample_space=self._sample_space,
                event_indices=sample_ids,
            )
            events[atom_id] = event

        return events

    def __repr__(self):
        series = pd.Series(self._atom_ids, name="Atom IDs")
        return repr(series)

    def __iter__(self):
        return iter(self.to_events().items())

    def __eq__(self, other):
        if not isinstance(other, SigmaAlgebra):
            return False
        return (
            self._sample_space == other._sample_space
            and self._atom_ids == other._atom_ids
        )

    def is_measurable(self, event):
        return is_measurable(self, event)

    @staticmethod
    def _validate_parameters(sample_space, atom_ids):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(atom_ids, dict):
            raise TypeError(
                "atom_ids must be a dictionary mapping sample indices to atom IDs."
            )
        if atom_ids.keys() != set(sample_space):
            raise ValueError(
                "atom_ids must contain an entry for every sample index in sample_space."
            )


def is_measurable(sigma_algebra, event):
    if not isinstance(sigma_algebra, SigmaAlgebra):
        raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
    if not isinstance(event, Event):
        raise TypeError("event must be an Event instance.")
    if event.sample_space != sigma_algebra._sample_space:
        raise ValueError("event must have the same sample_space as the sigma_algebra.")

    event_sample_ids = set(event)
    for event_sample_id in event_sample_ids:
        atom_id = sigma_algebra._atom_ids[event_sample_id]
        atom_sample_ids = set(sigma_algebra._atom_id_to_sample_ids[atom_id])
        if not event_sample_ids.issuperset(atom_sample_ids):
            return False
    return True
