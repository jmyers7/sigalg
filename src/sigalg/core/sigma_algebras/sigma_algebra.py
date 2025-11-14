from ..feature_representations import SampleSpaceFeatures
import pandas as pd


class SigmaAlgebra:

    def __init__(self, sample_space_features, atom_ids):
        self._validate_parameters(sample_space_features, atom_ids)
        self._sample_space_features = sample_space_features
        self._atom_ids = atom_ids

        atom_id_to_sample_ids = {}
        for sample_id, atom_id in atom_ids.items():
            if atom_id not in atom_id_to_sample_ids:
                atom_id_to_sample_ids[atom_id] = []
            atom_id_to_sample_ids[atom_id].append(sample_id)
        self._atom_id_to_sample_ids = atom_id_to_sample_ids

    def to_events(self):
        from ..feature_representations import EventFeatures  # lazy import

        events = {}
        for atom_id, sample_ids in self._atom_id_to_sample_ids.items():
            event = EventFeatures(
                sample_space_features=self._sample_space_features,
                sample_index=sample_ids,
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
            self._sample_space_features == other._sample_space_features
            and self._atom_ids == other._atom_ids
        )

    def is_measurable(self, event_features):
        return is_measurable(self, event_features)

    @staticmethod
    def _validate_parameters(space_features, atom_ids):
        if not isinstance(space_features, SampleSpaceFeatures):
            raise TypeError("space_features must be a SampleSpaceFeatures instance.")
        if not isinstance(atom_ids, dict):
            raise TypeError(
                "atom_ids must be a dictionary mapping sample indices to atom IDs."
            )
        for atom_id in atom_ids.keys():
            if atom_id not in space_features.sample_index:
                raise ValueError(
                    f"atom_id '{atom_id}' not found in space_features sample_index."
                )


def is_measurable(sigma_algebra, event_features):
    from ..feature_representations import EventFeatures

    if not isinstance(sigma_algebra, SigmaAlgebra):
        raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
    if not isinstance(event_features, EventFeatures):
        raise TypeError("event_features must be an EventFeatures instance.")
    if event_features.sample_space_features != sigma_algebra._sample_space_features:
        raise ValueError(
            "event_features must have the same space_features as the sigma_algebra."
        )

    event_sample_ids = set(event_features.sample_index)
    for event_sample_id in event_sample_ids:
        atom_id = sigma_algebra._atom_ids[event_sample_id]
        atom_sample_ids = set(sigma_algebra._atom_id_to_sample_ids[atom_id])
        if not event_sample_ids.issuperset(atom_sample_ids):
            return False
    return True
