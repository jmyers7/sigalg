import pandas as pd

from ..probability_measures import ProbabilityMeasure
from .probability_space import ProbabilitySpace
from .sample_space import SampleSpace


class Event(SampleSpace):
    def __init__(
        self,
        sample_space,
        event_indices,
    ):
        self._validate_parameters(sample_space, event_indices)
        self._sample_space = sample_space
        self._index = pd.Index(event_indices)
        super().__init__(event_indices)

        if isinstance(sample_space, ProbabilitySpace):
            self._probability = sample_space.P(self)
            probs = (
                {idx: sample_space.P(idx) / self._probability for idx in event_indices}
                if self._probability > 0
                else None
            )
            self._probability_measure = (
                ProbabilityMeasure(self, probs) if self._probability > 0 else None
            )
        else:
            self._probability = None
            self._probability_measure = None

    @property
    def sample_space(self):
        return self._sample_space

    @property
    def probability(self):
        return self._probability

    @property
    def probability_measure(self):
        return self._probability_measure

    def P(self, key):
        if self._probability_measure is None:
            raise ValueError("Event has no associated probability measure.")
        return self._probability_measure(key)

    def _set_default_sigma_algebra(self):
        from ..sigma_algebras import SigmaAlgebra

        sample_space_sigma_algebra = self._sample_space.sigma_algebra
        atom_ids = {
            idx: sample_space_sigma_algebra.atom_ids[idx] for idx in self._index
        }
        self._sigma_algebra = SigmaAlgebra(
            sample_space=self,
            atom_ids=atom_ids,
        )

    def __repr__(self):
        if self._probability is not None:
            return f"Event({list(self._index)}, P={self._probability:.4f})"
        return f"Event({list(self._index)})"

    def __eq__(self, other):
        return isinstance(other, Event) and self._index.equals(other._index)

    @staticmethod
    def _validate_parameters(sample_space, event_indices):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")

        # Check for duplicates
        if len(event_indices) != len(set(event_indices)):
            raise ValueError("Event indices must be unique (no duplicates allowed).")

        # Check all indices exist in sample space
        valid_indices = set(sample_space.index)
        for idx in event_indices:
            if idx not in valid_indices:
                raise ValueError(f"Index '{idx}' not found in sample space.")
