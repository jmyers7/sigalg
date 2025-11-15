import pandas as pd
from .sample_space import SampleSpace


class Event(SampleSpace):
    def __init__(self, sample_space, event_indices, probability=None):
        self._validate_parameters(sample_space, event_indices)
        self._sample_space = sample_space
        self._index = pd.Index(event_indices)
        self._probability = probability

    @property
    def sample_space(self):
        return self._sample_space

    @property
    def probability(self):
        return self._probability

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
