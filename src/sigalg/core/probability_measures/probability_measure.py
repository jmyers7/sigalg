import pandas as pd

from ..spaces import SampleSpace


class ProbabilityMeasure:
    def __init__(self, sample_space: SampleSpace, probabilities):
        self._validate_parameters(sample_space, probabilities)
        self._sample_space = sample_space
        self._probabilities = pd.Series(probabilities)

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def probabilities(self) -> pd.Series:
        return self._probabilities

    def __call__(self, key) -> float:
        from ..spaces import Event

        if isinstance(key, str):
            return self._probabilities[key]
        elif isinstance(key, Event):
            if key.sample_space != self._sample_space:
                raise ValueError("Event must be from the same sample space.")
            return self._probabilities.loc[list(key.index)].sum()
        else:
            raise TypeError("Key must be a string (sample index) or Event.")

    def __getitem__(self, key) -> float:
        return self(key)

    def __repr__(self):
        return f"ProbabilityMass(\n{self._probabilities}\n)"

    def __eq__(self, other):
        if not isinstance(other, ProbabilityMeasure):
            return False
        return self._probabilities.equals(other._probabilities)

    @staticmethod
    def uniform(sample_space: SampleSpace):
        n = len(sample_space)
        probabilities = dict.fromkeys(sample_space.index, 1.0 / n) if n > 0 else None
        return ProbabilityMeasure(sample_space, probabilities)

    @staticmethod
    def _validate_parameters(sample_space, probabilities):
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(probabilities, dict):
            raise TypeError("probabilities must be a dictionary.")

        prob_indices = set(probabilities.keys())
        space_indices = set(sample_space.index)

        if prob_indices != space_indices:
            raise ValueError("Probabilities keys must match sample space indices.")

        for prob in probabilities.values():
            if not (0.0 <= prob <= 1.0):
                raise ValueError("All probabilities must be in [0, 1].")

        total = sum(probabilities.values())
        if not abs(total - 1.0) < 1e-10:
            raise ValueError(f"Probabilities must sum to 1, got {total}.")
