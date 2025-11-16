from __future__ import annotations

from collections.abc import Hashable
from numbers import Real

import pandas as pd

from ..spaces import Event, SampleSpace


class ProbabilityMeasure:

    # --------------------- constructor --------------------- #

    def __init__(
        self, sample_space: SampleSpace, probabilities: dict[Hashable, Real]
    ) -> None:
        self._validate_parameters(sample_space, probabilities)
        self._sample_space = sample_space
        self._probabilities = probabilities

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def probabilities(self) -> dict[Hashable, Real]:
        return self._probabilities.copy()

    # --------------------- conversion methods --------------------- #

    def to_pandas(self) -> pd.Series:
        return pd.Series(self._probabilities, name="probability")

    # --------------------- class methods --------------------- #

    @classmethod
    def uniform(cls, sample_space: SampleSpace) -> ProbabilityMeasure:
        n = len(sample_space)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on empty sample space."
            )
        probabilities = dict.fromkeys(sample_space.index, 1.0 / n)
        return cls(sample_space, probabilities)

    # --------------------- access methods --------------------- #

    def __call__(self, key: Hashable | list[Hashable] | Event) -> Real:
        if isinstance(key, Event):
            if key.sample_space != self._sample_space:
                raise ValueError("Event must be from the same sample space.")
            return self.to_pandas().loc[list(key.index)].sum()
        elif isinstance(key, list):
            for idx in key:
                if idx not in self._probabilities:
                    raise KeyError(f"Index '{idx}' not found in sample space.")
            return sum(self._probabilities[idx] for idx in key)
        else:
            if key not in self._probabilities:
                raise KeyError(f"Index '{key}' not found in sample space.")
            return self._probabilities[key]

    def __getitem__(self, key: Hashable | list[Hashable] | Event) -> Real:
        return self(key)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"ProbabilityMeasure(\n{self.to_pandas()}\n)"

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProbabilityMeasure):
            return False
        return self.to_pandas().equals(other.to_pandas())

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace, probabilities: dict[Hashable, Real]
    ) -> None:
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(probabilities, dict):
            raise TypeError("probabilities must be a dictionary.")

        prob_indices = set(probabilities.keys())
        space_indices = set(sample_space.index)

        if prob_indices != space_indices:
            raise ValueError("Probabilities keys must match sample space indices.")

        for key, prob in probabilities.items():
            if not isinstance(prob, Real):
                raise TypeError(
                    f"Probability for '{key}' must be a Real number, got {type(prob)}."
                )
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"Probability for '{key}' must be in [0, 1], got {prob}."
                )

        total = sum(probabilities.values())
        if not abs(total - 1.0) < 1e-10:
            raise ValueError(f"Probabilities must sum to 1, got {total}.")
