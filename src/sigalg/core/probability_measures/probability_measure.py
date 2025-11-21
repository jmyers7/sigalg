from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..spaces import Event, SampleSpace


class ProbabilityMeasure:

    # --------------------- constructor --------------------- #

    def __init__(
        self, sample_space: SampleSpace, probabilities: dict[Hashable, Real]
    ) -> None:
        self._validate_parameters(sample_space, probabilities)
        self._sample_space = sample_space
        self._probabilities = probabilities
        self._values: pd.Series = pd.Series(probabilities, name="probability")

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def probabilities(self) -> dict[Hashable, Real]:
        return self._probabilities.copy()

    @property
    def values(self) -> pd.Series:
        return self._values.copy()

    # --------------------- methods --------------------- #

    def P(self, key: Hashable | Event) -> Real:
        return self(key)

    def conditional_probability(self, event_A: Event, event_B: Event) -> Real:
        if event_A.sample_space != self.sample_space:
            raise ValueError(
                "event_A must be from this probability space's sample space."
            )
        if event_B.sample_space != self.sample_space:
            raise ValueError(
                "event_B must be from this probability space's sample space."
            )
        prob_B = self.P(event_B)
        if prob_B < 1e-10:
            raise ValueError("Cannot compute conditional probability: P(B) = 0")
        intersection_indices = [idx for idx in event_A.values if idx in event_B.values]
        if not intersection_indices:
            return 0.0
        intersection_event = self.sample_space.get_event(intersection_indices)
        prob_intersection = self.P(intersection_event)
        return prob_intersection / prob_B

    def are_independent(
        self, event_A: Event, event_B: Event, tolerance: Real = 1e-10
    ) -> bool:
        if event_A.sample_space != self.sample_space:
            raise ValueError(
                "event_A must be from this probability space's sample space."
            )
        if event_B.sample_space != self.sample_space:
            raise ValueError(
                "event_B must be from this probability space's sample space."
            )
        prob_A = self.P(event_A)
        prob_B = self.P(event_B)
        intersection_indices = [idx for idx in event_A.values if idx in event_B.values]
        if not intersection_indices:
            prob_intersection = 0.0
        else:
            intersection_event = self.sample_space.get_event(intersection_indices)
            prob_intersection = self.P(intersection_event)
        return abs(prob_intersection - prob_A * prob_B) < tolerance

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
        probabilities = dict.fromkeys(sample_space.values, 1.0 / n)
        return cls(sample_space, probabilities)

    # --------------------- access methods --------------------- #

    def __call__(self, key: Hashable | list[Hashable] | Event) -> Real:
        from ..spaces import Event

        if isinstance(key, Event):
            if key.sample_space != self._sample_space:
                raise ValueError("Event must be from the same sample space.")
            return self.to_pandas().loc[list(key.values)].sum()
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
        from ..spaces import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(probabilities, dict):
            raise TypeError("probabilities must be a dictionary.")

        prob_indices = set(probabilities.keys())
        space_indices = set(sample_space.values)

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


class ProbabilityMeasureMethods:
    # --------------------- properties --------------------- #

    @property
    def probabilities(self) -> dict[Hashable, Real]:
        return self.probability_measure.probabilities

    # --------------------- methods --------------------- #

    def P(self, key: Hashable | Event) -> Real:
        return self.probability_measure(key)

    def conditional_probability(self, event_A: Event, event_B: Event) -> Real:
        return self.probability_measure.conditional_probability(event_A, event_B)

    def are_independent(
        self, event_A: Event, event_B: Event, tolerance: Real = 1e-10
    ) -> bool:
        return self.probability_measure.are_independent(event_A, event_B, tolerance)
