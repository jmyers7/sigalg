from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace


class ProbabilityMeasure:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        probabilities: dict[Hashable, Real],
        sample_space: SampleSpace | None = None,
        name: str = "P",
    ) -> None:
        self._validate_parameters(
            probabilities=probabilities, sample_space=sample_space, name=name
        )
        if sample_space is None:
            self.sample_space = self._generate_sample_space(probabilities)
        else:
            self.sample_space = sample_space
        self.probabilities = probabilities
        self.values: pd.Series = pd.Series(probabilities, name=name)
        self.values.index.name = self.sample_space.name
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError("name must be a string.")
        self._name = new_name
        self.values.name = new_name

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
        prob_intersection = self.P(event_A & event_B)
        return bool(abs(prob_intersection - prob_A * prob_B) < tolerance)

    @staticmethod
    def _generate_sample_space(probabilities: dict[Hashable, Real]) -> SampleSpace:
        from ..base.sample_space import SampleSpace

        indices = list(probabilities.keys())
        return SampleSpace(indices)

    # --------------------- factory methods --------------------- #

    @classmethod
    def uniform(cls, sample_space: SampleSpace) -> ProbabilityMeasure:
        n = len(sample_space)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on empty sample space."
            )
        probabilities = dict.fromkeys(sample_space.values, 1.0 / n)
        return cls(probabilities=probabilities, sample_space=sample_space)

    # --------------------- access methods --------------------- #

    def __call__(self, key: Hashable | list[Hashable] | Event) -> Real:
        from ..base import Event

        if isinstance(key, Event):
            if key.sample_space != self.sample_space:
                raise ValueError("Event must be from the same sample space.")
            return self.values.loc[list(key.values)].sum()
        elif isinstance(key, list):
            for idx in key:
                if idx not in self.probabilities:
                    raise KeyError(f"Index '{idx}' not found in sample space.")
            return sum(self.probabilities[idx] for idx in key)
        else:
            if key not in self.probabilities:
                raise KeyError(f"Index '{key}' not found in sample space.")
            return self.probabilities[key]

    def __getitem__(self, key: Hashable | list[Hashable] | Event) -> Real:
        return self(key)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Probability measure '{self.name}':\n{self.values.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProbabilityMeasure):
            return False
        if self.sample_space != other.sample_space:
            return False
        return self.values.equals(other.values)

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        probabilities: dict[Hashable, Real],
        sample_space: SampleSpace | None,
        name: str,
    ) -> None:
        from ..base import SampleSpace

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("If provided, sample_space must be a SampleSpace instance.")
        if not isinstance(probabilities, dict):
            raise TypeError("probabilities must be a dictionary.")

        if sample_space is not None:
            prob_indices = set(probabilities.keys())
            space_indices = set(sample_space.values)
            if prob_indices != space_indices:
                raise ValueError("Probabilities keys must match sample space indices.")

        for sample_id, prob in probabilities.items():
            if not isinstance(prob, Real):
                raise TypeError(
                    f"Probability for '{sample_id}' must be a Real number, got {type(prob)}."
                )
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"Probability for '{sample_id}' must be in [0, 1], got {prob}."
                )

        total = sum(probabilities.values())
        if not abs(total - 1.0) < 1e-10:
            raise ValueError(f"Probabilities must sum to 1, got {total}.")

        if not isinstance(name, str):
            raise ValueError("'name' must be a string.")


class ProbabilityMeasureMethods:
    @property
    def probabilities(self) -> dict[Hashable, Real]:
        return self.probability_measure.probabilities

    def P(self, key: Hashable | Event) -> Real:
        return self.probability_measure(key)

    def conditional_probability(self, event_A: Event, event_B: Event) -> Real:
        return self.probability_measure.conditional_probability(event_A, event_B)

    def are_independent(
        self, event_A: Event, event_B: Event, tolerance: Real = 1e-10
    ) -> bool:
        return self.probability_measure.are_independent(event_A, event_B, tolerance)
