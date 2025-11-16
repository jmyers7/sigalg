from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd

from ..probability_measures import ProbabilityMeasure
from ..sigma_algebras import SigmaAlgebra
from .event import Event
from .sample_space import SampleSpace


class ProbabilitySpace:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        sigma_algebra: SigmaAlgebra = None,
        probability_measure: ProbabilityMeasure = None,
    ) -> None:
        self._validate_parameters(sample_space, sigma_algebra, probability_measure)
        self._sample_space = sample_space
        if sigma_algebra is None:
            self._sigma_algebra = SigmaAlgebra.power_set(sample_space)
        else:
            self._sigma_algebra = sigma_algebra
        if probability_measure is None:
            self._probability_measure = ProbabilityMeasure.uniform(sample_space)
        else:
            self._probability_measure = probability_measure

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        return self._probability_measure

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        return self._sigma_algebra

    @property
    def index(self) -> pd.Index:
        return self.sample_space.index

    # --------------------- setter methods --------------------- #

    def set_sigma_algebra(self, sigma_algebra: SigmaAlgebra) -> None:
        if not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
        if sigma_algebra.sample_space != self.sample_space:
            raise ValueError("sigma_algebra must be defined on this sample space.")
        self._sigma_algebra = sigma_algebra

    def set_probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        if not isinstance(probability_measure, ProbabilityMeasure):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure instance."
            )
        if probability_measure.sample_space != self.sample_space:
            raise ValueError(
                "probability_measure must be defined on this sample space."
            )
        self._probability_measure = probability_measure

    # --------------------- probability methods --------------------- #

    def P(self, key: Hashable | Event) -> float:
        return self.probability_measure(key)

    def get_event(self, event_indices: list[Hashable]) -> Event:
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of Hashable items.")
        return self[event_indices]

    def get_event_as_probability_space(
        self, event_indices: list[Hashable]
    ) -> ProbabilitySpace:
        event = self.get_event(event_indices)
        event_probability = self.probability_measure(event)
        if event_probability < 1e-10:
            raise ValueError(
                "Cannot create ProbabilitySpace for event with zero probability."
            )
        event_sample_space = SampleSpace(list(event.index))
        conditional_probabilities = {
            idx: self.probability_measure(idx) / event_probability
            for idx in event.index
        }
        event_probability_measure = ProbabilityMeasure(
            sample_space=event_sample_space, probabilities=conditional_probabilities
        )
        event_atom_ids = {idx: self.sigma_algebra.atom_ids[idx] for idx in event.index}
        event_sigma_algebra = SigmaAlgebra(
            sample_space=event_sample_space, atom_ids=event_atom_ids
        )
        return ProbabilitySpace(
            sample_space=event_sample_space,
            sigma_algebra=event_sigma_algebra,
            probability_measure=event_probability_measure,
        )

    def conditional_probability(self, event_A: Event, event_B: Event) -> float:
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
        intersection_indices = [idx for idx in event_A.index if idx in event_B.index]
        if not intersection_indices:
            return 0.0
        intersection_event = self[intersection_indices]
        prob_intersection = self.P(intersection_event)
        return prob_intersection / prob_B

    def are_independent(
        self, event_A: Event, event_B: Event, tolerance: float = 1e-10
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
        intersection_indices = [idx for idx in event_A.index if idx in event_B.index]
        if not intersection_indices:
            prob_intersection = 0.0
        else:
            intersection_event = self[intersection_indices]
            prob_intersection = self.P(intersection_event)
        return abs(prob_intersection - prob_A * prob_B) < tolerance

    def sample(self, size: int = 1, random_state: int | None = None) -> list[Hashable]:
        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer.")
        if random_state is not None:
            np.random.seed(random_state)
        outcomes = list(self.sample_space)
        probabilities = [self.P(outcome) for outcome in outcomes]
        samples = np.random.choice(outcomes, size=size, p=probabilities)
        return [
            outcomes[outcomes.index(s)] if hasattr(outcomes, "index") else s
            for s in samples
        ]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        return len(self.sample_space)

    def __getitem__(self, key: Hashable | list[Hashable]) -> Hashable | Event:
        return self.sample_space[key]

    def __iter__(self) -> iter:
        return iter(self.sample_space)

    # --------------------- equality & hashing --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProbabilitySpace):
            return False
        return (
            self._sample_space == other._sample_space
            and self._sigma_algebra == other._sigma_algebra
            and self._probability_measure == other._probability_measure
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"ProbabilitySpace({list(self.sample_space)}, P={self._probability_measure.to_pandas()})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace,
        sigma_algebra: SigmaAlgebra,
        probability_measure: ProbabilityMeasure,
    ) -> None:
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if sigma_algebra is not None and not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
        if sigma_algebra is not None and sigma_algebra.sample_space != sample_space:
            raise ValueError("sigma_algebra must be defined on the given sample_space.")
        if probability_measure is not None and not isinstance(
            probability_measure, ProbabilityMeasure
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure instance."
            )
        if (
            probability_measure is not None
            and probability_measure.sample_space != sample_space
        ):
            raise ValueError(
                "probability_measure must be defined on the given sample_space."
            )
