"""Abstract base class for geometric pricing models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real

from ....core.measures.parametrized_probability_measure import (
    ParametrizedProbabilityMeasure,
)
from ....core.measures.probability_measure import ProbabilityMeasure
from ....processes.base.stochastic_process import StochasticProcess
from ..claims.claim import Claim


class GeometricPricingModel(ABC, StochasticProcess):
    """Abstract base class for geometric pricing models."""

    _properties = StochasticProcess._properties + [
        "_driving_process",
        "_emms",
    ]

    # --------------------- data generation methods --------------------- #

    @property
    @abstractmethod
    def driving_process(self) -> StochasticProcess:
        """The driving process of the model."""
        pass

    # --------------------- probability methods --------------------- #

    @property
    @abstractmethod
    def EMMs(self) -> ParametrizedProbabilityMeasure | ProbabilityMeasure:
        """Return the equivalent martingale measures of the model."""
        pass

    @abstractmethod
    def risk_neutral_probs(self) -> tuple:
        """Return the risk-neutral probabilities of the model."""
        pass

    # --------------------- finance methods --------------------- #

    @abstractmethod
    def price(self, claim: Claim, emm: ProbabilityMeasure | None = None) -> Real:
        """Price a claim under the model given an equivalent martingale measure."""
        pass
