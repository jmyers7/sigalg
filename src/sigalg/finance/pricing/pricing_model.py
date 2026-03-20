"""Later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import TYPE_CHECKING

from ...processes.base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ..claims.claim import Claim


class PricingModel(ABC, StochasticProcess):
    """Abstract base class for various types of pricing models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enum_mode: str | None = None

    @abstractmethod
    def replicating_portfolio(
        self, claim: Claim
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        """Compute the replicating portfolio for a given contingent claim relative to this pricing model."""
        pass
