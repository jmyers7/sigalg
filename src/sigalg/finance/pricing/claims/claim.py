"""Later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ....processes.base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ..geometric_pricing_models.geometric_pricing_model import GeometricPricingModel


class Claim(ABC, StochasticProcess):
    """Pass."""

    @property
    @abstractmethod
    def payoff(self, model: GeometricPricingModel) -> StochasticProcess:
        """Pass."""
        pass
