"""Marker class for discrete pricing models."""

from abc import ABC, abstractmethod
from numbers import Real

from ...processes.base.stochastic_process import StochasticProcess
from ..claims.claim import Claim


class DiscretePricingModel(StochasticProcess, ABC):
    """Marker class for discrete pricing models."""

    _repr_name = "DiscretePricingModel"
    _str_name = "Discrete pricing model"

    @abstractmethod
    def price(self, claim: Claim) -> Real:
        """Pass."""
        ...

    @abstractmethod
    def price_process(self, claim: Claim) -> StochasticProcess:
        """Pass."""
        ...

    @abstractmethod
    def delta(self, claim: Claim) -> StochasticProcess:
        """Pass."""
        ...

    @abstractmethod
    def replicating_portfolio(self, claim: Claim):
        """Pass."""
        ...
