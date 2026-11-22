"""Marker class for continuous pricing models."""

from abc import ABC, abstractmethod
from numbers import Real

from ...processes.base.stochastic_process import StochasticProcess
from ..claims.claim import Claim


class ContinuousPricingModel(StochasticProcess, ABC):
    """Marker class for continuous pricing models."""

    _repr_name = "ContinuousPricingModel"
    _str_name = "Continuous pricing model"
    _default_y_axis_label = "price"

    @abstractmethod
    def price(self, claim: Claim, **kwargs) -> Real:
        """Pass."""
        ...

    @abstractmethod
    def price_process(self, claim: Claim, **kwargs) -> StochasticProcess:
        """Pass."""
        ...
