"""Marker class for discrete pricing models."""

from abc import ABC, abstractmethod
from numbers import Real

from ...processes.base.stochastic_process import StochasticProcess
from ..claims.claim import Claim


class DiscretePricingModel(StochasticProcess, ABC):
    """Marker class for discrete pricing models."""

    _repr_name = "DiscretePricingModel"
    _str_name = "Discrete pricing model"
    _default_y_axis_label = "price"

    @abstractmethod
    def price(self, claim: Claim, **kwargs) -> Real:
        """Pass."""
        ...

    @abstractmethod
    def price_process(self, claim: Claim, **kwargs) -> StochasticProcess:
        """Pass."""
        ...
