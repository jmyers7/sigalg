"""Later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from numbers import Real

from ....core.base.time import Time
from ....core.probability_measures.parametrized_probability_measure import (
    ParametrizedProbabilityMeasure,
)
from ....core.probability_measures.probability_measure import ProbabilityMeasure
from ....processes.base.stochastic_process import StochasticProcess
from ..claims.claim import Claim


# TODO: Expand docstring
class GeometricPricingModel(ABC, StochasticProcess):
    """Abstract base class for geometric pricing models.

    Parameters
    ----------
    initial_price : Real
        The initial price of the underlying asset.
    risk_free_rate : Real
        The risk-free rate of return.
    time : Time | None, default=None
        The time index of the model. If `None`, a time index will be generated later through data generation methods.
    name : Hashable | None, default="S"
        The name of the model.

    Raises
    ------
    TypeError
        If `initial_price` is not a positive real number, or if `risk_free_rate` is not a positive real number.
    """

    def __init__(
        self,
        initial_price: Real,
        risk_free_rate: Real,
        time: Time | None = None,
        name: Hashable | None = "S",
    ):
        if not isinstance(initial_price, Real) or initial_price <= 0:
            raise TypeError("initial_price must be a positive real number")
        if not isinstance(risk_free_rate, Real) or risk_free_rate <= 0:
            raise TypeError("risk_free_rate must be a positive real number")

        self.initial_price = initial_price
        self.risk_free_rate = risk_free_rate
        self.risk_free_gross_return = 1 + risk_free_rate
        super().__init__(time=time, is_discrete_state=True, name=name)

        # caches
        self._driving_process: StochasticProcess | None = None
        self._emms: ParametrizedProbabilityMeasure | ProbabilityMeasure | None = None

    # --------------------- data generation methods --------------------- #

    @property
    @abstractmethod
    def driving_process(self) -> StochasticProcess:
        """The driving process of the model."""
        pass

    # --------------------- probability methods --------------------- #

    @property
    @abstractmethod
    def emms(self) -> ParametrizedProbabilityMeasure | ProbabilityMeasure:
        """Return the equivalent martingale measures of the model."""
        pass

    @property
    @abstractmethod
    def risk_neutral_probs(self) -> tuple:
        """Return the risk-neutral probabilities of the model."""
        pass

    # --------------------- finance methods --------------------- #

    @abstractmethod
    def price(self, claim: Claim, emm: ProbabilityMeasure | None = None) -> Real:
        """Price a claim under the model given an equivalent martingale measure."""
        pass
