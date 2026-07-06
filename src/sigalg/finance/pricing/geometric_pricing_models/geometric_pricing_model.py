"""Abstract base class for geometric pricing models."""

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


class GeometricPricingModel(ABC, StochasticProcess):
    """Abstract base class for geometric pricing models."""

    _properties = StochasticProcess._properties + [
        "_driving_process",
        "_emms",
    ]
    _repr_name = "Geometric price process"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_price_data(
        cls,
        initial_price: Real,
        risk_free_rate: Real,
        index: Time | None = None,
        name: Hashable = "S",
    ):
        """Initialize a geometric pricing model from basic pricing data.

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
            If `initial_price` or `risk_free_rate` is not a real number.
        ValueError
            If `initial_price` or `risk_free_rate` is not positive.
        """
        if not isinstance(initial_price, Real):
            raise TypeError("initial_price must be a real number.")
        if not isinstance(risk_free_rate, Real):
            raise TypeError("risk_free_rate must be a real number.")
        if initial_price <= 0:
            raise ValueError("initial_price must be positive.")
        if risk_free_rate <= 0:
            raise ValueError("risk_free_rate must be positive.")

        process = cls(index=index, name=name)
        process.initial_price = initial_price
        process.risk_free_rate = risk_free_rate
        process.risk_free_gross_return = 1 + risk_free_rate

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
