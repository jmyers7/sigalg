"""Later."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ....core.random_objects.random_variable import RandomVariable
from ..base.claim import Claim
from ..base.geometric_pricing_model import GeometricPricingModel


class EuropeanOption(Claim):
    r"""Model a European option contingent claim.

    The European option is a claim comes in two types: *call* and *put*. The payoff of a European call option is given by

    $$
    \max(S_T - K, 0),
    $$

    while the payoff of a European put option is given by

    $$
    \max(K - S_T, 0),
    $$

    where $S_T$ is the underlying asset price at maturity and $K$ is the strike price.

    Parameters
    ----------
    pricing_model : PricingModel
        A pricing model representing the underlying asset price dynamics.
    strike : Real
        The strike price of the European option.
    option_type : str, default "call"
        The type of the European option. It can be either "call" or "put".

    Raises
    ------
    TypeError
        If the strike price is not a positive real number, or if the pricing model is not a PricingModel, or if the option type is not "call" or "put".
    """

    def __init__(
        self,
        pricing_model: GeometricPricingModel,
        strike: Real,
        option_type: str = "call",
    ):
        if not isinstance(strike, Real) or strike <= 0:
            raise TypeError("Strike price must be a positive real number.")
        if not isinstance(pricing_model, GeometricPricingModel):
            raise TypeError("Pricing model must be a PricingModel.")
        if not isinstance(option_type, str) or option_type not in ["call", "put"]:
            raise TypeError("Option type must be either 'call' or 'put'.")

        super().__init__(is_path_independent=True)

        self.pricing_model = pricing_model
        self.strike = strike
        self.option_type = option_type

    @property
    def payoff(self) -> RandomVariable:
        """Return the payoff of the European option.

        Returns
        -------
        option : RandomVariable
            A random variable representing the payoff of the European option.
        """
        if self.pricing_model.data is None:
            raise ValueError(
                "Price trajectories of the underlying asset must be enumerated before computing the payoff."
            )

        price = self.pricing_model.last_rv
        K = self.strike

        if self.option_type == "call":
            result = (price - K) * (price - K >= 0)
            return result.with_name("EuropeanCallPayoff")
        elif self.option_type == "put":
            result = (K - price) * (K - price >= 0)
            return result.with_name("EuropeanPutPayoff")

    def _backward_induction_base_case(self) -> np.ndarray:
        S = self.pricing_model.last_rv.data.values
        K = self.strike

        if self.option_type == "call":
            return np.maximum(S - K, 0)
        elif self.option_type == "put":
            return np.maximum(K - S, 0)

    def _backward_induction_dense(
        self,
        V_next: np.ndarray,
        S_next: np.ndarray,
        S_curr: np.ndarray,
        strike: float,
        risk_free_rate: float,
        risk_neutral_prob: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = 1 + risk_free_rate
        q = risk_neutral_prob

        V_curr = (V_next.reshape(-1, 2) @ np.array([q, 1 - q])) / R
        Delta_curr = (
            np.diff(V_next.reshape(-1, 2)).squeeze()
            / np.diff(S_next.reshape(-1, 2)).squeeze()
        )
        B_curr = V_curr - S_curr * Delta_curr
        tau_curr = np.zeros(shape=(len(V_curr),))

        return B_curr, Delta_curr, V_curr, tau_curr
