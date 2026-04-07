"""Later."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ....core.random_objects.random_variable import RandomVariable
from ..geometric_pricing_models.geometric_pricing_model import GeometricPricingModel
from .claim import Claim


class AsianOption(Claim):
    r"""Model an Asian option contingent claim.

    The Asian option is a claim comes in two types: *call* and *put*. The payoff of an Asian call option is given by

    $$
    \max(\bar{S}_T - K, 0),
    $$

    while the payoff of an Asian put option is given by

    $$
    \max(K - \bar{S}_T, 0),
    $$

    where $\bar{S}_T$ is the average underlying asset price over the option's life and $K$ is the strike price.

    Parameters
    ----------
    pricing_model : PricingModel
        A pricing model representing the underlying asset price dynamics.
    strike : Real
        The strike price of the Asian option.
    option_type : str, default "call"
        The type of the Asian option. It can be either "call" or "put".

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

        super().__init__(is_path_independent=False)

        self.pricing_model = pricing_model
        self.strike = strike
        self.option_type = option_type

    @property
    def payoff(self) -> RandomVariable:
        """Return the payoff of the Asian option.

        Returns
        -------
        option : RandomVariable
            A random variable representing the payoff of the Asian option.
        """
        if self.pricing_model.data is None:
            raise ValueError(
                "Price trajectories of the underlying asset must be enumerated before computing the payoff."
            )
        if self.pricing_model.enum_mode == "sparse":
            raise ValueError(
                "Payoff of an Asian option cannot be computed from a pricing model that is sparsely enumerated."
            )

        S = self.pricing_model
        K = self.strike

        if self.option_type == "call":
            result = (S.mean() - K) * (S.mean() - K >= 0)
            return result.with_name("AsianCallPayoff")
        elif self.option_type == "put":
            result = (K - S.mean()) * (K - S.mean() >= 0)
            return result.with_name("AsianPutPayoff")

    def _backward_induction_base_case(self) -> tuple[np.ndarray, np.ndarray]:
        S = self.pricing_model.data.values
        K = self.strike

        if self.option_type == "call":
            exercise_value = np.maximum(np.mean(S, axis=1) - K, 0)
            tau = np.where(exercise_value == 0, 0, 1)
            return exercise_value, tau
        elif self.option_type == "put":
            exercise_value = np.maximum(K - np.mean(S, axis=1), 0)
            tau = np.where(exercise_value == 0, 0, 1)
            return exercise_value, tau

    def _backward_induction(
        self,
        enum_mode: str,
        V_forward: np.ndarray,
        S_forward: np.ndarray,
        S_curr: np.ndarray,
        strike: float,
        risk_free_rate: float,
        risk_neutral_prob: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = 1 + risk_free_rate
        q = risk_neutral_prob

        if enum_mode == "dense":
            V_curr = (V_forward.reshape(-1, 2) @ np.array([q, 1 - q])) / R
            Delta_curr = (
                np.diff(V_forward.reshape(-1, 2)).squeeze()
                / np.diff(S_forward.reshape(-1, 2)).squeeze()
            )
            B_curr = V_curr - S_curr * Delta_curr
            tau_curr = np.zeros(shape=(len(V_curr),))

            return B_curr, Delta_curr, V_curr, tau_curr

        elif enum_mode == "sparse":
            raise NotImplementedError(
                "Backward induction for Asian options is not implemented for a pricing model in sparse enumeration mode."
            )
