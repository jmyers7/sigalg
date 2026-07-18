"""Later."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np

from .claim import Claim

if TYPE_CHECKING:
    from ....core.random_objects.random_variable import RandomVariable
    from ..geometric_pricing_models.binomial_pricing_model import BinomialPricingModel
    from ..geometric_pricing_models.geometric_pricing_model import GeometricPricingModel


class AsianOption(Claim):
    """Pass."""

    def __init__(
        self,
        strike: Real,
        option_type: Literal["call", "put"] = "call",
    ):
        if not isinstance(strike, Real) or strike <= 0:
            raise TypeError("Strike price must be a positive real number.")
        if not isinstance(option_type, str) or option_type not in ["call", "put"]:
            raise TypeError("Option type must be either 'call' or 'put'.")

        super().__init__(is_path_independent=False)

        self.strike = strike
        self.option_type = option_type

    @property
    def payoff(self, model: GeometricPricingModel) -> RandomVariable:
        """Return the payoff of the Asian option.

        Returns
        -------
        option : RandomVariable
            A random variable representing the payoff of the Asian option.
        """
        S = model
        K = self.strike

        if self.option_type == "call":
            result = (S.mean() - K) * (S.mean() - K >= 0)
            return result.with_name("AsianCallPayoff")
        elif self.option_type == "put":
            result = (K - S.mean()) * (K - S.mean() >= 0)
            return result.with_name("AsianPutPayoff")

    def _backward_induction_base_case(
        self, model: BinomialPricingModel
    ) -> tuple[np.ndarray, np.ndarray]:
        S = model.data.values
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
