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


class AmericanOption(Claim):
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

        super().__init__(is_path_independent=True)

        self.strike = strike
        self.option_type = option_type

    @property
    def payoff(self, model: GeometricPricingModel) -> RandomVariable:
        """Pass."""
        pass

    # --------------------- binomial pricing methods --------------------- #

    def _backward_induction_base_case(
        self, model: BinomialPricingModel
    ) -> tuple[np.ndarray, np.ndarray]:
        S = model.last_rv.data.values
        K = self.strike

        if self.option_type == "call":
            exercise_value = np.maximum(S - K, 0)
            tau = np.where(exercise_value == 0, 0, 1)
            return exercise_value, tau
        elif self.option_type == "put":
            exercise_value = np.maximum(K - S, 0)
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
        strike = self.strike

        if enum_mode == "dense":
            continuation = (V_forward.reshape(-1, 2) @ np.array([q, 1 - q])) / R

            if self.option_type == "call":
                intrinsic = np.maximum(S_curr - strike, 0)
            elif self.option_type == "put":
                intrinsic = np.maximum(strike - S_curr, 0)

            V_curr = np.maximum(intrinsic, continuation)
            Delta_curr = (
                np.diff(V_forward.reshape(-1, 2)).squeeze()
                / np.diff(S_forward.reshape(-1, 2)).squeeze()
            )
            B_curr = ((V_forward - S_forward * np.repeat(Delta_curr, repeats=2)) / R)[
                ::2
            ]
            tau_curr = intrinsic > continuation

            return B_curr, Delta_curr, V_curr, tau_curr

        elif enum_mode == "sparse":
            continuation = (q * V_forward[:-1] + (1 - q) * V_forward[1:]) / R

            if self.option_type == "call":
                intrinsic = np.maximum(S_curr - strike, 0)
            elif self.option_type == "put":
                intrinsic = np.maximum(strike - S_curr, 0)

            V_curr = np.maximum(intrinsic, continuation)
            Delta_curr = (V_forward[:-1] - V_forward[1:]) / (
                S_forward[:-1] - S_forward[1:]
            )

            B_curr = (V_forward[:-1] - S_forward[:-1] * Delta_curr) / R
            tau_curr = intrinsic > continuation

            return B_curr, Delta_curr, V_curr, tau_curr
