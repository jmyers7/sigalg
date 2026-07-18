"""Later."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np

from ....processes.base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ..geometric_pricing_models.binomial_pricing_model import BinomialPricingModel
    from ..geometric_pricing_models.geometric_pricing_model import GeometricPricingModel


class EuropeanOption(StochasticProcess):
    """Pass."""

    @classmethod
    def from_model(
        cls,
        model: GeometricPricingModel,
        strike: Real,
        option_type: Literal["call", "put"],
        name: Hashable = "Phi",
    ):
        """Pass."""
        S_T = model.last_rv
        factors = [0] * (len(model.time) - 1)

        if option_type == "call":
            factors += [(S_T >= strike) * (S_T - strike)]
        elif option_type == "put":
            factors += [(S_T <= strike) * (strike - S_T)]

        result = cls.concatenate(factors=factors, name=name)
        result._model = model
        result._strike = strike
        result._option_type = option_type

        return result

    # --------------------- properties --------------------- #

    @property
    def model(self) -> GeometricPricingModel | None:
        """Pass."""
        return self._model

    @property
    def strike(self) -> Real | None:
        """Pass."""
        return self._strike

    @property
    def option_type(self) -> Literal["call", "put"] | None:
        """Pass."""
        return self._option_type

    # --------------------- binomial model methods --------------------- #

    def _backward_induction_base_case(self) -> tuple[np.ndarray, np.ndarray]:
        S = self.model.last_rv.data.values
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
            V_curr = (q * V_forward[:-1] + (1 - q) * V_forward[1:]) / R
            Delta_curr = (V_forward[:-1] - V_forward[1:]) / (
                S_forward[:-1] - S_forward[1:]
            )
            B_curr = V_curr - S_curr * Delta_curr
            tau_curr = np.zeros(shape=(len(V_curr),))

            return B_curr, Delta_curr, V_curr, tau_curr

    # --------------------- trinomial model methods --------------------- #
