"""A class modeling a European option."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .claim import Claim

if TYPE_CHECKING:
    from collections.abc import Hashable
    from numbers import Real

    from ....processes.stopping_times.stopping_time import StoppingTime
    from ..discrete_pricing_models.discrete_pricing_model import DiscretePricingModel


class EuropeanOption(Claim):
    """A class modeling a European option."""

    @classmethod
    def from_model(
        cls,
        model: DiscretePricingModel,
        strike: Real,
        option_type: Literal["call", "put"],
        name: Hashable = "Phi",
    ):
        """Pass."""
        data = model.data.copy()
        data.iloc[:, :-1] = 0

        if option_type == "call":
            data.iloc[:, -1] = (data.iloc[:, -1] - strike).clip(lower=0)

        elif option_type == "put":
            data.iloc[:, -1] = (strike - data.iloc[:, -1]).clip(lower=0)

        claim = cls._from_validated(
            data=data,
            sig_alg=model.sig_alg,
            measure=model.measure,
            index_kind=model.index_kind,
            index_name=model.index_name,
            name=name,
        )

        claim._model = model
        claim._strike = strike
        claim._option_type = option_type

        return claim

    # --------------------- properties --------------------- #

    @property
    def model(self) -> DiscretePricingModel | None:
        """Get the underlying model of the option."""
        return self._model

    @property
    def strike(self) -> Real | None:
        """Get the strike price of the option."""
        return self._strike

    @property
    def option_type(self) -> Literal["call", "put"] | None:
        """Get the option type."""
        return self._option_type

    # --------------------- binomial model methods --------------------- #

    def replicating_portfolio(
        self,
    ) -> tuple[
        DiscretePricingModel,
        DiscretePricingModel,
        DiscretePricingModel,
        Real,
        StoppingTime,
    ]:
        """Pass."""
        return self.model._replicating_portfolio(claim=self)

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
