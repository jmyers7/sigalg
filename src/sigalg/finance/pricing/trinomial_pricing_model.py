from __future__ import annotations

from collections.abc import Hashable
from numbers import Real

import pandas as pd
from scipy.stats import multinomial

from ...core.base.time import Time
from ...processes.base.stochastic_process import StochasticProcess
from ...processes.types.iid_process import IIDProcess
from ..claims.claim import Claim
from .pricing_model import PricingModel


class TrinomialPricingModel(PricingModel):
    """Pass."""

    def __init__(
        self,
        initial_price: Real,
        up_factor: Real,
        middle_factor: Real,
        down_factor: Real,
        up_prob: Real,
        down_prob: Real,
        risk_free_rate: Real,
        time: Time | None = None,
        name: Hashable | None = "S",
    ) -> None:
        self.initial_price = initial_price
        self.up_factor = up_factor
        self.middle_factor = middle_factor
        self.down_factor = down_factor
        self.up_prob = up_prob
        self.down_prob = down_prob
        self.middle_prob = 1 - up_prob - down_prob
        self.risk_free_rate = risk_free_rate
        self.risk_free_gross_return = 1 + risk_free_rate

        super().__init__(
            time=time,
            is_discrete_time=True,
            is_discrete_state=True,
            name=name,
        )

        self._driving_process: StochasticProcess | None = None

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.from_enumeration().cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    @property
    def driving_process(self) -> StochasticProcess:
        """Pass."""
        if self._driving_process is None:
            T = self.time[1:]
            u = self.up_factor
            m = self.middle_factor
            d = self.down_factor
            p_u = self.up_prob
            p_m = self.middle_prob
            p_d = self.down_prob

            support = {0: u, 1: m, 2: d}

            Z = IIDProcess(
                distribution=multinomial(1, [p_u, p_m, p_d]),
                support=support,
                time=T,
                name="driving_process",
            )

            self._driving_process = Z

        return self._driving_process

    # --------------------- finance methods --------------------- #

    def replicating_portfolio(
        self, claim: Claim
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        """Pass."""
        pass
