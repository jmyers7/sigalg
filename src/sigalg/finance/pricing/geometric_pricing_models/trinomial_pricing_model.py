from __future__ import annotations

from collections.abc import Hashable
from numbers import Real

import pandas as pd
from scipy.stats import multinomial

from ....core.base.time import Time
from ....core.probability_measures.parametrized_probability_measures import (
    ParametrizedProbabilityMeasures,
)
from ....core.probability_measures.probability_measure import ProbabilityMeasure
from ....processes.base.stochastic_process import StochasticProcess
from ....processes.types.iid_process import IIDProcess
from ..base.claim import Claim
from ..base.geometric_pricing_model import GeometricPricingModel


class TrinomialPricingModel(GeometricPricingModel):
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
        if not isinstance(up_factor, Real) or up_factor <= 0:
            raise TypeError("up_factor must be a positive real number")
        if not isinstance(middle_factor, Real) or middle_factor <= 0:
            raise TypeError("middle_factor must be a positive real number")
        if not isinstance(down_factor, Real) or down_factor <= 0:
            raise TypeError("down_factor must be a positive real number")
        if not isinstance(up_prob, Real) or not (0 <= up_prob <= 1):
            raise TypeError("up_prob must be a real number in [0, 1]")
        if not isinstance(down_prob, Real) or not (0 <= down_prob <= 1):
            raise TypeError("down_prob must be a real number in [0, 1]")
        if up_prob + down_prob > 1:
            raise ValueError("The sum of up_prob and down_prob must be at most 1")

        self.up_factor = up_factor
        self.middle_factor = middle_factor
        self.down_factor = down_factor
        self.up_prob = up_prob
        self.down_prob = down_prob
        self.middle_prob = 1 - up_prob - down_prob

        super().__init__(
            initial_price=initial_price,
            risk_free_rate=risk_free_rate,
            time=time,
            name=name,
        )

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.from_enumeration().cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    @property
    def driving_process(self) -> StochasticProcess:
        """Pass."""
        if self._driving_process is None:
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
                time=self.time[1:],
                name="driving_process",
            )

            self._driving_process = Z

        return self._driving_process

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        return self.driving_process.probability_measure.with_name(name=name)

    def emms(self) -> ParametrizedProbabilityMeasures:
        """Return the equivalent martingale measures of the model."""
        pass

    # --------------------- finance methods --------------------- #

    @property
    def is_complete(self) -> bool:
        """Whether the model is complete."""
        pass

    def price(self, claim: Claim, emm: ProbabilityMeasure | None = None) -> Real:
        """Price a claim under the model."""
        pass
