"""A class modeling a trinomial pricing model."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .geometric_pricing_model import GeometricPricingModel

if TYPE_CHECKING:
    from ....core.base.index import Index
    from ....core.probability_measures.parametrized_probability_measure import (
        ParametrizedProbabilityMeasure,
    )
    from ....core.probability_measures.probability_measure import ProbabilityMeasure
    from ....processes.base.stochastic_process import StochasticProcess
    from ....processes.types.iid_process import IIDProcess
    from ..claims.claim import Claim


class TrinomialPricingModel(GeometricPricingModel):
    """A class modeling a trinomial pricing model."""

    _repr_name = "Trinomial price process"
    _properties = GeometricPricingModel._properties + [
        "_initial_price",
        "_risk_free_rate",
        "_risk_free_gross_return",
        "_up_prob",
        "_middle_prob",
        "_down_prob",
        "_up_factor",
        "_middle_factor",
        "_down_factor",
        "_enum_mode",
        "_sparse_price_array",
    ]

    # --------------------- constructors --------------------- #

    @classmethod
    def generate(
        cls,
        mode: Literal["enum", "sim"],
        initial_price: Real,
        risk_free_rate: Real,
        up_prob: Real,
        down_prob: Real,
        up_factor: Real,
        middle_factor: Real,
        down_factor: Real,
        n_trajectories: int | None = None,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "S",
        random_state: int | np.random.Generator | None = None,
    ) -> TrinomialPricingModel:
        """Pass."""
        if not isinstance(initial_price, Real):
            raise TypeError("initial_price must be a real number.")
        if not isinstance(risk_free_rate, Real):
            raise TypeError("risk_free_rate must be a real number.")
        if initial_price <= 0:
            raise ValueError("initial_price must be positive.")
        if risk_free_rate <= 0:
            raise ValueError("risk_free_rate must be positive.")
        if not isinstance(up_prob, Real):
            raise TypeError("up_prob must be a real number.")
        if not (0 <= up_prob <= 1):
            raise ValueError("up_prob must be in the interval [0, 1].")
        if not isinstance(down_prob, Real):
            raise TypeError("down_prob must be a real number.")
        if not (0 <= down_prob <= 1):
            raise ValueError("down_prob must be in the interval [0, 1].")
        if not isinstance(up_factor, Real):
            raise TypeError("up_factor must be a real number.")
        if up_factor <= 1:
            raise ValueError("up_factor must be greater than 1.")
        if not isinstance(middle_factor, Real):
            raise TypeError("middle_factor must be a real number.")
        if middle_factor <= 0:
            raise ValueError("middle_factor must be positive.")
        if not isinstance(down_factor, Real):
            raise TypeError("down_factor must be a real number.")
        if down_factor >= 1:
            raise ValueError("down_factor must be a less than 1.")
        if up_prob + down_prob > 1:
            raise ValueError("The sum of up_prob and down_prob must be at most 1.")

        index, random_state = cls._validate_and_return_generation_params(
            index=index,
            length=length,
            mode=mode,
            random_state=random_state,
        )
        process = cls(index=index, name=name)
        process._mode = mode
        process._n_trajectories = n_trajectories
        process._random_state = random_state

        process._initial_price = initial_price
        process._risk_free_rate = risk_free_rate
        process._up_prob = up_prob
        process._middle_prob = 1 - up_prob - down_prob
        process._down_prob = down_prob
        process._up_factor = up_factor
        process._middle_factor = middle_factor
        process._down_factor = down_factor

        if mode == "enum":
            return process._enumeration_logic()
        else:
            return process._simulation_logic()

    # --------------------- enumeration methods --------------------- #

    def _enumeration_hook(self) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
        return self.driving_process.prob_measure

    # --------------------- simulation methods --------------------- #

    def _simulation_hook(self) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    # --------------------- properties --------------------- #

    @property
    def initial_price(self) -> Real | None:
        """Pass."""
        return self._initial_price

    @initial_price.setter
    def initial_price(self, value: Real) -> None:
        """Pass."""
        self._initial_price = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def risk_free_rate(self) -> Real | None:
        """Pass."""
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: Real) -> None:
        """Pass."""
        self._risk_free_rate = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def risk_free_gross_return(self) -> Real | None:
        """Pass."""
        if self._risk_free_gross_return is None and self.risk_free_rate is not None:
            self._risk_free_gross_return = self.risk_free_rate + 1

        return self._risk_free_gross_return

    @property
    def up_prob(self) -> Real | None:
        """Pass."""
        return self._up_prob

    @up_prob.setter
    def up_prob(self, value: Real) -> None:
        """Pass."""
        self._up_prob = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def middle_prob(self) -> Real | None:
        """Pass."""
        return self._middle_prob

    @middle_prob.setter
    def middle_prob(self, value: Real) -> None:
        """Pass."""
        self._middle_prob = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def down_prob(self) -> Real | None:
        """Pass."""
        return self._down_prob

    @down_prob.setter
    def down_prob(self, value: Real) -> None:
        """Pass."""
        self._down_prob = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def up_factor(self) -> Real | None:
        """Pass."""
        return self._up_factor

    @up_factor.setter
    def up_factor(self, value: Real) -> None:
        """Pass."""
        self._up_factor = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def middle_factor(self) -> Real | None:
        """Pass."""
        return self._middle_factor

    @middle_factor.setter
    def middle_factor(self, value: Real) -> None:
        """Pass."""
        self._middle_factor = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def down_factor(self) -> Real | None:
        """Pass."""
        return self._down_factor

    @down_factor.setter
    def down_factor(self, value: Real) -> None:
        """Pass."""
        self._down_factor = value
        new_process = self._generate_new_instance()
        self.__dict__.update(new_process.__dict__)

    @property
    def driving_process(self) -> StochasticProcess:
        """Pass."""
        from scipy.stats import multinomial

        from ....processes.types.iid_process import IIDProcess

        if self._driving_process is None:
            T = self.time[1:]
            p_u = self.up_prob
            p_m = self.middle_prob
            p_d = self.down_prob
            u = self.up_factor
            m = self.middle_factor
            d = self.down_factor
            support = {0: u, 1: m, 2: d}

            if self.mode == "enum":
                self._driving_process = IIDProcess.generate(
                    mode="enum",
                    distribution=multinomial(1, [p_u, p_m, p_d]),
                    support=support,
                    index=T,
                    name="driving_process",
                )

            elif self.mode == "sim":
                self._driving_process = IIDProcess.generate(
                    mode="sim",
                    distribution=multinomial(1, [p_u, p_m, p_d]),
                    support=support,
                    n_trajectories=self.n_trajectories,
                    index=T,
                    random_state=self.random_state,
                    name="driving_process",
                )

            else:
                print(self.mode)
                raise ValueError("mode must be either 'enum' or 'sim'.")

        return self._driving_process

    def risk_neutral_probs(self, theta: float) -> tuple[Real, Real, Real]:
        """Later."""
        if not isinstance(theta, Real):
            raise TypeError("The parameter theta must be a real number.")
        if theta <= 0 or theta >= 1:
            raise ValueError("The parameter theta must be in the open interval (0,1).")

        R = self.risk_free_gross_return
        u = self.up_factor
        m = self.middle_factor
        d = self.down_factor

        if R <= d or R >= u:
            raise ValueError(
                "There is arbitrage in the model. The risk-free gross return R must be in the interval [down_factor, up_factor]."
            )

        a = max((m - R) / (m - d), 0)
        b = (u - R) / (u - d)
        q_d = (b - a) * (theta - 1) + b
        q_u = ((m - d) * q_d + (R - m)) / (u - m)
        q_m = ((d - u) * q_d + (u - R)) / (u - m)

        return q_u, q_m, q_d

    @property
    def emms(self) -> ParametrizedProbabilityMeasure:
        """Return the equivalent martingale measures of the model."""
        from scipy.stats import multinomial

        if self._emms is None:

            def parametrization(theta):
                q_u, q_m, q_d = self.risk_neutral_probs(theta=theta)

                Z = IIDProcess(
                    distribution=multinomial(1, [q_u, q_m, q_d]),
                    support=[0, 1, 2],
                    time=self.time[1:],
                ).from_enumeration()

                probabilities = Z.prob_measure.data.values

                return dict(zip(self.domain, probabilities, strict=True))

            self._emms = ParametrizedProbabilityMeasure(
                sample_space=self.domain, parametrization=parametrization, name="Q"
            )

        return self._emms

    # --------------------- finance methods --------------------- #

    def price(self, claim: Claim, emm: ProbabilityMeasure | None = None) -> Real:
        """Price a claim under the model."""
        pass
