"""Class representing data for Black-Scholes pricing."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .continuous_pricing_model import ContinuousPricingModel

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...processes.base.stochastic_process import StochasticProcess
    from ...validation.index_validator import IndexLike
    from ..claims.claim import Claim


class BlackScholes(ContinuousPricingModel):
    """Class representing data for Black-Scholes pricing."""

    _repr_name = "BlackScholesModel"
    _str_name = "Black-Scholes model"

    # --------------------- constructors --------------------- #

    @ContinuousPricingModel.generator
    def generate(
        cls,
        initial_price: Real,
        risk_free_rate: Real,
        volatility: Real,
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        domain_name: Hashable | None = None,
        index: IndexLike | None = None,
        length: int | None = None,
        name: Hashable = "S",
        random_state: int | np.random.Generator | None = None,
    ) -> BlackScholes:
        r"""Generate trajectories of the Black Scholes model by Monte Carlo simulation.

        Parameters
        ----------
        mode : Literal["enum", "sim"], default="sim"
            Not used. (This parameter is only included for consistency with the parent class.)
        n_trajectories : int | None, default=None
            The number of trajectories to simulate.
        index : Time | None, default=None
            The time index of the model. Must be provided. (The default is only included for consistency with the parent class.)
        length : int | None, default=None
            Not used. (This parameter is only included for consistency with the parent class.)
        name : Hashable, default="S"
            The name of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional random state for reproducibility.

        Returns
        -------
        model : BlackScholes
            The Black-Scholes pricing model.
        """
        if not isinstance(initial_price, Real) or initial_price <= 0:
            raise ValueError("initial_price must be a positive real number.")
        if not isinstance(risk_free_rate, Real) or risk_free_rate <= 0:
            raise ValueError("risk_free_rate must be a positive real number.")
        if not isinstance(volatility, Real) or volatility <= 0:
            raise ValueError("volatility must be a positive real number.")

        return {
            "initial_price": initial_price,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
        }

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        risk_free_rate: Real,
        volatility: Real,
        name: Hashable = "S",
    ) -> BlackScholes:
        """Pass."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame")
        if not isinstance(risk_free_rate, Real) or risk_free_rate <= 0:
            raise ValueError("risk_free_rate must be a positive real number.")
        if not isinstance(volatility, Real) or volatility <= 0:
            raise ValueError("volatility must be a positive real number.")

        process = cls.with_uniform(mapping=data, index_kind="Time", name=name)
        process._risk_free_rate = risk_free_rate
        process._volatility = volatility

        return process

    # --------------------- properties --------------------- #

    @property
    def risk_free_rate(self) -> Real:
        """Pass."""
        return self._risk_free_rate

    @property
    def volatility(self) -> Real:
        """Pass."""
        return self._volatility

    # --------------------- finance methods --------------------- #

    def price(self, claim: Claim, **kwargs) -> Real:
        """Compute the Black-Scholes price of a European call or put option.

        Parameters
        ----------
        claim : Claim
            The contingent claim to price.

        Returns
        -------
        price : Real
            The Black-Scholes price of the option.
        """
        ...

    def price_process(self, claim: Claim, **kwargs) -> StochasticProcess:
        """Pass."""
        from scipy.stats import norm

        from ...processes.base.stochastic_process import StochasticProcess

        K, r, sigma = (claim.strike, self.risk_free_rate, self.volatility)

        s = np.array(self)[:, :-1]
        tau = (self.index[-1] - np.array(self.index))[:-1]

        d_plus = (np.log(s / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
        d_neg = d_plus - sigma * np.sqrt(tau)
        N = norm.cdf

        if claim.option_type == "call":
            pricing_arr = s * N(d_plus) - K * np.exp(-r * tau) * N(d_neg)

        else:
            pricing_arr = K * np.exp(-r * tau) * N(-d_neg) - s * N(-d_plus)

        data = pd.DataFrame(
            pricing_arr, index=self.domain.data, columns=self.index.data[:-1]
        )
        data.insert(loc=data.shape[1], column=self.index[-1], value=claim.last_rv)

        return StochasticProcess._from_validated(
            data=data,
            sig_alg=self.sig_alg,
            measure=self.measure,
            index_kind="Time",
            index_name=self.index_name,
            name="X",
        )

    def delta(self, claim: Claim, **kwargs) -> Real:
        """Compute the delta process of a European call or put option.

        Returns
        -------
        delta : StochasticProcess
            The delta process of the option.
        """
        from scipy.stats import norm

        from ...processes.base.stochastic_process import StochasticProcess

        K, r, sigma = (claim.strike, self.risk_free_rate, self.volatility)

        s = np.array(self)[:, :-1]
        tau = (self.index[-1] - np.array(self.index))[:-1]

        d_plus = (np.log(s / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
        N = norm.cdf

        if claim.option_type == "call":
            delta_arr = N(d_plus)
        else:
            delta_arr = N(d_plus) - 1

        data = pd.DataFrame(
            delta_arr, index=self.domain.data, columns=self.index.data[:-1]
        )
        # data.insert(loc=data.shape[1], column=self.index[-1], value=claim.last_rv)

        return StochasticProcess._from_validated(
            data=data,
            sig_alg=self.sig_alg,
            measure=self.measure,
            index_kind="Time",
            index_name=self.index_name,
            name="Delta",
        )

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        dt = self._index.data[1] - self._index.data[0]
        initial_time = self._index.data[0]

        increments_arr = self._random_state.normal(
            scale=np.sqrt(dt), size=(self._n_trajectories, len(self._index) - 1)
        )
        increments = pd.DataFrame(increments_arr, columns=self._index.data[1:])

        drift_term = pd.DataFrame(
            self._risk_free_rate * dt,
            index=increments.index,
            columns=increments.columns,
        ).cumsum(axis=1)

        self._driver_data = increments.cumsum(axis=1)
        log_trajectories = drift_term + self._volatility * self._driver_data
        trajectories = np.exp(log_trajectories) * self._initial_price

        self._driver_data.insert(loc=0, column=initial_time, value=0.0)
        trajectories.insert(loc=0, column=initial_time, value=self._initial_price)

        return trajectories
