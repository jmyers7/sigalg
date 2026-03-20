"""Later."""

from __future__ import annotations

from numbers import Real

import numpy as np

from ...core.random_objects.random_variable import RandomVariable
from ...processes.base.stochastic_process import StochasticProcess
from ..pricing.pricing_model import PricingModel
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
        pricing_model: PricingModel,
        strike: Real,
        option_type: str = "call",
    ):
        if not isinstance(strike, Real) or strike <= 0:
            raise TypeError("Strike price must be a positive real number.")
        if not isinstance(pricing_model, PricingModel):
            raise TypeError("Pricing model must be a PricingModel.")
        if not isinstance(option_type, str) or option_type not in ["call", "put"]:
            raise TypeError("Option type must be either 'call' or 'put'.")

        self.pricing_model = pricing_model
        self.strike = strike
        self.option_type = option_type

    @property
    def payout(self) -> RandomVariable:
        """Return the payout of the Asian option.

        Returns
        -------
        option : RandomVariable
            A random variable representing the payoff of the Asian option.
        """
        S = self.pricing_model
        K = self.strike

        if self.option_type == "call":
            result = (S.mean() - K) * (S.mean() - K >= 0)
            return result.with_name("AsianCallPayout")
        elif self.option_type == "put":
            result = (K - S.mean()) * (K - S.mean() >= 0)
            return result.with_name("AsianPutPayout")

    def replicating_portfolio(
        self,
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        r"""Compute the replicating portfolio for the Asian option given the pricing model.

        The core idea of a *replicating portfolio* is this: Suppose that an individual sells a contingent claim on an underlying asset (generically called an *underlying*). The seller accepts a premium from the buyer for the claim at time $t=0$, and then at some specified maturity time $t=T$, the seller must pay the exercise value of the claim to the buyer. The claim is a *derivative*, in the sense that its value depends on (or derives from) the price of the underlying. The seller is thus interested in hedging their short position on the claim against an increase in the price of the underlying, which would increase the exercise value of the claim that the seller would owe the buyer.

        The underlying asset is assumed to be traded in a market that includes a bank account with risk-free, per-period interest rate $r$. The seller's hedging strategy is to trade in the underlying asset itself, as well as hold a cash position at the bank, so that when the contingent claim matures, the seller's portfolio will cover the exercise value owed to the buyer.

        The replicating portfolio thus consists of a pair $(B_t,N_t)$ of processes, indexed $t=0,1,\ldots,T-1$, where $B_t$ represents the cash position at time $t$, and $N_t$ counts the number of units of the underlying held in the portfolio at time $t$. A third process $V_t$ represents the total value of the portfolio, given by

        $$
        V_t = B_t + S_t N_t,
        $$

        where $S_t$ is the price of the underlying at time $t$. A positive value of $B_t$ represents money held in the bank accruing interest for the seller at rate $r$, while a negative value represents a loan on which the seller pays interest at rate $r$. A positive value of $N_t$ represents a *long position* on the underlying, while a negative value represents a *short position*.

        The replicating portfolio is *self-financing*, in the sense that

        $$
        V_t = (1+r) B_{t-1} + S_t N_{t-1}
        $$

        for each $t=1,2,\ldots,T$. The right-hand side of this equation represents the evolution of the value of the portfolio over the time interval $[t-1,t]$, in which the amount $B_{t-1}$ in the bank accrues interest at rate $r$ and the price of the underlying changes from $S_{t-1}$ to $S_t$. This equation says that this evolved value of the old portfolio is equal to the value $V_t$ of the new portofolio at time $t$.

        The existence of the replicating portfolio also allows us to determine a fair, "risk-neutral" premium for the contingent claim paid by the buyer. Under the no-arbitrage assumption, this premium should coincide with the initial price

        $$
        V_0 = B_0 + S_0 N_0
        $$

        of the replicating portfolio.

        The replicating portfolio for an Asian option may only be computed relative to the pricing model in `dense` enumeration mode. The `from_enumeration` method of the pricing model must have been called before computing the replicating portfolio.

        Raises
        ------
        ValueError
            If the price trajectories of the underlying asset have not been enumerated before calling this method, or if the pricing model is in `sparse` enumeration mode.

        Returns
        -------
        bank_value, underlying_units, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
            A tuple containing the bank account process, the underlying units process, the total portfolio value process, and the risk-neutral price of the claim.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.finance import AsianOption, BinomialPricingModel
        >>> S_0 = 100
        >>> u = 1.1
        >>> p = 0.7
        >>> r = 0.01
        >>> T = Time.discrete(length=3)
        >>> S = BinomialPricingModel(
        ...     initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        ... )
        >>> S.from_enumeration(enum_mode="dense") # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'S':
        time          0           1           2           3
        trajectory
        0           100  110.000000  121.000000  133.100000
        1           100  110.000000  121.000000  110.000000
        2           100  110.000000  100.000000  110.000000
        3           100  110.000000  100.000000   90.909091
        4           100   90.909091  100.000000  110.000000
        5           100   90.909091  100.000000   90.909091
        6           100   90.909091   82.644628   90.909091
        7           100   90.909091   82.644628   75.131480
        >>> K = 100
        >>> call_option = AsianOption(pricing_model=S, strike=K, option_type="call")
        >>> B, N, V, price = call_option.replicating_portfolio()
        >>> print(B) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'bank_account_value':
        time                0          1          2
        trajectory
        0          -38.134572 -46.564062 -17.079208
        1          -38.134572 -46.564062 -17.079208
        2          -38.134572 -46.564062 -22.277228
        3          -38.134572 -46.564062 -22.277228
        4          -38.134572  -0.560775  -1.071536
        5          -38.134572  -0.560775  -1.071536
        6          -38.134572  -0.560775   0.000000
        7          -38.134572  -0.560775   0.000000
        >>> print(N) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'underlying_units':
        time              0         1         2
        trajectory
        0           0.42436  0.497525  0.250000
        1           0.42436  0.497525  0.250000
        2           0.42436  0.497525  0.250000
        3           0.42436  0.497525  0.250000
        4           0.42436  0.006853  0.011905
        5           0.42436  0.006853  0.011905
        6           0.42436  0.006853 -0.000000
        7           0.42436  0.006853 -0.000000
        >>> print(V) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'portfolio_value':
        time               0         1          2          3
        trajectory
        0           4.301408  8.163660  13.170792  16.025000
        1           4.301408  8.163660  13.170792  10.250000
        2           4.301408  8.163660   2.722772   5.000000
        3           4.301408  8.163660   2.722772   0.227273
        4           4.301408  0.062246   0.118940   0.227273
        5           4.301408  0.062246   0.118940  -0.000000
        6           4.301408  0.062246  -0.000000  -0.000000
        7           4.301408  0.062246  -0.000000  -0.000000
        >>> print(price)
        4.301408148315952
        """
        if self.pricing_model.data is None:
            raise ValueError(
                "Price trajectories of the underlying asset must be enumerated before computing the replicating portfolio for an Asian option."
            )
        if self.pricing_model.enum_mode == "sparse":
            raise ValueError(
                "Replicating portfolio for Asian options is not implemented for a pricing model in sparse enumeration mode."
            )

        S = self.pricing_model
        S_arr = S.data.values
        R = S.risk_free_gross_return
        u = S.up_factor
        d = S.down_factor
        q = (R - d) / (u - d)
        T = S.time[-1]

        S_dict = {t: S_arr[:: (2 ** (T - t)), t] for t in S.time}
        B_dict = dict.fromkeys(S.time[:-1])
        N_dict = dict.fromkeys(S.time[:-1])
        V_dict = dict.fromkeys(S.time)
        V_dict[T] = self.payout.data.values

        for t in reversed(range(T)):
            V_dict[t] = (V_dict[t + 1].reshape(-1, 2) @ np.array([q, 1 - q])) / R
            N_dict[t] = (
                np.diff(V_dict[t + 1].reshape(-1, 2)).squeeze()
                / np.diff(S_dict[t + 1].reshape(-1, 2)).squeeze()
            )
            B_dict[t] = V_dict[t] - S_dict[t] * N_dict[t]

        B_cols = [np.repeat(B_dict[t], repeats=2 ** (T - t)) for t in S.time[:-1]]
        N_cols = [np.repeat(N_dict[t], repeats=2 ** (T - t)) for t in S.time[:-1]]
        V_cols = [np.repeat(V_dict[t], repeats=2 ** (T - t)) for t in S.time]

        B_arr = np.column_stack(B_cols)
        N_arr = np.column_stack(N_cols)
        V_arr = np.column_stack(V_cols)

        B = (
            StochasticProcess(
                domain=S.domain,
                time=S.time[:-1],
                name="bank_account_value",
                is_discrete_state=True,
            )
            .from_numpy(B_arr)
            .with_probability_measure(probability_measure=S.probability_measure)
        )
        N = (
            StochasticProcess(
                domain=S.domain,
                time=S.time[:-1],
                name="underlying_units",
                is_discrete_state=True,
            )
            .from_numpy(N_arr)
            .with_probability_measure(probability_measure=S.probability_measure)
        )
        V = (
            StochasticProcess(
                domain=S.domain,
                time=S.time,
                name="portfolio_value",
                is_discrete_state=True,
            )
            .from_numpy(V_arr)
            .with_probability_measure(probability_measure=S.probability_measure)
        )

        return B, N, V, V[0].data.to_numpy()[0]
