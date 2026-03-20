"""Binomial pricing model."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import bernoulli, binom

from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ...processes.base.stochastic_process import StochasticProcess
from ...processes.types.iid_process import IIDProcess
from .pricing_model import PricingModel

if TYPE_CHECKING:
    from ..claims.european_option import Claim


class BinomialPricingModel(PricingModel):
    r"""Binomial pricing model for a risky asset.

    This class produces a binomial model for the price proccess $S_t$ of a risky asset, often referred to generically as a *stock*. Beginning from its initial price $S_0$, and given a time horizon $T$, this model supposes that the price process evolves according to the following dynamics:

    $$
    S_{t+1} = S_t Z_{t+1},
    $$

    for each $t=0,1,\ldots,T-1$, where $Z_t$ is a random variable that takes the value $u>1$ with some probability $p$ and the value $d = 1/u$ with probability $1-p$. The factors $u$ and $d$ are called the *up-factor* and *down-factor*, respectively.

    The risky asset is assumed to be traded in a market along with a non-risky asset with gross return $R = 1 + r$ at each time step, where $r$ is the *risk-free rate*. The non-risky asset is often conceptualized as a *bank account* with per-period interest rate $r$.

    The probability $p$ is the real-world probability that drives the price process of the stock. However, under the *no-arbitrage condition* $d < R< u$, a second probability $q$, called the *risk-neutral probability*, may be defined via the equation

    $$
    q = \frac{R - d}{u - d}.
    $$

    This risk-neutral probability is the key component in pricing various contingent claims using the binomial model.

    As a subclass of `StochasticProcess`, an instance of `BinomialPricingModel` carries a `probability_measure` attribute, which corresponds to the real-world measure. The risk-neutral measure is accessible via the `risk_neutral_measure` property.

    Parameters
    ----------
    initial_price : Real
        The initial price of the risky asset.
    up_factor : Real
        The up-factor of the model, which must be greater than 1.
    risk_free_rate : Real
        The risk-free rate of the non-risky asset, which must be positive.
    time : Time | None, default=None
        The time index for the pricing model.
    name: Hashable | None, default="S"
        The name of the stochastic process.

    Raises
    ------
    TypeError
        If any of the parameters are of the wrong type or do not satisfy the required conditions.
    ValueError
        If the no-arbitrage condition is violated.

    Examples
    --------
    >>> from sigalg.core import Time
    >>> from sigalg.finance import BinomialPricingModel
    >>> S_0 = 100
    >>> u = 1.1
    >>> p = 0.7
    >>> r = 0.01
    >>> T = 3
    >>> time = Time.discrete(length=T)
    >>> S = BinomialPricingModel(
    ...     initial_price=S_0,
    ...     up_factor=u,
    ...     up_prob=p,
    ...     risk_free_rate=r,
    ...     time=time,
    ... ).from_enumeration()
    >>> S # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'S':
    time            0           1           2           3
    trajectory
    0           100.0  110.000000  121.000000  133.100000
    1           100.0   90.909091  100.000000  110.000000
    2           100.0   90.909091   82.644628   90.909091
    3           100.0   90.909091   82.644628   75.131480
    """

    def __init__(
        self,
        initial_price: Real,
        up_factor: Real,
        up_prob: Real,
        risk_free_rate: Real,
        time: Time | None = None,
        name: Hashable | None = "S",
    ) -> None:
        if not isinstance(initial_price, Real) or initial_price <= 0:
            raise TypeError("initial_price must be a positive real number")
        if not isinstance(up_factor, Real) or up_factor <= 1:
            raise TypeError("up_factor must be a real number greater than 1")
        if not isinstance(up_prob, Real) or not (0 <= up_prob <= 1):
            raise TypeError("up_prob must be a real number in the interval [0,1]")
        if not isinstance(risk_free_rate, Real) or risk_free_rate <= 0:
            raise TypeError("risk_free_rate must be a positive real number")

        self.initial_price = initial_price
        self.up_factor = up_factor
        self.down_factor = 1 / up_factor
        self.up_prob = up_prob
        self.risk_free_rate = risk_free_rate
        self.risk_free_gross_return = 1 + risk_free_rate

        u = self.up_factor
        d = self.down_factor
        R = self.risk_free_gross_return

        if R <= d or R >= u:
            raise ValueError(
                "no-arbitrage condition violated: down_factor < risk_free_gross_return < up_factor"
            )

        self.risk_neutral_prob = (R - d) / (u - d)

        super().__init__(
            time=time,
            is_discrete_time=True,
            is_discrete_state=True,
            name=name,
        )

        self._driving_process: StochasticProcess | None = None
        self._risk_neutral_measure: ProbabilityMeasure | None = None
        self._sparse_price_array: np.ndarray | None = None
        self.enum_mode: str | None = None

    # --------------------- data generation methods --------------------- #

    def from_enumeration(
        self, length: int | None = None, enum_mode: str = "sparse", **kwargs
    ) -> StochasticProcess:
        r"""Generate price trajectories of the binomial pricing model via enumeration.

        Suppose that $S_t$ is the price process of the underlying asset, and that $u$ and $d$ are the up- and down-factors of the model, respectively. Then $S_t$ is a random walk on the set of prices

        $$
        \{S_0 u^m d^n : m,n \geq 0\}.
        $$

        At a fixed time horizon $T$, the final price $S_T$ takes one of the $T+1$ values in the set

        $$
        \{S_0 u^{n} d^{T-n} : 0\leq n \leq T\}.
        $$

        There are exactly $\binom{T}{n}$ many random walks (i.e., price trajectories) that terminate at the final price $S_T = S_0 u^{n} d^{T-n}$, and thus a total of $2^T = \sum_{n=0}^T \binom{T}{n}$ many random walks that end at *some* final price.

        This method enumerates these price trajectories in one of two modes: either `dense` mode or `sparse` mode. In `dense` mode, the method enumerates all $2^T$ price trajectories of the model. In `sparse` mode, the method enumerates only $T+1$ price trajectories of the model, which are of the special forms:

        $$
        \begin{gather*}
        S_0 \to S_0 u \to S_0u^2 \to S_0u^3 \to \ldots \to S_0u^{T-1} \to S_0 u^T \\
        S_0 \to S_0 d \to S_0du \to S_0du^2 \to \ldots \to S_0du^{T-2} \to S_0 du^{T-1} \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^2u \to \ldots \to S_0d^2u^{T-3} \to S_0 d^2u^{T-2} \\
        \cdots \quad \cdots \quad \cdots \quad \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^3 \to \ldots \to S_0d^{T-1} \to S_0 d^T
        \end{gather*}
        $$

        The `dense` mode of enumeration should only be used for small values of $T$, as the number of price trajectories grows exponentially in $T$.

        Parameters
        ----------
        length : int | None, default=None
            The length of the enumeration, which must be a positive integer. If `None`, the length of the enumeration is taken to be the length of the time index of the model.
        enum_mode : str, default="sparse"
            The mode of enumeration, which must be either "sparse" or "dense". See above for details.

        Raises
        ------
        TypeError
            If `enum_mode` is not a string or is not one of "sparse" or "dense".
        """
        if not isinstance(enum_mode, str) or enum_mode not in {"sparse", "dense"}:
            raise TypeError("enum_mode must be either 'sparse' or 'dense'")
        self.enum_mode = enum_mode
        self._risk_neutral_measure = None
        return super().from_enumeration(length=length, enum_mode=enum_mode, **kwargs)

    def _enumeration_logic(self, enum_mode: str) -> pd.DataFrame:
        self._generate_sparse_price_array()

        if enum_mode == "sparse":
            return pd.DataFrame(self._sparse_price_array)

        elif enum_mode == "dense":
            S = self.initial_price * self.driving_process.from_enumeration().cumprod()
            S.insert_rv(state=self.initial_price, time=0, in_place=True)
            return S.data

        else:
            raise ValueError("enum_mode must be either 'sparse' or 'dense'")

    def _generate_sparse_price_array(self) -> np.ndarray:
        u = self.up_factor
        d = self.down_factor
        T = self.time[-1]

        total_powers = np.tile(np.array(range(1, T + 1)), reps=[T + 1, 1])
        self._u_powers = np.maximum(
            0, total_powers - np.array(range(T + 1)).reshape(-1, 1)
        )
        self._d_powers = total_powers - self._u_powers

        price_factors = (np.ones(shape=(T + 1, T)) * u**self._u_powers) * (
            np.ones(shape=(T + 1, T)) * d**self._d_powers
        )

        self._sparse_price_array = self.initial_price * np.insert(
            arr=price_factors,
            obj=[0],
            values=np.ones(shape=(T + 1, 1)),
            axis=1,
        )

    @property
    def driving_process(self) -> StochasticProcess:
        r"""Return the driving process of the binomial pricing model.

        The driving process is an IID process $Z_t$ representing the up and down movements of the underlying asset in the binomial model. It takes the value $u$ with probability $p$ and the value $d$ with probability $1-p$, where $u$ is the up-factor, $d$ is the down-factor, and $p$ is the real-world probability of an up move. The driving process is defined for times $t=1,2,\ldots,T$, where $T$ is the final time of the model.

        This property should only be used for small values of $T$, as the number of price trajectories is equal to $2^T$.

        Returns
        -------
        driving_process : StochasticProcess
            The driving process of the binomial pricing model.
        """
        if self._driving_process is None:
            T = self.time[1:]
            u = self.up_factor
            p = self.up_prob
            d = self.down_factor
            support = {0: u, 1: d}

            Z = IIDProcess(
                distribution=bernoulli(1 - p),
                support=support,
                time=T,
                name="driving_process",
            )

            self._driving_process = Z

        return self._driving_process

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        if self.enum_mode == "sparse":
            return self._generate_probability_measure(
                type="binomial", prob=self.up_prob, name=name
            )

        elif self.enum_mode == "dense":
            return self._generate_probability_measure(
                type="iid", prob=self.up_prob, name=name
            )

        else:
            raise ValueError(
                "Price trajectories must be enumerated before generating probability measures. Call from_enumeration."
            )

    @property
    def risk_neutral_measure(self) -> ProbabilityMeasure:
        """Later."""
        if self._risk_neutral_measure is None:
            if self.enum_mode == "sparse":
                self._risk_neutral_measure = self._generate_probability_measure(
                    type="binomial", prob=self.risk_neutral_prob, name="Q"
                )

            elif self.enum_mode == "dense":
                self._risk_neutral_measure = self._generate_probability_measure(
                    type="iid", prob=self.risk_neutral_prob, name="Q"
                )

            else:
                raise ValueError(
                    "Price trajectories must be enumerated before generating probability measures. Call from_enumeration."
                )

        return self._risk_neutral_measure

    def _generate_probability_measure(
        self, type: str, prob: Real, name: Hashable | None
    ) -> ProbabilityMeasure:
        T = self.time[-1]

        if type == "binomial":
            probs = dict(
                zip(self.domain, binom(n=T, p=1 - prob).pmf(range(T + 1)), strict=False)
            )

            return ProbabilityMeasure(sample_space=self.domain, name=name).from_dict(
                probs
            )

        elif type == "iid":
            values = list(product([0, 1], repeat=T))

            distribution = bernoulli(1 - prob)
            element_wise_probabilities = distribution.pmf(values)
            probabilities = pd.Series(
                data=np.prod(element_wise_probabilities, axis=1),
                index=self.domain.data,
            )
            probabilities /= probabilities.sum()

            return ProbabilityMeasure(sample_space=self.domain, name=name).from_pandas(
                probabilities
            )

        else:
            raise ValueError("type must be either 'iid' or 'binomial'")

    # --------------------- finance methods --------------------- #

    def replicating_portfolio(
        self, claim: Claim
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        r"""Compute the replicating portfolio for the binomial pricing model given a contingent claim.

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

        Recall that `from_enumeration` method generates price trajectories of $S_t$ in one of two modes: either `dense` mode or `sparse` mode. In `dense` mode, the method enumerates all $2^T$ price trajectories of the model. In `sparse` mode, the method enumerates only $T+1$ price trajectories of the model, which are of the special forms:

        $$
        \begin{gather*}
        S_0 \to S_0 u \to S_0u^2 \to S_0u^3 \to \ldots \to S_0u^{T-1} \to S_0 u^T \\
        S_0 \to S_0 d \to S_0du \to S_0du^2 \to \ldots \to S_0du^{T-2} \to S_0 du^{T-1} \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^2u \to \ldots \to S_0d^2u^{T-3} \to S_0 d^2u^{T-2} \\
        \cdots \quad \cdots \quad \cdots \quad \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^3 \to \ldots \to S_0d^{T-1} \to S_0 d^T
        \end{gather*}
        $$

        where $u$ and $d$ are the up- and down-factors of the model. In the latter case, in which the trajectories were generated in `sparse` mode, then this method computes the replicating portfolio along these same $T+1$ price trajectories.

        Parameters
        ----------
        claim : Claim
            The contingent claim for which to compute the replicating portfolio. The payout of the claim must be defined on the same domain as the price process of the underlying asset, and the price trajectories of the underlying asset must have been enumerated before calling this method.

        Raises
        ------
        TypeError
            If the claim is not an instance of `Claim` or if the claim payout is not defined on the same domain as the price process of the underlying asset.
        ValueError
            If the price trajectories have not been enumerated.

        Returns
        -------
        bank_value, underlying_units, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
            A tuple containing the bank account process, the underlying units process, the total portfolio value process, and the risk-neutral price of the claim.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.finance import BinomialPricingModel, EuropeanOption
        >>> S_0 = 100
        >>> u = 1.1
        >>> p = 0.7
        >>> r = 0.01
        >>> T = 3
        >>> time = Time.discrete(length=T)
        >>> S = BinomialPricingModel(
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     up_prob=p,
        ...     risk_free_rate=r,
        ...     time=time,
        ... ).from_enumeration()
        >>> print(S) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'S':
        time            0           1           2           3
        trajectory
        0           100.0  110.000000  121.000000  133.100000
        1           100.0   90.909091  100.000000  110.000000
        2           100.0   90.909091   82.644628   90.909091
        3           100.0   90.909091   82.644628   75.131480
        >>> K = 100
        >>> call_option = EuropeanOption(pricing_model=S, strike=K, option_type="call")
        >>> print(call_option.payout) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'EuropeanCallPayout':
               EuropeanCallPayout
        trajectory
        0                    33.1
        1                    10.0
        2                    -0.0
        3                    -0.0
        >>> B, N, V, price = S.replicating_portfolio(claim=call_option)
        >>> print(B) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'bank_account_value':
        time                0          1          2
        trajectory
        0          -50.150931 -73.822294 -99.009901
        1          -50.150931 -24.674118 -47.147572
        2          -50.150931 -24.674118  -0.000000
        3          -50.150931 -24.674118  -0.000000
        >>> print(N) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'underlying_units':
        time               0         1        2
        trajectory
        0           0.587304  0.797939  1.00000
        1           0.587304  0.301542  0.52381
        2           0.587304  0.301542  0.00000
        3           0.587304  0.301542  0.00000
        >>> print(V) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'portfolio_value':
        time               0          1          2     3
        trajectory
        0           8.579463  13.950993  21.990099  33.1
        1           8.579463   2.738827   5.233380  10.0
        2           8.579463   2.738827  -0.000000  -0.0
        3           8.579463   2.738827  -0.000000  -0.0
        """
        from ..claims.european_option import Claim

        if not isinstance(claim, Claim):
            raise TypeError("claim must be an instance of Claim")
        if claim.payout.domain != self.domain:
            raise TypeError(
                "The claim payout must be defined on the same domain as the price process"
            )
        if self.enum_mode is None:
            raise ValueError(
                "Price trajectories must be enumerated before generating a replicating portfolio. Call from_enumeration."
            )

        return claim.replicating_portfolio()

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        return f"Price process '{self.name}'"
