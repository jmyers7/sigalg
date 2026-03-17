"""Binomial pricing model."""

from collections.abc import Hashable
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import binom

from sigalg.core.base.time import Time
from sigalg.core.probability_measures.probability_measure import ProbabilityMeasure
from sigalg.core.random_objects.random_variable import RandomVariable
from sigalg.processes.base.stochastic_process import StochasticProcess


class BinomialPricingModel(StochasticProcess):
    r"""Binomial pricing model for a risky asset.

    This class produces a binomial model for the price proccess $S_t$ of a risky asset, often referred to generically as a *stock*. Beginning from its initial price $S_0$, and given a time horizon $T$, this model supposes that the price process evolves according to the following dynamics:

    $$
    S_{t+1} = S_t Z_{t+1},
    $$

    for each $t=0,1,\ldots,T-1$, where $Z_t$ is a random variable that takes the value $u>1$ with some probability $p$ and the value $d = 1/u$ with probability $1-p$. The factors $u$ and $d$ are called the *up-factor* and *down-factor*, respectively.

    It follows that the process $S_t$ is a random walk on the set of prices

    $$
    \{S_0 u^m d^n : m,n \geq 0\}.
    $$

    At a fixed time horizon $T$, the final price $S_T$ takes its value in the subset

    $$
    \{S_0 u^{n} d^{T-n} : 0\leq n \leq T\}.
    $$

    For a fixed $n$, there are exactly $\binom{T}{n}$ many walks that terminate at the price $S_T = S_0 u^{n} d^{T-n}$. As a subclass of `StochasticProcess`, an instance of `BinomialPricingModel` carries a `from_enumeration` method. This method enumerates all price trajectories of the particular forms:

    $$
    \begin{gather*}
    S_0 \to S_0 u \to S_0u^2 \to S_0u^3 \to \ldots \to S_0u^{T-1} \to S_0 u^T \\
    S_0 \to S_0 d \to S_0du \to S_0du^2 \to \ldots \to S_0du^{T-2} \to S_0 du^{T-1} \\
    S_0 \to S_0 d \to S_0d^2 \to S_0d^2u \to \ldots \to S_0d^2u^{T-3} \to S_0 d^2u^{T-2} \\
    \cdots \quad \cdots \quad \cdots \quad \\
    S_0 \to S_0 d \to S_0d^2 \to S_0d^3 \to \ldots \to S_0d^{T-1} \to S_0 d^T
    \end{gather*}
    $$

    Thus, the method enumerates only $T+1$ of the total number $2^T$ of possible price trajectories. These special trajectories were selected because of their simple form and because they are in one-to-one correspondence with the set of possible the final prices $S_0 u^{n} d^{T-n}$.

    The risky asset is assumed to be traded in a market along with a non-risky asset with gross return $R = 1 + r$ at each time step, where $r$ is the *risk-free rate*. The non-risky asset is often conceptualized as a *bank account* with per-period interest rate $r$.

    The probability $p$ is the real-world probability that drives the price process of the stock. However, under the *no-arbitrage condition* $d < R< u$, a second probability $q$, called the *risk-neutral probability*, may be defined via the equation

    $$
    q = \frac{R - d}{u - d}.
    $$

    This risk-neutral probability is the key component in pricing various contingent claims using the binomial model. As a subclass of `StochasticProcess`, an instance of `BinomialPricingModel` carries a `probability_measure` attribute, which corresponds to the risk-neutral measure. Given an enumerated trajectory of the form above corresponding to the final price $S_0 u^n d^{T-N}$, this probability measure—denoted $Q$—gives

    $$
    Q (S_0 u^n d^{T-n}) = \binom{T}{n} q^n (1-q)^{T-n}.
    $$

    The appearance of the binomial coefficients gives this model its name.


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
    >>> from sigalg.finance import BinomialPricingModel, european_option
    >>> S_0 = 100
    >>> u = 1.1
    >>> r = 0.01
    >>> T = 3
    >>> time = Time.discrete(length=T)
    >>> S = BinomialPricingModel(
    ...     initial_price=S_0,
    ...     up_factor=u,
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

    # TODO: Add unit tests
    def __init__(
        self,
        initial_price: Real,
        up_factor: Real,
        risk_free_rate: Real,
        time: Time | None = None,
        name: Hashable | None = "S",
    ) -> None:
        if not isinstance(initial_price, Real) or initial_price <= 0:
            raise TypeError("initial_price must be a positive real number")
        if not isinstance(up_factor, Real) or up_factor <= 1:
            raise TypeError("up_factor must be a real number greater than 1")
        if not isinstance(risk_free_rate, Real) or risk_free_rate <= 0:
            raise TypeError("risk_free_rate must be a positive real number")

        self.initial_price = initial_price
        self.up_factor = up_factor
        self.down_factor = 1 / up_factor
        self.risk_free_rate = risk_free_rate
        self.risk_free_gross_return = 1 + risk_free_rate

        u = self.up_factor
        d = self.down_factor
        R = self.risk_free_gross_return

        if R <= d or R >= u:
            raise ValueError(
                "no-arbitrage condition violated: down_factor < risk_free_gross_return < up_factor"
            )

        super().__init__(
            time=time,
            is_discrete_time=True,
            is_discrete_state=True,
            name=name,
        )

        # Caches
        self._driving_process: StochasticProcess | None = None
        self._risk_neutral_prob: ProbabilityMeasure | None = None

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
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

        S_arr = self.initial_price * np.insert(
            arr=price_factors,
            obj=[0],
            values=np.ones(shape=(T + 1, 1)),
            axis=1,
        )

        return pd.DataFrame(S_arr)

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "Q"
    ) -> ProbabilityMeasure:
        u = self.up_factor
        d = self.down_factor
        R = self.risk_free_gross_return
        q = (R - d) / (u - d)
        T = self.time[-1]

        probs = dict(
            zip(self.domain, binom(n=T, p=1 - q).pmf(range(T + 1)), strict=False)
        )

        return ProbabilityMeasure(sample_space=self.domain, name=name).from_dict(probs)

    # --------------------- finance methods --------------------- #

    # TODO: Write unit tests
    def replicating_portfolio(
        self, claim: RandomVariable
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

        Recall that the price trajectories enumerated by the method `from_enumeration` are of the special forms

        $$
        \begin{gather*}
        S_0 \to S_0 u \to S_0u^2 \to S_0u^3 \to \ldots \to S_0u^{T-1} \to S_0 u^T \\
        S_0 \to S_0 d \to S_0du \to S_0du^2 \to \ldots \to S_0du^{T-2} \to S_0 du^{T-1} \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^2u \to \ldots \to S_0d^2u^{T-3} \to S_0 d^2u^{T-2} \\
        \cdots \quad \cdots \quad \cdots \quad \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^3 \to \ldots \to S_0d^{T-1} \to S_0 d^T
        \end{gather*}
        $$

        where $u$ and $d$ are the up- and down-factors of the model. The processes $B_t$, $N_t$, and $V_t$ generated by this method are computed along these same price trajectories.

        Parameters
        ----------
        claim : RandomVariable
            The claim to be replicated, which must be a random variable defined on the same domain as the price process and measurable with respect to the final price's sigma algebra.

        Raises
        ------
        TypeError
            If the claim is not an instance of RandomVariable or if its domain does not match the domain of the price process.
        ValueError
            If the claim is not measurable with respect to the final price's sigma algebra, or if the price trajectories have not been enumerated.

        Returns
        -------
        bank_value, underlying_units, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
            A tuple containing the bank account process, the underlying units process, the total portfolio value process, and the risk-neutral price of the claim.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.finance import BinomialPricingModel, european_option
        >>> S_0 = 100
        >>> u = 1.1
        >>> r = 0.01
        >>> T = 3
        >>> time = Time.discrete(length=T)
        >>> S = BinomialPricingModel(
        ...     initial_price=S_0,
        ...     up_factor=u,
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
        >>> call_option = european_option(price=S[T], strike=K)
        >>> print(call_option) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'european_call':
                    european_call
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
        if not isinstance(claim, RandomVariable) or claim.domain != self.domain:
            raise TypeError(
                "claim must be an instance of RandomVariable with the same domain as the model"
            )

        if not hasattr(self, "_is_enumerated"):
            raise ValueError(
                "Price trajectories must be enumerated before generating a replicating portfolio. Call from_enumeration."
            )

        u = self.up_factor
        d = self.down_factor
        R = self.risk_free_gross_return
        q = (R - d) / (u - d)
        T = self.time[-1]
        S = self.data.values

        if not claim.is_measurable(self[T].sigma_algebra):
            raise ValueError(
                "claim must be measurable with respect to the final price's sigma algebra"
            )

        V = dict.fromkeys(self.time)
        B = dict.fromkeys(self.time[:-1])
        N = dict.fromkeys(self.time[:-1])

        V_arr = np.zeros(shape=(T + 1, T + 1))
        N_arr = np.zeros(shape=(T + 1, T))
        B_arr = np.zeros(shape=(T + 1, T))

        V[T] = claim.data.values
        V_arr[:, T] = V[T]

        for t in reversed(range(T)):
            V[t] = (q * V[t + 1][:-1] + (1 - q) * V[t + 1][1:]) / R
            N[t] = (V[t + 1][:-1] - V[t + 1][1:]) / (u - d) / S[: (t + 1), t]
            B[t] = V[t] - S[: (t + 1), t] * N[t]

            V_arr[:, t] = np.concatenate((V[t], np.repeat(V[t][-1], T - t)))

            if t != T + 1:
                N_arr[:, t] = np.concatenate((N[t], np.repeat(N[t][-1], T - t)))
                B_arr[:, t] = np.concatenate((B[t], np.repeat(B[t][-1], T - t)))

        V = StochasticProcess(
            domain=self.domain, time=self.time, name="portfolio_value"
        ).from_numpy(V_arr)
        N = StochasticProcess(
            domain=self.domain, time=self.time[:-1], name="underlying_units"
        ).from_numpy(N_arr)
        B = StochasticProcess(
            domain=self.domain, time=self.time[:-1], name="bank_account_value"
        ).from_numpy(B_arr)

        return B, N, V, V[0].data.to_numpy()[0]
