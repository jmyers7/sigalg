"""Binomial pricing model."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import bernoulli, binom

from ....core.base.time import Time
from ....core.probability_measures.probability_measure import ProbabilityMeasure
from ....processes.base.stochastic_process import StochasticProcess
from ....processes.stopping_times.stopping_time import StoppingTime
from ....processes.types.iid_process import IIDProcess
from ..claims.claim import Claim
from .geometric_pricing_model import GeometricPricingModel


class BinomialPricingModel(GeometricPricingModel):
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

    As a subclass of `StochasticProcess`, an instance of `BinomialPricingModel` carries a `prob_measure` attribute, which corresponds to the real-world measure. The risk-neutral measure is accessible via the `emms` property.

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
    ... ).from_enumeration(enum_mode="sparse")
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
        risk_free_rate: Real,
        up_prob: Real,
        up_factor: Real,
        down_factor: Real | None = None,
        time: Time | None = None,
        name: Hashable | None = "S",
    ) -> None:
        if not isinstance(up_factor, Real) or up_factor <= 1:
            raise TypeError("up_factor must be a real number greater than 1")
        if down_factor is not None and (
            not isinstance(down_factor, Real) or down_factor >= 1
        ):
            raise TypeError("down_factor must be a real number less than 1")
        if not isinstance(up_prob, Real) or not (0 <= up_prob <= 1):
            raise TypeError("up_prob must be a real number in the interval [0,1]")

        self.up_prob = up_prob
        self._up_factor = up_factor
        if down_factor is None or np.abs(down_factor - 1 / up_factor) < 1e-5:
            self.is_recombining = True
            down_factor = 1 / up_factor
        else:
            self.is_recombining = False
        self._down_factor = down_factor

        super().__init__(
            initial_price=initial_price,
            risk_free_rate=risk_free_rate,
            time=time,
            name=name,
        )

        # caches
        self.enum_mode: str | None = None
        self._sparse_price_array: np.ndarray | None = None

    # --------------------- properties --------------------- #

    def _clear_generated_child_attributes(self) -> None:
        self._driving_process = None
        self._emms = None
        self._sparse_price_array = None

    @property
    def up_factor(self) -> Real:
        """Later."""
        return self._up_factor

    @property
    def down_factor(self) -> Real:
        """Later."""
        return self._down_factor

    @down_factor.setter
    def down_factor(self, value: Real) -> None:
        if not isinstance(value, Real) or value >= 1:
            raise TypeError("value must be a real number less than 1")

        if np.abs(value - 1 / self.up_factor) < 1e-5:
            self.is_recombining = True
            self._down_factor = 1 / self.up_factor
        else:
            self.is_recombining = False
            self._down_factor = value

        self._clear_generated_attributes()

    # --------------------- data generation methods --------------------- #

    def from_enumeration(
        self, length: int | None = None, enum_mode: str = "dense", **kwargs
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
        enum_mode : str, default="dense"
            The mode of enumeration, which must be either "sparse" or "dense". See above for details.

        Raises
        ------
        TypeError
            If `enum_mode` is not a string, is not one of "sparse" or "dense", or if a sparse tree is generated with out down_factor equal to 1 / up_factor.
        """
        if not isinstance(enum_mode, str) or enum_mode not in {"sparse", "dense"}:
            raise TypeError("enum_mode must be either 'sparse' or 'dense'")
        if enum_mode == "sparse" and not self.is_recombining:
            raise TypeError(
                "Cannot enumerate a sparse tree if down_factor does not equal 1 / up_factor"
            )
        self.enum_mode = enum_mode
        self._emms = None
        return super().from_enumeration(enum_mode=enum_mode, **kwargs)

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

    def _simulation_logic(
        self, n_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        trajectories = (
            self.driving_process.from_simulation(
                n_trajectories=n_trajectories, random_state=random_state
            ).cumprod()
            * self.initial_price
        )
        trajectories.insert_rv(
            time=self.time[0], state=self.initial_price, in_place=True
        )

        return trajectories.data

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

            self._driving_process = IIDProcess(
                distribution=bernoulli(1 - p),
                support=support,
                time=T,
                name="driving_process",
            )

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

    @property
    def risk_neutral_probs(self) -> tuple[Real, Real]:
        """Later."""
        R = self.risk_free_gross_return
        u = self.up_factor
        d = self.down_factor

        if R <= d or R >= u:
            raise ValueError(
                "no-arbitrage condition violated: down_factor < risk_free_gross_return < up_factor"
            )

        q_u = (R - d) / (u - d)
        q_d = 1 - q_u

        return q_u, q_d

    @property
    def emms(self) -> ProbabilityMeasure:
        """Return the equivalent martingale measures of the model."""
        if self._emms is None:
            if self.enum_mode == "sparse":
                self._emms = self._generate_probability_measure(
                    type="binomial", prob=self.risk_neutral_probs[0], name="Q"
                )

            elif self.enum_mode == "dense":
                self._emms = self._generate_probability_measure(
                    type="iid", prob=self.risk_neutral_probs[0], name="Q"
                )

            else:
                raise ValueError(
                    "Price trajectories must be enumerated before generating probability measures. Call from_enumeration."
                )

        return self._emms

    # --------------------- finance methods --------------------- #

    def price(self, claim: Claim, emm: ProbabilityMeasure | None = None) -> Real:
        """Price a claim under the model."""
        pass

    def replicating_portfolio(
        self, claim: Claim
    ) -> tuple[
        StochasticProcess, StochasticProcess, StochasticProcess, Real, StoppingTime
    ]:
        r"""Compute the replicating portfolio for a given contingent claim.

        The core idea of a *replicating portfolio* is this: Suppose that an individual sells a contingent claim on an underlying asset (generically called an *underlying*). The seller accepts a premium from the buyer for the claim at time $t=0$, and then at some specified maturity time $t=T$, the seller must pay the exercise value of the claim to the buyer. The claim is a *derivative*, in the sense that its value depends on (or derives from) the price of the underlying. The seller is thus interested in hedging their short position on the claim against an increase in the price of the underlying, which would increase the exercise value of the claim that the seller would owe the buyer.

        The underlying asset is assumed to be traded in a market that includes a bank account with risk-free, per-period interest rate $r$. The seller's hedging strategy is to trade in the underlying asset itself, as well as hold a cash position at the bank, so that when the contingent claim matures, the seller's portfolio will cover the exercise value owed to the buyer.

        The replicating portfolio thus consists of a pair $(B_t,\Delta_t)$ of processes, indexed $t=0,1,\ldots,T-1$, where $B_t$ represents the cash position at time $t$, and $\Delta_t$ counts the number of units of the underlying held in the portfolio at time $t$. A third process $V_t$ represents the total value of the portfolio, given by

        $$
        V_t = B_t + S_t \Delta_t,
        $$

        where $S_t$ is the price of the underlying at time $t$. A positive value of $B_t$ represents money held in the bank accruing interest for the seller at rate $r$, while a negative value represents a loan on which the seller pays interest at rate $r$. A positive value of $\Delta_t$ represents a *long position* on the underlying, while a negative value represents a *short position*.

        The replicating portfolio is *self-financing*, in the sense that

        $$
        V_t = (1+r) B_{t-1} + S_t \Delta_{t-1}
        $$

        for each $t=1,2,\ldots,T$. The right-hand side of this equation represents the evolution of the value of the portfolio over the time interval $[t-1,t]$, in which the amount $B_{t-1}$ in the bank accrues interest at rate $r$ and the price of the underlying changes from $S_{t-1}$ to $S_t$. This equation says that this evolved value of the old portfolio is equal to the value $V_t$ of the new portfolio at time $t$.

        The existence of the replicating portfolio also allows us to determine a fair, "risk-neutral" premium for the contingent claim paid by the buyer. Under the no-arbitrage assumption, this premium should coincide with the initial price

        $$
        V_0 = B_0 + S_0 \Delta_0
        $$

        of the replicating portfolio.

        Recall that `from_enumeration` method generates price trajectories in one of two modes: either `dense` mode or `sparse` mode. In `dense` mode, the method enumerates all $2^T$ price trajectories of the model. In `sparse` mode, the method enumerates only the following canonical $T+1$ price trajectories:

        $$
        \begin{gather*}
        S_0 \to S_0 u \to S_0u^2 \to S_0u^3 \to \ldots \to S_0u^{T-1} \to S_0 u^T \\
        S_0 \to S_0 d \to S_0du \to S_0du^2 \to \ldots \to S_0du^{T-2} \to S_0 du^{T-1} \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^2u \to \ldots \to S_0d^2u^{T-3} \to S_0 d^2u^{T-2} \\
        \cdots \quad \cdots \quad \cdots \quad \\
        S_0 \to S_0 d \to S_0d^2 \to S_0d^3 \to \ldots \to S_0d^{T-1} \to S_0 d^T
        \end{gather*}
        $$

        where $u$ and $d$ are the up- and down-factors of the model. If the price trajectories have been enumerated in `dense` mode, then the replicating portfolio is computed via backward induction through the full binomial tree of price trajectories. If the price trajectories have been enumerated in `sparse` mode, and if the price process of the claim is path-independent, then the replicating portfolio is computed via backward induction through the reduced binomial tree of $T+1$ price trajectories described above.

        Parameters
        ----------
        claim : Claim
            The contingent claim for which to compute the replicating portfolio. The payoff of the claim must be defined on the same domain as the price process of the underlying asset, and the price trajectories of the underlying asset must have been enumerated before calling this method.

        Raises
        ------
        TypeError
            If the claim is not an instance of `Claim` or if the claim payoff is not defined on the same domain as the price process of the underlying asset.
        ValueError
            If the price trajectories have not been enumerated.

        Returns
        -------
        bank_value, underlying_units, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
            A tuple containing the bank account process, the underlying units process, the total portfolio value process, and the risk-neutral price of the claim.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.finance import AsianOption, BinomialPricingModel, EuropeanOption
        >>> S_0 = 100
        >>> u = 1.1
        >>> p = 0.7
        >>> r = 0.01
        >>> T = 3
        >>> time = Time.discrete(length=T)
        >>> S = BinomialPricingModel(
        ...     initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=time
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
        >>> asian_call = AsianOption(pricing_model=S, strike=K, option_type="call")
        >>> B, Delta, V, price, tau = S.replicating_portfolio(claim=asian_call)
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
        >>> print(Delta) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'underlying_units':
        time              0         1         2
        trajectory
        0           0.42436  0.497525   0.250000
        1           0.42436  0.497525   0.250000
        2           0.42436  0.497525   0.250000
        3           0.42436  0.497525   0.250000
        4           0.42436  0.006853   0.011905
        5           0.42436  0.006853   0.011905
        6           0.42436  0.006853  -0.000000
        7           0.42436  0.006853  -0.000000
        >>> print(V) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'portfolio_value':
        time               0         1          2          3
        trajectory
        0           4.301408  8.163660  13.170792  16.025000
        1           4.301408  8.163660  13.170792  10.250000
        2           4.301408  8.163660   2.722772   5.000000
        3           4.301408  8.163660   2.722772   0.227273
        4           4.301408  0.062246   0.118940   0.227273
        5           4.301408  0.062246   0.118940   0.000000
        6           4.301408  0.062246   0.000000   0.000000
        7           4.301408  0.062246   0.000000   0.000000
        >>> print(price)
        4.301408148315952
        >>> S.from_enumeration(enum_mode="sparse") # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'S':
        time            0           1           2           3
        trajectory
        0           100.0  110.000000  121.000000  133.100000
        1           100.0   90.909091  100.000000  110.000000
        2           100.0   90.909091   82.644628   90.909091
        3           100.0   90.909091   82.644628   75.131480
        >>> K = 100
        >>> euro_call = EuropeanOption(pricing_model=S, strike=K, option_type="call")
        >>> B, Delta, V, price, tau = S.replicating_portfolio(claim=euro_call)
        >>> print(B) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'bank_account_value':
        time                0          1          2
        trajectory
        0          -50.150931 -73.822294 -99.009901
        1          -50.150931 -24.674118 -47.147572
        2          -50.150931 -24.674118   0.000000
        3          -50.150931 -24.674118   0.000000
        >>> print(Delta) # doctest: +NORMALIZE_WHITESPACE
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
        2           8.579463   2.738827   0.000000   0.0
        3           8.579463   2.738827   0.000000   0.0
        >>> print(price)
        8.57946313365138
        """
        T = self.time[-1]

        B_arr, Delta_arr, V_arr, tau_arr = self._initialize_replicating_arrays()
        V_arr[:, -1], tau_arr[:, -1] = claim._backward_induction_base_case()

        for t in reversed(range(T)):
            V_forward, S_forward, S_curr = self._extract_tree_nodes(t=t, V_arr=V_arr)

            B_curr, Delta_curr, V_curr, tau_curr = claim._backward_induction(
                enum_mode=self.enum_mode,
                V_forward=V_forward,
                S_forward=S_forward,
                S_curr=S_curr,
                strike=claim.strike,
                risk_free_rate=self.risk_free_rate,
                risk_neutral_prob=self.risk_neutral_probs[0],
            )

            B_arr[:, t], Delta_arr[:, t], V_arr[:, t], tau_arr[:, t] = (
                self._broadcast_node_values(
                    t=t,
                    B_curr=B_curr,
                    Delta_curr=Delta_curr,
                    V_curr=V_curr,
                    tau_curr=tau_curr,
                )
            )

        tau_arr = np.where(
            tau_arr.max(axis=1) == 0,
            np.inf,
            np.argmax(tau_arr, axis=1),
        )

        B, Delta, V, price, tau = self._convert_replicating_arrays_to_processes(
            B_arr=B_arr, Delta_arr=Delta_arr, V_arr=V_arr, tau_arr=tau_arr
        )

        return B, Delta, V, price, tau

    def _initialize_replicating_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]

        if self.enum_mode == "dense":
            B_arr = np.zeros(shape=(2**T, T))
            Delta_arr = np.zeros(shape=(2**T, T))
            V_arr = np.zeros(shape=(2**T, T + 1))
            tau_arr = np.zeros(shape=(2**T, T + 1))

        elif self.enum_mode == "sparse":
            B_arr = np.zeros(shape=(T + 1, T))
            Delta_arr = np.zeros(shape=(T + 1, T))
            V_arr = np.zeros(shape=(T + 1, T + 1))
            tau_arr = np.zeros(shape=(T + 1, T + 1))

        return B_arr, Delta_arr, V_arr, tau_arr

    def _extract_tree_nodes(
        self, t: int, V_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]

        if self.enum_mode == "dense":
            V_forward = V_arr[:: (2 ** (T - t - 1)), t + 1]  # shape (2^(t+1),)
            S_forward = self.data.values[  # shape (2^(t+1),)
                :: (2 ** (T - t - 1)), t + 1
            ]
            S_curr = self.data.values[:: (2 ** (T - t)), t]  # shape (2^t,)

        elif self.enum_mode == "sparse":
            V_forward = V_arr[: t + 2, t + 1]  # shape (t+2,)
            S_forward = self.data.values[: t + 2, t + 1]  # shape(t+2,)
            S_curr = self.data.values[: t + 1, t]  # shape (t+1,)

        return V_forward, S_forward, S_curr

    def _broadcast_node_values(
        self,
        t: int,
        B_curr: np.ndarray,
        Delta_curr: np.ndarray,
        V_curr: np.ndarray,
        tau_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]

        if self.enum_mode == "dense":
            # all have shape (2^T,)
            B = np.repeat(B_curr, repeats=2 ** (T - t))
            Delta = np.repeat(Delta_curr, repeats=2 ** (T - t))
            V = np.repeat(V_curr, repeats=2 ** (T - t))
            tau = np.repeat(tau_curr, repeats=2 ** (T - t))

        elif self.enum_mode == "sparse":
            # all have shape (T+1,)
            B = np.concatenate((B_curr, np.repeat(B_curr[-1], T - t)))
            Delta = np.concatenate((Delta_curr, np.repeat(Delta_curr[-1], T - t)))
            V = np.concatenate((V_curr, np.repeat(V_curr[-1], T - t)))
            tau = np.concatenate((tau_curr, np.repeat(tau_curr[-1], T - t)))

        return B, Delta, V, tau

    def _convert_replicating_arrays_to_processes(
        self,
        B_arr: np.ndarray,
        Delta_arr: np.ndarray,
        V_arr: np.ndarray,
        tau_arr: np.ndarray,
    ) -> tuple[
        StochasticProcess, StochasticProcess, StochasticProcess, Real, StoppingTime
    ]:
        B = (
            StochasticProcess(
                domain=self.domain,
                time=self.time[:-1],
                name="bank_account_value",
                is_discrete_state=True,
            )
            .from_numpy(B_arr)
            .with_probability_measure(prob_measure=self.prob_measure)
        )

        Delta = (
            StochasticProcess(
                domain=self.domain,
                time=self.time[:-1],
                name="underlying_units",
                is_discrete_state=True,
            )
            .from_numpy(Delta_arr)
            .with_probability_measure(prob_measure=self.prob_measure)
        )

        V = (
            StochasticProcess(
                domain=self.domain,
                time=self.time,
                name="portfolio_value",
                is_discrete_state=True,
            )
            .from_numpy(V_arr)
            .with_probability_measure(prob_measure=self.prob_measure)
        )

        price = V[0].data.to_numpy()[0]

        tau = StoppingTime(
            filtration=self.natural_filtration, name="stopping_time"
        ).from_numpy(array=tau_arr)

        return B, Delta, V, price, tau

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        return f"Price process '{self.name}'"
