"""Binomial pricing model."""

from collections.abc import Hashable
from numbers import Real

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.stats import bernoulli

from sigalg.core.base.time import Time
from sigalg.core.probability_measures.probability_measure import ProbabilityMeasure
from sigalg.core.random_objects.random_variable import RandomVariable
from sigalg.processes.base.stochastic_process import StochasticProcess
from sigalg.processes.types.iid_process import IIDProcess


class BinomialPricingModel(StochasticProcess):
    r"""Binomial pricing model for a risky asset.

    This class produces a binomial model for the price proccess $S_t$ of a risky asset, often referred to generically as a *stock*. Beginning from its initial price $S_0$, and given a time horizon $T$, the price process evolves according to the following dynamics:

    $$
    S_{t+1} = S_t Z_{t+1},
    $$

    for each $t=0,1,\ldots,T-1$, where $Z_t$ is a random variable that takes the value $u>1$ with some probability $q$ and the value $d = 1/u$ with probability $1-q$. The process $Z_t$ is called the *driving process* of the model, the probability $q$ is called the *risk-neutral probability*, and the factors $u$ and $d$ are called the *up-factor* and *down-factor*, respectively.

    The risky asset is assumed to be traded in a market with a non-risky asset that returns $R = 1 + r$ at each time step, where $r$ is the *risk-free rate*. The non-risky asset is often conceptualized as a *bank account* with per-period interest rate $r$.

    The risk-neutral probability $q$ is determined by the up-factor $u$ and the risk-free gross return $R$ as follows:

    $$
    q = \frac{R - d}{u - d}.
    $$

    That $q$ is a valid probability is a consequence of the *no-arbitrage condition* $d < R < u$.

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
    >>> S_0 = 100 # initial stock price
    >>> u = 1.1 # up factor
    >>> r = 0.01 # risk-free rate
    >>> T = Time.discrete(length=3)
    >>> S = BinomialPricingModel(initial_price=S_0, up_factor=u, risk_free_rate=r, time=T).from_enumeration()
    >>> print(S) # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'S':
    time          0           1           2           3
    trajectory
    0           100   90.909091   82.644628   75.131480
    1           100   90.909091   82.644628   90.909091
    2           100   90.909091  100.000000   90.909091
    3           100   90.909091  100.000000  110.000000
    4           100  110.000000  100.000000   90.909091
    5           100  110.000000  100.000000  110.000000
    6           100  110.000000  121.000000  110.000000
    7           100  110.000000  121.000000  133.100000
    """

    # TODO: Add unit tests for input validation
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
        self.risk_free_return = 1 + risk_free_rate

        if (
            self.down_factor >= self.risk_free_return
            or self.up_factor <= self.risk_free_return
        ):
            raise ValueError(
                "no-arbitrage condition violated: down_factor < risk_free_return < up_factor"
            )

        self.risk_neutral_probability = (self.risk_free_return - self.down_factor) / (
            self.up_factor - self.down_factor
        )

        super().__init__(
            time=time,
            is_discrete_time=True,
            is_discrete_state=True,
            name=name,
        )

        # Cachess
        self._driving_process: StochasticProcess | None = None

    # --------------------- properties --------------------- #

    @property
    def driving_process(self) -> StochasticProcess:
        """Later."""
        if self._driving_process is None:
            T = self.time[1:]
            u = self.up_factor
            d = self.down_factor
            q = self.risk_neutral_probability
            support = {0: d, 1: u}

            Z = IIDProcess(
                distribution=bernoulli(q),
                support=support,
                time=T,
                name="driving_process",
            )
            self._driving_process = Z

        return self._driving_process

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
        S = self.initial_price * self.driving_process.from_enumeration().cumprod()
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    def _simulation_logic(
        self, n_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        S = (
            self.initial_price
            * self.driving_process.from_simulation(
                n_trajectories=n_trajectories, random_state=random_state
            ).cumprod()
        )
        S.insert_rv(state=self.initial_price, time=0, in_place=True)
        return S.data

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "Q"
    ) -> ProbabilityMeasure:
        return self.driving_process.probability_measure.with_name(name)

    # --------------------- finance methods --------------------- #

    # TODO: Write unit tests
    def replicating_portfolio(
        self, claim: RandomVariable
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        r"""Compute the replicating portfolio for the binomial pricing model given a contingent claim.

        This method computes three proccesses $B_t$, $N_t$, and $V_t$ for $t=0,1,\ldots,T$. For each $t\geq 1$, the process $B_t$ represents the amount of money in the portfolio invested in the non-risky asset over the time interval $[t-1,t]$ (with risk-free return $R$), while the process $N_t$ represents the number of units of the risky asset held in the portfolio over the same interval. We have $B_0 = B_1$ and $N_0 = N_1$, by convention. The processes $B_t$ and $N_t$ are constructed so that the portfolio is *self-financing*, in the sense that

        $$
        B_tR + N_t S_t = B_{t+1} + N_{t+1} S_t,
        $$

        for all $t=0,1,\ldots,T-1$, where $S_t$ is the price of the risky asset at time $t$. The process $V_t$ represents the total value of the portfolio at time $t$, and so for each $t\geq 1$ we have

        $$
        V_t = B_tR + N_t S_t.
        $$

        At time $t=0$, the value of the portfolio is given by the above self-financing condition, which reads

        $$
        V_0 = B_1 + N_1 S_0.
        $$

        Finally, the processes $B_t$, $N_t$, and $V_t$ are constructed so that the portfolio replicates a given contingent claim $\Phi(S_T)$, in the sense that $\Phi(S_T) = V_T$, where $T$ is the final time of the model. The *risk-neutral price* of the claim is then given by $V_0$.

        Parameters
        ----------
        claim : RandomVariable
            The claim to be replicated, which must be a random variable defined on the model's sample space and measurable with respect to the final price's sigma algebra.

        Raises
        ------
        TypeError
            If the claim is not an instance of RandomVariable or if its domain does not match the model's sample space.
        ValueError
            If the claim is not measurable with respect to the final price's sigma algebra.

        Returns
        -------
        non_risky, risky, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
            The non-risky asset process, risky asset process, portfolio value process, and risk-neutral price that replicate the claim.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, european_option
        >>> s = 100 # initial stock price
        >>> u = 1.1 # up factor
        >>> r = 0.01 # risk-free rate
        >>> model = BinomialPricingModel(initial_price=s, up_factor=u, risk_free_rate=r, length=3)
        >>> S = model.price_process
        >>> print(S) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'price_process':
        time          0           1           2           3
        trajectory
        0           100   90.909091   82.644628   75.131480
        1           100   90.909091   82.644628   90.909091
        2           100   90.909091  100.000000   90.909091
        3           100   90.909091  100.000000  110.000000
        4           100  110.000000  100.000000   90.909091
        5           100  110.000000  100.000000  110.000000
        6           100  110.000000  121.000000  110.000000
        7           100  110.000000  121.000000  133.100000
        >>> call_option = european_option(price=S[3], strike=100)
        >>> B, N, V, price = model.replicating_portfolio(claim=call_option)
        >>> # print the non-risky "bond" value process
        >>> print(B) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'non_risky':
        time                0          1          2          3
        trajectory
        0          -50.150931 -50.150931 -24.674118   0.000000
        1          -50.150931 -50.150931 -24.674118   0.000000
        2          -50.150931 -50.150931 -24.674118 -47.147572
        3          -50.150931 -50.150931 -24.674118 -47.147572
        4          -50.150931 -50.150931 -73.822294 -47.147572
        5          -50.150931 -50.150931 -73.822294 -47.147572
        6          -50.150931 -50.150931 -73.822294 -99.009901
        7          -50.150931 -50.150931 -73.822294 -99.009901
        >>> # print the risky "stock" process giving the number of units of the stock held in the replicating portfolio
        >>> print(N) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'risky':
        time               0         1         2        3
        trajectory
        0           0.587304  0.587304  0.301542  0.00000
        1           0.587304  0.587304  0.301542  0.00000
        2           0.587304  0.587304  0.301542  0.52381
        3           0.587304  0.587304  0.301542  0.52381
        4           0.587304  0.587304  0.797939  0.52381
        5           0.587304  0.587304  0.797939  0.52381
        6           0.587304  0.587304  0.797939  1.00000
        7           0.587304  0.587304  0.797939  1.00000
        >>> # print the total value of the replicating portfolio
        >>> print(V) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'portfolio_value':
        time               0          1          2     3
        trajectory
        0           8.579463   2.738827   0.000000  -0.0
        1           8.579463   2.738827   0.000000  -0.0
        2           8.579463   2.738827   5.233380  -0.0
        3           8.579463   2.738827   5.233380  10.0
        4           8.579463  13.950993   5.233380  -0.0
        5           8.579463  13.950993   5.233380  10.0
        6           8.579463  13.950993  21.990099  10.0
        7           8.579463  13.950993  21.990099  33.1
        >>> # check that V[3] equals the claim
        >>> print(call_option) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'european_call':
            european_call
        trajectory
        0                    -0.0
        1                    -0.0
        2                    -0.0
        3                    10.0
        4                    -0.0
        5                    10.0
        6                    10.0
        7                    33.1
        >>> # check the risk-neutral price of the claim
        >>> print(price)
        8.579463133651387
        """
        if not isinstance(claim, RandomVariable) or claim.domain != self.sample_space:
            raise TypeError(
                "claim must be an instance of RandomVariable with the same domain as the model's sample space"
            )

        t_final = self.time[-1]
        S = self.price_process

        if not claim.is_measurable(S[t_final].sigma_algebra):
            raise ValueError(
                "claim must be measurable with respect to the final price's sigma algebra"
            )

        B = dict.fromkeys(self.time)
        N = dict.fromkeys(self.time)
        V = dict.fromkeys(self.time)
        V[t_final] = claim
        R = self.risk_free_return

        for t in reversed(self.time[1:]):
            num_blocks = len(self.sample_space) // 2 ** (t_final - t + 1)
            num_repeats = 2 ** (t_final - t + 1)

            first_row_block_idx = (
                np.array([2 * i for i in range(num_blocks)]) * num_repeats // 2
            )
            second_row_block_idx = (
                np.array([2 * i + 1 for i in range(num_blocks)]) * num_repeats // 2
            )

            blocks = [
                np.array(
                    [
                        [R, S[t](first_row_block_idx[i])],
                        [R, S[t](second_row_block_idx[i])],
                    ]
                )
                for i in range(num_blocks)
            ]
            replicating_matrix = block_diag(*blocks)

            portfolio_value = V[t].data.to_numpy()[:: num_repeats // 2]
            portfolio_vec = np.linalg.solve(replicating_matrix, portfolio_value)

            B_arr = np.repeat(portfolio_vec[::2], num_repeats)
            N_arr = np.repeat(portfolio_vec[1::2], num_repeats)

            B[t] = RandomVariable(domain=S.domain).from_numpy(B_arr)
            N[t] = RandomVariable(domain=S.domain).from_numpy(N_arr)
            V[t - 1] = B[t] + N[t] * S[t - 1]

        B[0] = B[1]
        N[0] = N[1]

        B_data = pd.concat([B[t].data for t in self.time], axis=1)
        B_data.columns = S.time

        N_data = pd.concat([N[t].data for t in self.time], axis=1)
        N_data.columns = S.time

        V_data = pd.concat([V[t].data for t in self.time], axis=1)
        V_data.columns = S.time

        B = StochasticProcess(
            time=self.time,
            domain=self.sample_space,
            name="non_risky",
            is_discrete_state=True,
        ).from_pandas(B_data)
        N = StochasticProcess(
            time=self.time,
            domain=self.sample_space,
            name="risky",
            is_discrete_state=True,
        ).from_pandas(N_data)
        V = StochasticProcess(
            time=self.time,
            domain=self.sample_space,
            name="portfolio_value",
            is_discrete_state=True,
        ).from_pandas(V_data)

        B._is_enumerated = True
        N._is_enumerated = True
        V._is_enumerated = True
        B._probability_measure = self.risk_neutral_prob
        N._probability_measure = self.risk_neutral_prob
        V._probability_measure = self.risk_neutral_prob

        return B, N, V, V[0].data.to_numpy()[0]
