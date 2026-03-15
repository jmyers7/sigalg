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

    for each $t=0,1,\ldots,T-1$, where $Z_t$ is a random variable that takes the value $u>1$ with some probability $p$ and the value $d = 1/u$ with probability $1-p$. The process $Z_t$ is called the *driving process* of the model, and the factors $u$ and $d$ are called the *up-factor* and *down-factor*, respectively.

    The risky asset is assumed to be traded in a market along with a non-risky asset with gross return $R = 1 + r$ at each time step, where $r$ is the *risk-free rate*. The non-risky asset is often conceptualized as a *bank account* with per-period interest rate $r$.

    The probability $p$ is the real-world probability that drives the price process of the stock. However, under the *no-arbitrage condition* $d < R< u$, a second probability $q$, called the *risk-neutral probability*, may be defined via the equation

    $$
    q = \frac{R - d}{u - d}.
    $$

    This risk-neutral probability is the key component in pricing various contingent claims using the binomial model. As a subclass of `StochasticProcess`, an instance of `BinomialPricingModel` carries a `probability_measure` attribute, which corresponds to the real-world probability. The risk-neutral probabilities are accessible via the attribute `risk_neutral_prob`. The driving process $Z_t$ is contained in the attribute `driving_process`.

    Parameters
    ----------
    initial_price : Real
        The initial price of the risky asset.
    up_factor : Real
        The up-factor of the model, which must be greater than 1.
    up_prob: Real
        The real-world probability of an up move in the price of the risky asset, which must be in the interval (0,1).
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
    >>> p = 0.7 # real-world probability of an up move
    >>> r = 0.01 # risk-free rate
    >>> T = Time.discrete(length=3)
    >>> S = BinomialPricingModel(initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T).from_enumeration()
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

    # TODO: Add unit tests
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
        if not isinstance(up_prob, Real) or not (0 < up_prob < 1):
            raise TypeError("up_prob must be a real number in the interval (0, 1)")
        if not isinstance(risk_free_rate, Real) or risk_free_rate <= 0:
            raise TypeError("risk_free_rate must be a positive real number")

        self.initial_price = initial_price
        self.up_factor = up_factor
        self.up_prob = up_prob
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

    # --------------------- properties --------------------- #

    # TODO: Write unit tests
    @property
    def driving_process(self) -> StochasticProcess:
        r"""Return the driving process of the binomial pricing model.

        The driving process is an IID process $Z_t$ representing the up and down movements of the underlying asset in the binomial model. It takes the value $u$ with probability $p$ and the value $d$ with probability $1-p$, where $u$ is the up-factor, $d$ is the down-factor, and $p$ is the real-world probability of an up move. The driving process is defined for times $t=1,2,\ldots,T$, where $T$ is the final time of the model.

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
            support = {0: d, 1: u}

            Z = IIDProcess(
                distribution=bernoulli(p),
                support=support,
                time=T,
                name="driving_process",
            )
            self._driving_process = Z

        return self._driving_process

    # TODO: Write unit tests
    @property
    def risk_neutral_prob(self) -> ProbabilityMeasure:
        r"""Return the risk-neutral probability measure of the binomial pricing model.

        The risk-neutral probability measure is a probability measure under which the discounted price process of the underlying asset is a martingale. In the binomial model, the risk-neutral probability $q$ is given by

        $$
        q = \frac{R - d}{u - d},
        $$

        where $R$ is the gross risk-free return, $u$ is the up-factor, and $d$ is the down-factor.

        Returns
        -------
        risk_neutral_prob : ProbabilityMeasure
            The risk-neutral probability measure of the binomial pricing model.
        """
        if self._risk_neutral_prob is None:
            u = self.up_factor
            d = self.down_factor
            R = self.risk_free_gross_return
            q = (R - d) / (u - d)

            inverse_support = {d: 0, u: 1}
            data = self.driving_process.data.map(lambda x: inverse_support[x])

            element_wise_probabilities = bernoulli(p=q).pmf(data)
            probabilities = pd.Series(
                data=np.prod(element_wise_probabilities, axis=1),
                index=self.domain.data,
            )
            probabilities /= probabilities.sum()

            self._risk_neutral_prob = ProbabilityMeasure(
                sample_space=self.domain, name="Q"
            ).from_pandas(probabilities)

        return self._risk_neutral_prob

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
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        return self.driving_process.probability_measure.with_name(name)

    # --------------------- finance methods --------------------- #

    # TODO: Write unit tests
    def replicating_portfolio(
        self, claim: RandomVariable
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        r"""Compute the replicating portfolio for the binomial pricing model given a contingent claim.

        This method computes three proccesses $B_t$, $N_t$, and $V_t$, the first two indexed $t=0,1,\ldots,T-1$, and the third indexed $t=0,1,\ldots,T$. The value $B_t$ represents the amount of money in the bank account held in the portfolio at time $t$, the value $N_t$ represents the number of units of stock held in portfolio at time $t$, and $V_t$ is the total value of the portfolio:

        $$
        V_t = B_t + S_t N_t,
        $$

        where $S_t$ is the price of the stock at time $t$. A positive value of $B_t$ represents money held in an account gaining interest for the investor, while a negative value represents a loan on which the investor pays interest. A positive value of $N_t$ represents a *long position* on the stock, while a negative value represents a *short position*.

        The portfolio is *self-financing*, in the sense that

        $$
        B_t + S_t N_t = R B_{t-1} + S_t N_{t-1}
        $$

        for each $t=1,2,\ldots,T-1$. The right-hand side of this equation represents the evolution of the value of the portfolio over the time interval $[t-1,t]$, in which the amount $B_{t-1}$ in the bank accrues interest according to the risk-free gross return $R = 1+ r$ (where $r$ is the risk-free rate) and the price of the stock changes from $S_{t-1}$ to $S_t$. This equation just says that this evolved value is equal to the value of the portofolio $V_t$ at time $t$.

        By definition, a *contingent claim* is a function $\Phi(S_T)$, where $S_T$ is the final price of the stock. The self-financing portfolio $(B_t,N_t)$ constructed by this method *replicates* a given continent claim, in the sense that $V_T = \Phi(S_T)$. In this case, the *fair risk-neutral price* of the contingent claim is the value

        $$
        V_0 = B_0 + S_0 N_0
        $$

        of the portfolio at time $t=0$.

        Parameters
        ----------
        claim : RandomVariable
            The claim to be replicated, which must be a random variable defined on the same domain as the price process and measurable with respect to the final price's sigma algebra.

        Raises
        ------
        TypeError
            If the claim is not an instance of RandomVariable or if its domain does not match the domain of the price process.
        ValueError
            If the claim is not measurable with respect to the final price's sigma algebra.

        Returns
        -------
        non_risky, risky, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
            The non-risky asset process, risky asset process, portfolio value process, and risk-neutral price that replicate the claim.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.finance import BinomialPricingModel, european_option
        >>> S_0 = 100  # initial stock price
        >>> u = 1.1  # up-factor
        >>> p = 0.7  # probability of an up move
        >>> r = 0.01  # risk-free rate
        >>> T = Time.discrete(length=3)
        >>> S = BinomialPricingModel(
        ...     initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        ... )
        >>> # display the price process
        >>> S.from_enumeration()  # doctest: +NORMALIZE_WHITESPACE
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
        >>> final_price = S[3]  # final price of the stock at maturity t=3 of the call option
        >>> K = 100  # strike price
        >>> call_option = european_option(price=S[3], strike=100)
        >>> B, N, V, price = S.replicating_portfolio(claim=call_option)
        >>> # display the bank account balance in the portfolio
        >>> B  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'non_risky':
        time                0          1          2
        trajectory
        0          -50.150931 -24.674118   0.000000
        1          -50.150931 -24.674118   0.000000
        2          -50.150931 -24.674118 -47.147572
        3          -50.150931 -24.674118 -47.147572
        4          -50.150931 -73.822294 -47.147572
        5          -50.150931 -73.822294 -47.147572
        6          -50.150931 -73.822294 -99.009901
        7          -50.150931 -73.822294 -99.009901
        >>> # display the number of shares held in the replicating portfolio
        >>> N  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'risky':
        time               0         1        2
        trajectory
        0           0.587304  0.301542  0.00000
        1           0.587304  0.301542  0.00000
        2           0.587304  0.301542  0.52381
        3           0.587304  0.301542  0.52381
        4           0.587304  0.797939  0.52381
        5           0.587304  0.797939  0.52381
        6           0.587304  0.797939  1.00000
        7           0.587304  0.797939  1.00000
        >>> # display the value of the replicating portfolio
        >>> V  # doctest: +NORMALIZE_WHITESPACE
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
        >>> # check that the price of the call option is equal to the value of the replicating portfolio at the final time
        >>> call_option  # doctest: +NORMALIZE_WHITESPACE
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
        >>> # print the price of the call option
        >>> float(price)
        8.579463133651387
        """
        if not isinstance(claim, RandomVariable) or claim.domain != self.domain:
            raise TypeError(
                "claim must be an instance of RandomVariable with the same domain as the model"
            )

        t_final = self.time[-1]

        if not claim.is_measurable(self[t_final].sigma_algebra):
            raise ValueError(
                "claim must be measurable with respect to the final price's sigma algebra"
            )

        B = dict.fromkeys(self.time)
        N = dict.fromkeys(self.time)
        V = dict.fromkeys(self.time)
        V[t_final] = claim
        R = self.risk_free_gross_return

        for t in reversed(self.time[1:]):
            num_blocks = 2 ** (t - 1)
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
                        [R, self[t](first_row_block_idx[i])],
                        [R, self[t](second_row_block_idx[i])],
                    ]
                )
                for i in range(num_blocks)
            ]
            replicating_matrix = block_diag(*blocks)

            portfolio_value = V[t].data.to_numpy()[:: num_repeats // 2]
            portfolio_vec = np.linalg.solve(replicating_matrix, portfolio_value)

            B_arr = np.repeat(portfolio_vec[::2], num_repeats)
            N_arr = np.repeat(portfolio_vec[1::2], num_repeats)

            B[t - 1] = RandomVariable(domain=self.domain).from_numpy(B_arr)
            N[t - 1] = RandomVariable(domain=self.domain).from_numpy(N_arr)
            V[t - 1] = B[t - 1] + self[t - 1] * N[t - 1]

        B_data = pd.concat([B[t].data for t in self.time[:-1]], axis=1)
        B_data.columns = self.time[:-1]

        N_data = pd.concat([N[t].data for t in self.time[:-1]], axis=1)
        N_data.columns = self.time[:-1]

        V_data = pd.concat([V[t].data for t in self.time], axis=1)
        V_data.columns = self.time

        B = StochasticProcess(
            time=self.time[:-1],
            domain=self.domain,
            name="non_risky",
            is_discrete_state=True,
        ).from_pandas(B_data)
        N = StochasticProcess(
            time=self.time[:-1],
            domain=self.domain,
            name="risky",
            is_discrete_state=True,
        ).from_pandas(N_data)
        V = StochasticProcess(
            time=self.time,
            domain=self.domain,
            name="portfolio_value",
            is_discrete_state=True,
        ).from_pandas(V_data)

        B._probability_measure = self.risk_neutral_prob
        N._probability_measure = self.risk_neutral_prob
        V._probability_measure = self.risk_neutral_prob

        return B, N, V, V[0].data.to_numpy()[0]
