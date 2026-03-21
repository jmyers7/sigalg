"""Later."""

from __future__ import annotations

from numbers import Real

from ....core.random_objects.random_variable import RandomVariable
from ..base.claim import Claim
from ..base.geometric_pricing_model import GeometricPricingModel


class EuropeanOption(Claim):
    r"""Model a European option contingent claim.

    The European option is a claim comes in two types: *call* and *put*. The payoff of a European call option is given by

    $$
    \max(S_T - K, 0),
    $$

    while the payoff of a European put option is given by

    $$
    \max(K - S_T, 0),
    $$

    where $S_T$ is the underlying asset price at maturity and $K$ is the strike price.

    Parameters
    ----------
    pricing_model : PricingModel
        A pricing model representing the underlying asset price dynamics.
    strike : Real
        The strike price of the European option.
    option_type : str, default "call"
        The type of the European option. It can be either "call" or "put".

    Raises
    ------
    TypeError
        If the strike price is not a positive real number, or if the pricing model is not a PricingModel, or if the option type is not "call" or "put".
    """

    def __init__(
        self,
        pricing_model: GeometricPricingModel,
        strike: Real,
        option_type: str = "call",
    ):
        if not isinstance(strike, Real) or strike <= 0:
            raise TypeError("Strike price must be a positive real number.")
        if not isinstance(pricing_model, GeometricPricingModel):
            raise TypeError("Pricing model must be a PricingModel.")
        if not isinstance(option_type, str) or option_type not in ["call", "put"]:
            raise TypeError("Option type must be either 'call' or 'put'.")

        super().__init__(is_path_independent=True)

        self.pricing_model = pricing_model
        self.strike = strike
        self.option_type = option_type

    @property
    def payoff(self) -> RandomVariable:
        """Return the payoff of the European option.

        Returns
        -------
        option : RandomVariable
            A random variable representing the payoff of the European option.
        """
        if self.pricing_model.data is None:
            raise ValueError(
                "Price trajectories of the underlying asset must be enumerated before computing the payoff."
            )

        price = self.pricing_model.last_rv
        K = self.strike

        if self.option_type == "call":
            result = (price - K) * (price - K >= 0)
            return result.with_name("EuropeanCallPayoff")
        elif self.option_type == "put":
            result = (K - price) * (K - price >= 0)
            return result.with_name("EuropeanPutPayoff")

    # def replicating_portfolio(
    #     self,
    # ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
    #     r"""Compute the replicating portfolio for the European option given the pricing model.

    #     The core idea of a *replicating portfolio* is this: Suppose that an individual sells a contingent claim on an underlying asset (generically called an *underlying*). The seller accepts a premium from the buyer for the claim at time $t=0$, and then at some specified maturity time $t=T$, the seller must pay the exercise value of the claim to the buyer. The claim is a *derivative*, in the sense that its value depends on (or derives from) the price of the underlying. The seller is thus interested in hedging their short position on the claim against an increase in the price of the underlying, which would increase the exercise value of the claim that the seller would owe the buyer.

    #     The underlying asset is assumed to be traded in a market that includes a bank account with risk-free, per-period interest rate $r$. The seller's hedging strategy is to trade in the underlying asset itself, as well as hold a cash position at the bank, so that when the contingent claim matures, the seller's portfolio will cover the exercise value owed to the buyer.

    #     The replicating portfolio thus consists of a pair $(B_t,N_t)$ of processes, indexed $t=0,1,\ldots,T-1$, where $B_t$ represents the cash position at time $t$, and $N_t$ counts the number of units of the underlying held in the portfolio at time $t$. A third process $V_t$ represents the total value of the portfolio, given by

    #     $$
    #     V_t = B_t + S_t N_t,
    #     $$

    #     where $S_t$ is the price of the underlying at time $t$. A positive value of $B_t$ represents money held in the bank accruing interest for the seller at rate $r$, while a negative value represents a loan on which the seller pays interest at rate $r$. A positive value of $N_t$ represents a *long position* on the underlying, while a negative value represents a *short position*.

    #     The replicating portfolio is *self-financing*, in the sense that

    #     $$
    #     V_t = (1+r) B_{t-1} + S_t N_{t-1}
    #     $$

    #     for each $t=1,2,\ldots,T$. The right-hand side of this equation represents the evolution of the value of the portfolio over the time interval $[t-1,t]$, in which the amount $B_{t-1}$ in the bank accrues interest at rate $r$ and the price of the underlying changes from $S_{t-1}$ to $S_t$. This equation says that this evolved value of the old portfolio is equal to the value $V_t$ of the new portfolio at time $t$.

    #     The existence of the replicating portfolio also allows us to determine a fair, "risk-neutral" premium for the contingent claim paid by the buyer. Under the no-arbitrage assumption, this premium should coincide with the initial price

    #     $$
    #     V_0 = B_0 + S_0 N_0
    #     $$

    #     of the replicating portfolio.

    #     The replicating portfolio for a European option may be computed relative to the pricing model in either `dense` or `sparse` enumeration mode. The `from_enumeration` method of the pricing model must have been called before computing the replicating portfolio. If the price trajectories were generated in `sparse` mode, then this method computes the replicating portfolio along the $T+1$ price trajectories the following special forms:

    #     $$
    #     \begin{gather*}
    #     S_0 \to S_0 u \to S_0u^2 \to S_0u^3 \to \ldots \to S_0u^{T-1} \to S_0 u^T \\
    #     S_0 \to S_0 d \to S_0du \to S_0du^2 \to \ldots \to S_0du^{T-2} \to S_0 du^{T-1} \\
    #     S_0 \to S_0 d \to S_0d^2 \to S_0d^2u \to \ldots \to S_0d^2u^{T-3} \to S_0 d^2u^{T-2} \\
    #     \cdots \quad \cdots \quad \cdots \quad \\
    #     S_0 \to S_0 d \to S_0d^2 \to S_0d^3 \to \ldots \to S_0d^{T-1} \to S_0 d^T
    #     \end{gather*}
    #     $$

    #     where $u$ and $d$ are the up- and down-factors of the model.

    #     Raises
    #     ------
    #     ValueError
    #         If the price trajectories of the underlying asset have not been enumerated before calling this method.

    #     Returns
    #     -------
    #     bank_value, underlying_units, portfolio_value, risk_neutral_price : tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]
    #         A tuple containing the bank account process, the underlying units process, the total portfolio value process, and the risk-neutral price of the claim.

    #     Examples
    #     --------
    #     >>> from sigalg.core import Time
    #     >>> from sigalg.finance import BinomialPricingModel, EuropeanOption
    #     >>> S_0 = 100
    #     >>> u = 1.1
    #     >>> p = 0.7
    #     >>> r = 0.01
    #     >>> T = Time.discrete(length=3)
    #     >>> S = BinomialPricingModel(
    #     ...     initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
    #     ... )
    #     >>> S.from_enumeration(enum_mode="dense") # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'S':
    #     time          0           1           2           3
    #     trajectory
    #     0           100  110.000000  121.000000  133.100000
    #     1           100  110.000000  121.000000  110.000000
    #     2           100  110.000000  100.000000  110.000000
    #     3           100  110.000000  100.000000   90.909091
    #     4           100   90.909091  100.000000  110.000000
    #     5           100   90.909091  100.000000   90.909091
    #     6           100   90.909091   82.644628   90.909091
    #     7           100   90.909091   82.644628   75.131480
    #     >>> K = 100
    #     >>> call_option = EuropeanOption(pricing_model=S, strike=K, option_type="call")
    #     >>> B, N, V, price = call_option.replicating_portfolio()
    #     >>> print(B) # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'bank_account_value':
    #     time                0          1          2
    #     trajectory
    #     0          -50.150931 -73.822294 -99.009901
    #     1          -50.150931 -73.822294 -99.009901
    #     2          -50.150931 -73.822294 -47.147572
    #     3          -50.150931 -73.822294 -47.147572
    #     4          -50.150931 -24.674118 -47.147572
    #     5          -50.150931 -24.674118 -47.147572
    #     6          -50.150931 -24.674118  -0.000000
    #     7          -50.150931 -24.674118  -0.000000
    #     >>> print(N) # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'underlying_units':
    #     time               0         1        2
    #     trajectory
    #     0           0.587304  0.797939  1.00000
    #     1           0.587304  0.797939  1.00000
    #     2           0.587304  0.797939  0.52381
    #     3           0.587304  0.797939  0.52381
    #     4           0.587304  0.301542  0.52381
    #     5           0.587304  0.301542  0.52381
    #     6           0.587304  0.301542  0.00000
    #     7           0.587304  0.301542  0.00000
    #     >>> print(V) # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'portfolio_value':
    #     time               0          1          2     3
    #     trajectory
    #     0           8.579463  13.950993  21.990099  33.1
    #     1           8.579463  13.950993  21.990099  10.0
    #     2           8.579463  13.950993   5.233380  10.0
    #     3           8.579463  13.950993   5.233380  -0.0
    #     4           8.579463   2.738827   5.233380  10.0
    #     5           8.579463   2.738827   5.233380  -0.0
    #     6           8.579463   2.738827  -0.000000  -0.0
    #     7           8.579463   2.738827  -0.000000  -0.0
    #     >>> print(price)
    #     8.57946313365138
    #     >>> S.from_enumeration(enum_mode="sparse") # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'S':
    #     time            0           1           2           3
    #     trajectory
    #     0           100.0  110.000000  121.000000  133.100000
    #     1           100.0   90.909091  100.000000  110.000000
    #     2           100.0   90.909091   82.644628   90.909091
    #     3           100.0   90.909091   82.644628   75.131480
    #     >>> K = 100
    #     >>> call_option = EuropeanOption(pricing_model=S, strike=K, option_type="call")
    #     >>> B, N, V, price = call_option.replicating_portfolio()
    #     >>> print(B) # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'bank_account_value':
    #     time                0          1          2
    #     trajectory
    #     0          -50.150931 -73.822294 -99.009901
    #     1          -50.150931 -24.674118 -47.147572
    #     2          -50.150931 -24.674118  -0.000000
    #     3          -50.150931 -24.674118  -0.000000
    #     >>> print(N) # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'underlying_units':
    #     time               0         1        2
    #     trajectory
    #     0           0.587304  0.797939  1.00000
    #     1           0.587304  0.301542  0.52381
    #     2           0.587304  0.301542  0.00000
    #     3           0.587304  0.301542  0.00000
    #     >>> print(V) # doctest: +NORMALIZE_WHITESPACE
    #     Stochastic process 'portfolio_value':
    #     time               0          1          2     3
    #     trajectory
    #     0           8.579463  13.950993  21.990099  33.1
    #     1           8.579463   2.738827   5.233380  10.0
    #     2           8.579463   2.738827  -0.000000  -0.0
    #     3           8.579463   2.738827  -0.000000  -0.0
    #     >>> print(price)
    #     8.57946313365138
    #     """
    #     if self.pricing_model.data is None:
    #         raise ValueError(
    #             "Price trajectories of the underlying asset must be enumerated before computing the replicating portfolio for a European option."
    #         )

    #     S = self.pricing_model
    #     T = S.time[-1]
    #     claim = self.payoff

    #     if S.enum_mode == "sparse":
    #         claim_arr = claim.data.values
    #         B_data, N_data, V_data = self._generate_sparse_replicating_data(
    #             S, claim_arr
    #         )

    #     elif S.enum_mode == "dense":
    #         S_dense = S.data
    #         S_sparse = pd.DataFrame(S._sparse_price_array)

    #         claim_dict = dict(zip(S[T].data, claim.data, strict=False))
    #         claim_arr = S_sparse[T].map(claim_dict).values

    #         B_data, N_data, V_data = self._generate_sparse_replicating_data(
    #             S, claim_arr
    #         )

    #         B_data = S_dense.iloc[:, :-1].apply(
    #             lambda col: col.map(
    #                 dict(zip(S_sparse[col.name], B_data[col.name], strict=False))
    #             )
    #         )

    #         N_data = S_dense.iloc[:, :-1].apply(
    #             lambda col: col.map(
    #                 dict(zip(S_sparse[col.name], N_data[col.name], strict=False))
    #             )
    #         )

    #         V_data = S_dense.apply(
    #             lambda col: col.map(
    #                 dict(zip(S_sparse[col.name], V_data[col.name], strict=False))
    #             )
    #         )

    #     else:
    #         raise ValueError("Enumeration mode must be either 'sparse' or 'dense'")

    #     B = StochasticProcess(
    #         domain=S.domain,
    #         time=S.time[:-1],
    #         name="bank_account_value",
    #         is_discrete_state=True,
    #     ).from_pandas(B_data)
    #     N = StochasticProcess(
    #         domain=S.domain,
    #         time=S.time[:-1],
    #         name="underlying_units",
    #         is_discrete_state=True,
    #     ).from_pandas(N_data)
    #     V = StochasticProcess(
    #         domain=S.domain,
    #         time=S.time,
    #         name="portfolio_value",
    #         is_discrete_state=True,
    #     ).from_pandas(V_data)

    #     return B, N, V, V[0].data.to_numpy()[0]

    # @staticmethod
    # def _generate_sparse_replicating_data(
    #     pricing_model: PricingModel, claim_arr: np.ndarray
    # ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     u = pricing_model.up_factor
    #     d = pricing_model.down_factor
    #     R = pricing_model.risk_free_gross_return
    #     q = pricing_model.risk_neutral_prob
    #     T = pricing_model.time[-1]
    #     S = pricing_model._sparse_price_array

    #     V = dict.fromkeys(pricing_model.time)
    #     B = dict.fromkeys(pricing_model.time[:-1])
    #     N = dict.fromkeys(pricing_model.time[:-1])

    #     V_arr = np.zeros(shape=(T + 1, T + 1))
    #     N_arr = np.zeros(shape=(T + 1, T))
    #     B_arr = np.zeros(shape=(T + 1, T))

    #     V[T] = claim_arr
    #     V_arr[:, T] = V[T]

    #     for t in reversed(range(T)):
    #         V[t] = (q * V[t + 1][:-1] + (1 - q) * V[t + 1][1:]) / R
    #         N[t] = (V[t + 1][:-1] - V[t + 1][1:]) / (u - d) / S[: (t + 1), t]
    #         B[t] = V[t] - S[: (t + 1), t] * N[t]

    #         V_arr[:, t] = np.concatenate((V[t], np.repeat(V[t][-1], T - t)))
    #         N_arr[:, t] = np.concatenate((N[t], np.repeat(N[t][-1], T - t)))
    #         B_arr[:, t] = np.concatenate((B[t], np.repeat(B[t][-1], T - t)))

    #     return pd.DataFrame(B_arr), pd.DataFrame(N_arr), pd.DataFrame(V_arr)
