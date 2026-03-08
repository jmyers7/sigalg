"""Module for contingent claims."""

from numbers import Real

from sigalg.core.random_objects.random_variable import RandomVariable


# TODO: Write unit tests
def european_option(
    price: RandomVariable, strike: Real, option_type: str = "call"
) -> RandomVariable:
    r"""Return a European option contingent claim.

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
    price : RandomVariable
        A random variable representing the underlying asset price at maturity.
    strike : Real
        The strike price of the European option.
    option_type : str, default "call"
        The type of the European option. It can be either "call" or "put".

    Raises
    ------
    TypeError
        If the strike price is not a positive real number or if the price is not a RandomVariable.

    Returns
    -------
    option : RandomVariable
        A random variable representing the payoff of the European option.

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
    if not isinstance(strike, Real) or strike <= 0:
        raise TypeError("Strike price must be a positive real number.")
    if not isinstance(price, RandomVariable):
        raise TypeError("Price must be a RandomVariable.")
    if not isinstance(option_type, str) or option_type not in ["call", "put"]:
        raise TypeError("Option type must be either 'call' or 'put'.")

    if option_type == "call":
        option = (price - strike) * (price - strike >= 0)
    else:
        option = (strike - price) * (strike - price >= 0)

    return option.with_name(f"european_{option_type}")
