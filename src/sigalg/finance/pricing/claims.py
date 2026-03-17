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
