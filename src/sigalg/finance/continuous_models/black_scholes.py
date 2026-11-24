"""A class modeling a Black-Scholes pricing model."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .continuous_pricing_model import ContinuousPricingModel

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...core.functions.random_variable import RandomVariable
    from ...processes.base.stochastic_process import StochasticProcess
    from ...validation.index_validator import IndexLike
    from ..claims.european_option import EuropeanOption


class hybridmethod(classmethod):
    """Descriptor class for methods that function as both instance and class methods."""

    def __get__(self, instance, owner=None):  # noqa: D105
        target = owner if instance is None else instance
        return self.__func__.__get__(target, owner)


class BlackScholes(ContinuousPricingModel):
    """A class modeling a Black-Scholes pricing model.

    The base constructor is not meant to be called directly by users. Instead, the user should call the `generate` class method. See the Examples section below for usage.

    Examples
    --------
    >>> import numpy as np
    >>> from sigalg import BlackScholes, Claim, Time
    >>> rng = np.random.default_rng(42)

    Build a Black-Scholes model with 5 simulated trajectories.

    >>> S_0, r, sigma, mu = 49, 0.05, 0.2, 0.04
    >>> T = Time.continuous(start=0, stop=2, dt=0.5)
    >>> S = BlackScholes.generate(
    ...     initial_price=S_0,
    ...     risk_free_rate=r,
    ...     volatility=sigma,
    ...     expected_return=mu,
    ...     n_trajectories=5,
    ...     index=T,
    ...     random_state=rng,
    ... )
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Black-Scholes model 'S':
    t      0.0        0.5        1.0        1.5        2.0
    omega
    0       49  52.191195  45.963051  52.141838  60.763155
    1       49  37.936065  32.192911  33.442436  32.625760
    2       49  49.871229  45.096545  52.100330  59.333120
    3       49  50.458865  60.374959  65.804632  59.451921
    4       49  52.665971  46.916201  54.195327  54.901139

    Define a European call option with the Black-Scholes model as the model for the price process of the underlying asset.

    >>> K = 50
    >>> Phi = Claim.european_option(strike=K, model=S, option_type="call")

    Print the payoffs of the option.

    >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
    European option 'Phi':
    t      0.0  0.5  1.0  1.5        2.0
    omega
    0      0.0  0.0  0.0  0.0  10.763155
    1      0.0  0.0  0.0  0.0   0.000000
    2      0.0  0.0  0.0  0.0   9.333120
    3      0.0  0.0  0.0  0.0   9.451921
    4      0.0  0.0  0.0  0.0   4.901139

    Compute the price process of the claim.

    >>> X = S.price_process(claim=Phi)
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    t           0.0       0.5        1.0        1.5        2.0
    omega
    0      7.386405  8.249011   2.979762   4.844518  10.763155
    1      7.386405  1.184672   0.075325   0.006723   0.000000
    2      7.386405  6.635949   2.587312   4.815220   9.333120
    3      7.386405  7.030119  13.421694  17.088206   9.451921
    4      7.386405  8.597073   3.448950   6.387844   4.901139

    Compute the Delta-hedging process for the claim.

    >>> Delta = S.delta(claim=Phi)
    >>> print(Delta)  # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'Delta':
    t           0.0       0.5       1.0       1.5  2.0
    omega
    0      0.664052  0.726999  0.471728  0.706807  1.0
    1      0.664052  0.242402  0.032057  0.004709  0.0
    2      0.664052  0.662075  0.434044  0.704867  1.0
    3      0.664052  0.679377  0.901952  0.985726  1.0
    4      0.664052  0.739151  0.512644  0.793097  1.0
    """

    _repr_name = "BlackScholesModel"
    _str_name = "Black-Scholes model"

    # --------------------- constructors --------------------- #

    @ContinuousPricingModel.generator
    def generate(
        cls,
        initial_price: Real,
        risk_free_rate: Real,
        volatility: Real,
        expected_return: Real | None = None,
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
        initial_price : Real
            The initial price of the asset.
        risk_free_rate : Real
            The continuous risk-free rate.
        volatility : Real
            The volatility of the asset.
        expected_return : Real | None, default=None
            The expected return of the underlying asset. If `None`, will be set equal to `risk_free_rate`.
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

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg import BlackScholes, Claim, Time
        >>> rng = np.random.default_rng(42)

        Build a Black-Scholes model with 5 simulated trajectories.

        >>> S_0, r, sigma, mu = 49, 0.05, 0.2, 0.04
        >>> T = Time.continuous(start=0, stop=2, dt=0.5)
        >>> S = BlackScholes.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     volatility=sigma,
        ...     expected_return=mu,
        ...     n_trajectories=5,
        ...     index=T,
        ...     random_state=rng,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Black-Scholes model 'S':
        t      0.0        0.5        1.0        1.5        2.0
        omega
        0       49  52.191195  45.963051  52.141838  60.763155
        1       49  37.936065  32.192911  33.442436  32.625760
        2       49  49.871229  45.096545  52.100330  59.333120
        3       49  50.458865  60.374959  65.804632  59.451921
        4       49  52.665971  46.916201  54.195327  54.901139

        Define a European call option with the Black-Scholes model as the model for the price process of the underlying asset.

        >>> K = 50
        >>> Phi = Claim.european_option(strike=K, model=S, option_type="call")

        Print the payoffs of the option.

        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        European option 'Phi':
        t      0.0  0.5  1.0  1.5        2.0
        omega
        0      0.0  0.0  0.0  0.0  10.763155
        1      0.0  0.0  0.0  0.0   0.000000
        2      0.0  0.0  0.0  0.0   9.333120
        3      0.0  0.0  0.0  0.0   9.451921
        4      0.0  0.0  0.0  0.0   4.901139

        Compute the price process of the claim.

        >>> X = S.price_process(claim=Phi)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        t           0.0       0.5        1.0        1.5        2.0
        omega
        0      7.386405  8.249011   2.979762   4.844518  10.763155
        1      7.386405  1.184672   0.075325   0.006723   0.000000
        2      7.386405  6.635949   2.587312   4.815220   9.333120
        3      7.386405  7.030119  13.421694  17.088206   9.451921
        4      7.386405  8.597073   3.448950   6.387844   4.901139

        Compute the Delta-hedging process for the claim.

        >>> Delta = S.delta(claim=Phi)
        >>> print(Delta)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Delta':
        t           0.0       0.5       1.0       1.5  2.0
        omega
        0      0.664052  0.726999  0.471728  0.706807  1.0
        1      0.664052  0.242402  0.032057  0.004709  0.0
        2      0.664052  0.662075  0.434044  0.704867  1.0
        3      0.664052  0.679377  0.901952  0.985726  1.0
        4      0.664052  0.739151  0.512644  0.793097  1.0
        """
        cls._validate_pricing_params(
            initial_price=initial_price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
        )

        if expected_return is None:
            expected_return = risk_free_rate

        return {
            "initial_price": initial_price,
            "risk_free_rate": risk_free_rate,
            "volatility": volatility,
            "expected_return": expected_return,
        }

    @staticmethod
    def _validate_pricing_params(
        initial_price: Real | None = None,
        risk_free_rate: Real | None = None,
        volatility: Real | None = None,
        expected_return: Real | None = None,
        time_to_maturity: Real | None = None,
        strike: Real | None = None,
        option_type: Literal["call", "put"] | None = None,
    ) -> None:

        if initial_price is not None and (
            not isinstance(initial_price, Real) or initial_price <= 0
        ):
            raise ValueError("initial_price must be a positive real number.")
        if risk_free_rate is not None and (
            not isinstance(risk_free_rate, Real) or risk_free_rate <= 0
        ):
            raise ValueError("risk_free_rate must be a positive real number.")
        if volatility is not None and (
            not isinstance(volatility, Real) or volatility <= 0
        ):
            raise ValueError("volatility must be a positive real number.")
        if expected_return is not None and (
            not isinstance(expected_return, Real) or expected_return <= 0
        ):
            raise ValueError("expected must be a positive real number.")
        if time_to_maturity is not None and (
            not isinstance(time_to_maturity, Real) or time_to_maturity <= 0
        ):
            raise ValueError("time_to_maturity must be a positive real number.")
        if strike is not None and (not isinstance(strike, Real) or strike <= 0):
            raise ValueError("strike must be a positive real number.")
        if option_type is not None and option_type not in ["call", "put"]:
            raise TypeError("option_type must be 'call' or 'put")

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        risk_free_rate: Real,
        volatility: Real | None = None,
        expected_return: Real | None = None,
        name: Hashable = "S",
    ) -> BlackScholes:
        """Generate a Black-Scholes model from a `pd.DataFrame`.

        Parameters
        ----------
        data : pd.DataFrame
            The data frame containing the price trajectories.
        risk_free_rate : Real
            The continuous risk-free rate.
        volatility : Real | None, default=None
            The volatility of the asset. If `None`, it will be estimated as a historical volatility.
        expected_return : Real | None, default=None
            The expected return of the underlying asset. If `None`, will be set equal to `risk_free_rate`.
        name : Hashable, default="S"
            The name of the model.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg import BlackScholes, Claim

        Define data for the price trajectories. (This is a somewhat artifical example in which we have five price trajectories.)

        >>> data = pd.DataFrame(
        ...     {
        ...         0.0: [49, 49, 49, 49, 49],
        ...         0.5: [52.45, 38.13, 50.12, 50.71, 52.93],
        ...         1.0: [46.42, 32.52, 45.55, 60.98, 47.39],
        ...         1.5: [52.93, 33.95, 52.89, 66.80, 55.01],
        ...         2.0: [61.99, 33.28, 60.53, 60.65, 56.01],
        ...     },
        ... )

        Build a Black-Scholes model using the data.

        >>> S = BlackScholes.from_data(data=data, risk_free_rate=0.05, expected_return=0.045)
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Black-Scholes model 'S':
        t      0.0    0.5    1.0    1.5    2.0
        omega
        0       49  52.45  46.42  52.93  61.99
        1       49  38.13  32.52  33.95  33.28
        2       49  50.12  45.55  52.89  60.53
        3       49  50.71  60.98  66.80  60.65
        4       49  52.93  47.39  55.01  56.01

        Define a European call option with the Black-Scholes model as the model for the price process of the underlying asset.

        >>> K = 50
        >>> Phi = Claim.european_option(strike=K, model=S, option_type="call")

        Print the payoffs of the option.

        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        European option 'Phi':
        t      0.0  0.5  1.0  1.5    2.0
        omega
        0      0.0  0.0  0.0  0.0  11.99
        1      0.0  0.0  0.0  0.0   0.00
        2      0.0  0.0  0.0  0.0  10.53
        3      0.0  0.0  0.0  0.0  10.65
        4      0.0  0.0  0.0  0.0   6.01

        Compute the price process of the claim.

        >>> X = S.price_process(claim=Phi)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        t           0.0       0.5        1.0        1.5    2.0
        omega
        0      6.732889  7.899673   2.719060   5.109367  11.99
        1      6.732889  0.865099   0.034833   0.002333   0.00
        2      6.732889  6.227428   2.321031   5.078744  10.53
        3      6.732889  6.634800  13.740568  18.046255  10.65
        4      6.732889  8.264169   3.206782   6.796204   6.01

        Compute the Delta-hedging process for the claim.

        >>> Delta = S.delta(claim=Phi)
        >>> print(Delta)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Delta':
        t           0.0       0.5       1.0       1.5  2.0
        omega
        0      0.672669  0.752669  0.479031  0.766536  1.0
        1      0.672669  0.208064  0.017970  0.001979  0.0
        2      0.672669  0.680719  0.435926  0.764651  1.0
        3      0.672669  0.700076  0.935131  0.995588  1.0
        4      0.672669  0.765975  0.526413  0.850999  1.0
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame")
        cls._validate_pricing_params(
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            expected_return=expected_return,
        )

        if expected_return is None:
            expected_return = risk_free_rate

        if volatility is None:
            log_returns = np.log(data).diff(axis=1).iloc[:, 1:]

            dt = np.diff(data.columns.to_numpy())
            if not np.allclose(dt, dt[0]):
                raise ValueError("The time steps in the data are not uniform.")
            dt = dt[0]

            volatility = log_returns.stack().std(ddof=1) / np.sqrt(dt)

        process = cls.with_uniform(mapping=data, index_kind="Time", name=name)
        process._risk_free_rate = risk_free_rate
        process._volatility = volatility
        process._expected_return = expected_return

        return process

    # --------------------- properties --------------------- #

    @property
    def risk_free_rate(self) -> Real:
        """Get the continuous risk-free rate of the model."""
        return self._risk_free_rate

    @property
    def risk_free_gross_return(self) -> Real:
        """Get the risk-free gross return."""
        dt = self.time[1] - self.time[0]
        return np.exp(self.risk_free_rate * dt)

    @property
    def volatility(self) -> Real:
        """Get the volatility of the model."""
        return self._volatility

    @property
    def expected_return(self) -> Real:
        """Get the expected return of the model."""
        return self._expected_return

    # --------------------- finance methods --------------------- #

    @hybridmethod
    def price(
        self_or_cls,
        claim: EuropeanOption | None = None,
        time_to_maturity: Real | None = None,
        price: Real | None = None,
        strike: Real | None = None,
        risk_free_rate: Real | None = None,
        volatility: Real | None = None,
        option_type: Literal["call", "put"] = "call",
        **kwargs,
    ) -> Real | RandomVariable:
        """Compute the Black-Scholes price of a European call or put option.

        This method may be called either as a class method, in which case explicit parameters like the current price and time to maturity must be passed, or the method may be called on an instance of `BlackScholes` with an explict European option passed as the `claim` parameter.

        Parameters
        ----------
        claim : EuropeanOption | None, default=None
            The contingent claim to price. If not given, the pricing parameters must be given.
        time_to_maturity : Real | None, default=None
            The time to maturity of the option.
        price : Real | None, default=None
            The current price of the underlying asset.
        strike : Real | None, default=None
            The strike price of the option.
        risk_free_rate : Real | None, default=None
            The continuous risk-free rate.
        volatility : Real | None, default=None
            The volatility of the underlying asset
        option_type : Literal["call", "put"], default="call"
            The option type.

        Returns
        -------
        price : Real | np.ndarray
            If the method is used as a class method, a real number is returned. If it is used as an instance method, an array containing the prices of each trajectory is return.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg import BlackScholes, Claim, Time
        >>> rng = np.random.default_rng(42)

        Define pricing parameters.

        >>> S_0, K, r, tau, sigma, mu = 49, 50, 0.05, 20 / 52, 0.2, 0.04

        Compute a price using the method as a class method with explicit pricing parameters.

        >>> price = BlackScholes.price(
        ...     time_to_maturity=tau,
        ...     price=S_0,
        ...     strike=K,
        ...     risk_free_rate=r,
        ...     volatility=sigma,
        ...     option_type="call",
        ... )
        >>> price
        2.4005273232717137

        Instead, the method may be called on an instance of `BlackScholes`. Define such an instance using the same parameters as above.

        >>> T = Time.continuous(start=0, stop=tau, dt=0.01)
        >>> S = BlackScholes.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     volatility=sigma,
        ...     expected_return=mu,
        ...     n_trajectories=10,
        ...     index=T,
        ...     random_state=rng,
        ... )

        Define a `Claim` instance and price it.

        >>> Phi = Claim.european_option(strike=K, model=S)
        >>> print(S.price(claim=Phi))  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_0_0':
                  X_0_0
        omega
        0      2.400527
        1      2.400527
        2      2.400527
        3      2.400527
        4      2.400527
        5      2.400527
        6      2.400527
        7      2.400527
        8      2.400527
        9      2.400527
        """
        from scipy.stats import norm

        from ...core.functions.random_variable import RandomVariable
        from ..claims.european_option import EuropeanOption

        BlackScholes._validate_pricing_params(
            initial_price=price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            strike=strike,
            option_type=option_type,
        )

        if claim is not None and not isinstance(claim, EuropeanOption):
            raise TypeError(
                "If passed, the claim must be an instance of EuropeanOption"
            )

        if isinstance(self_or_cls, type):
            if claim is not None:
                raise ValueError(
                    "If the price method is used as a class method, an explicit claim cannot be passed."
                )

            tau, s, K, r, sigma = (
                time_to_maturity,
                price,
                strike,
                risk_free_rate,
                volatility,
            )

            d_plus = (np.log(s / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
            d_neg = d_plus - sigma * np.sqrt(tau)
            N = norm.cdf

            if option_type == "call":
                return (s * N(d_plus) - K * np.exp(-r * tau) * N(d_neg)).astype(Real)

            else:
                return (K * np.exp(-r * tau) * N(-d_neg) - s * N(-d_plus)).astype(Real)

        else:
            if claim is None:
                raise ValueError(
                    "If the price method is used as an instance method, an explicit claim must be passed."
                )

            tau, s, K, r, sigma = (
                claim.time[-1] - claim.time[0],
                np.array(self_or_cls[claim.time[0]]),
                claim.strike,
                self_or_cls.risk_free_rate,
                self_or_cls.volatility,
            )

            d_plus = (np.log(s / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
            d_neg = d_plus - sigma * np.sqrt(tau)
            N = norm.cdf

            if option_type == "call":
                pricing_arr = s * N(d_plus) - K * np.exp(-r * tau) * N(d_neg)

            else:
                pricing_arr = K * np.exp(-r * tau) * N(-d_neg) - s * N(-d_plus)

            data = pd.Series(pricing_arr, index=self_or_cls.domain.data)

            return RandomVariable._from_validated(
                data=data,
                sig_alg=self_or_cls.sig_alg,
                measure=self_or_cls.measure,
                index_kind=None,
                index_name=None,
                name=f"X_{claim.time[0]}".replace(".", "_"),
            )

    def price_process(
        self, claim: EuropeanOption | None = None, **kwargs
    ) -> StochasticProcess:
        """Compute the price process of a European call or put option.

        Parameters
        ----------
        claim : EuropeanOption | None, default=None
            The contingent claim to price.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg import BlackScholes, Claim, Time
        >>> rng = np.random.default_rng(42)

        Build a Black-Scholes model with 5 simulated trajectories.

        >>> S_0, r, sigma, mu = 49, 0.05, 0.2, 0.4
        >>> T = Time.continuous(start=0, stop=2, dt=0.5)
        >>> S = BlackScholes.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     volatility=sigma,
        ...     expected_return=mu,
        ...     n_trajectories=5,
        ...     index=T,
        ...     random_state=rng,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Black-Scholes model 'S':
        t      0.0        0.5        1.0         1.5         2.0
        omega
        0       49  62.484205  65.880193   89.475752  124.833844
        1       49  45.417716  46.143046   57.387449   67.027446
        2       49  59.706701  64.638204   89.404524  121.895933
        3       49  60.410229  86.537204  112.921200  122.140000
        4       49  63.052615  67.246371   92.999553  112.790724

        Define a European call option with the Black-Scholes model as the model for the price process of the underlying asset.

        >>> K = 50
        >>> Phi = Claim.european_option(strike=K, model=S, option_type="call")

        Print the payoffs of the option.

        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        European option 'Phi':
        t      0.0  0.5  1.0  1.5        2.0
        omega
        0      0.0  0.0  0.0  0.0  74.833844
        1      0.0  0.0  0.0  0.0  17.027446
        2      0.0  0.0  0.0  0.0  71.895933
        3      0.0  0.0  0.0  0.0  72.140000
        4      0.0  0.0  0.0  0.0  62.790724

        Compute the price process of the claim.

        >>> X = S.price_process(claim=Phi)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        t           0.0        0.5        1.0        1.5        2.0
        omega
        0      7.386405  16.809103  18.560704  40.710274  74.833844
        1      7.386405   4.006231   3.065372   9.084691  17.027446
        2      7.386405  14.327734  17.376318  40.639046  71.895933
        3      7.386405  14.947097  38.980742  64.155705  72.140000
        4      7.386405  17.327843  19.875574  44.234062  62.790724
        """
        from scipy.stats import norm

        from ...processes.base.stochastic_process import StochasticProcess
        from ..claims.european_option import EuropeanOption

        if not isinstance(claim, EuropeanOption):
            raise TypeError("The claim must be an instance of EuropeanOption")

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

    @hybridmethod
    def delta(
        self_or_cls,
        claim: EuropeanOption | None = None,
        time_to_maturity: Real | None = None,
        price: Real | None = None,
        strike: Real | None = None,
        risk_free_rate: Real | None = None,
        volatility: Real | None = None,
        option_type: Literal["call", "put"] = "call",
        **kwargs,
    ) -> Real | StochasticProcess:
        """Compute the delta process of a European call or put option.

        This method may be called either as a class method, in which case explicit parameters like the current price and time to maturity must be passed, or the method may be called on an instance of `BlackScholes` with an explict European option passed as the `claim` parameter.

        Parameters
        ----------
        claim : EuropeanOption | None, default=None
            The contingent claim whose delta process is desired. If not given, the pricing parameters must be given.
        time_to_maturity : Real | None, default=None
            The time to maturity of the option.
        price : Real | None, default=None
            The current price of the underlying asset.
        strike : Real | None, default=None
            The strike price of the option.
        risk_free_rate : Real | None, default=None
            The continuous risk-free rate.
        volatility : Real | None, default=None
            The volatility of the underlying asset
        option_type : Literal["call", "put"], default="call"
            The option type.

        Returns
        -------
        delta : Real | StochasticProcess
            If `claim` is passed, the entire delta process is returned. Otherwise, if only the pricing parameters are passed, then a single real number is returned.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg import BlackScholes, Claim, Time
        >>> rng = np.random.default_rng(42)

        Define pricing parameters.

        >>> S_0, K, r, tau, sigma, mu = 49, 50, 0.05, 20 / 52, 0.2, 0.4

        Compute a delta using the method as a class method with explicit pricing parameters.

        >>> delta = BlackScholes.delta(
        ...     time_to_maturity=tau,
        ...     price=S_0,
        ...     strike=K,
        ...     risk_free_rate=r,
        ...     volatility=sigma,
        ...     option_type="call",
        ... )
        >>> delta
        0.5216046610663964

        Instead, the method may be called on an instance of `BlackScholes`. Define such an instance using the same parameters as above.

        >>> T = Time.continuous(start=0, stop=tau, dt=0.01)
        >>> S = BlackScholes.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     volatility=sigma,
        ...     expected_return=mu,
        ...     n_trajectories=10,
        ...     index=T,
        ...     random_state=rng,
        ... )

        Define a `Claim` instance and compute its delta process.

        >>> Phi = Claim.european_option(strike=K, model=S)
        >>> Delta = S.delta(claim=Phi)
        >>> print(Delta[0])  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Delta_0_0':
               Delta_0_0
        omega
        0       0.521605
        1       0.521605
        2       0.521605
        3       0.521605
        4       0.521605
        5       0.521605
        6       0.521605
        7       0.521605
        8       0.521605
        9       0.521605
        """
        from scipy.stats import norm

        from ...processes.base.stochastic_process import StochasticProcess
        from ..claims.european_option import EuropeanOption

        BlackScholes._validate_pricing_params(
            initial_price=price,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            strike=strike,
            option_type=option_type,
        )

        if claim is not None and not isinstance(claim, EuropeanOption):
            raise TypeError(
                "If passed, the claim must be an instance of EuropeanOption"
            )

        if isinstance(self_or_cls, type):
            if claim is not None:
                raise ValueError(
                    "If the delta method is used as a class method, an explicit claim cannot be passed."
                )

            tau, s, K, r, sigma = (
                time_to_maturity,
                price,
                strike,
                risk_free_rate,
                volatility,
            )

            d_plus = (np.log(s / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
            N = norm.cdf

            if option_type == "call":
                return N(d_plus).astype(Real)
            else:
                return (N(d_plus) - 1).astype(Real)

        else:
            self = self_or_cls

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
            data.insert(
                loc=len(self.index.data) - 1,
                column=self.index.data[-1],
                value=(claim.data.iloc[:, -1] > 0).astype(float),
            )

            return StochasticProcess._from_validated(
                data=data,
                sig_alg=self.sig_alg,
                measure=self.measure,
                index_kind="Time",
                index_name=self.index_name,
                name="Delta",
            )

    def hedging_portfolio(
        self, claim: EuropeanOption, **kwargs
    ) -> tuple[
        StochasticProcess,
        StochasticProcess,
        StochasticProcess,
    ]:
        """Compute the Delta-hedging portfolio for a given European option.

        Parameters
        ----------
        claim : EuropeanOption
            The option for which to compute the replicating portfolio.

        Returns
        -------
        hedging_portfolio : tuple[StochasticProcess, StochasticProcess, StochasticProcess,]
            Three processes `Delta`, `B`, and `V`. The first gives the number of units of the risky asset, the second gives the cash value of the position in the non-risky asset, and the third is the value process of the two-asset hedging portfolio.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg import BlackScholes, Claim, Time
        >>> rng = np.random.default_rng(42)

        Define a Black-Scholes model. Notice that we will rebalance the hedging portfolio over subintervals of length `0.001`. We simulate `1_000` price trajectories.

        >>> S_0, r, sigma = 49, 0.05, 0.1
        >>> T = Time.continuous(start=0, stop=1, dt=0.001)
        >>> S = BlackScholes.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     volatility=sigma,
        ...     n_trajectories=1_000,
        ...     index=T,
        ...     random_state=rng,
        ... )

        Define a European call option with strike `K=50`, and get the risk-free gross return of the model.

        >>> K = 50
        >>> Phi = Claim.european_option(strike=K, model=S, option_type="call")
        >>> R = S.risk_free_gross_return

        Compute the hedging portfolio.

        >>> Delta, B, V = S.hedging_portfolio(Phi)

        The sum of the value of the portfolio at maturity and the premium (with interest earned) is supposed to replicate the payoff of the option. Thus the present value of this sum should be close to the theoretical price of the option. We check this.

        >>> premium = S.price(Phi).item()
        >>> round(premium, 2)
        2.73

        >>> simulated_price = (V.discount(gross_return=R) + premium).last_rv.expectation().item()
        >>> round(simulated_price, 2)
        2.79

        The stochastic process `V` is the value process for the hedging portfolio. It is self-financing, and so does not require any infusions or withdrawals of capital. One may then show that the discounted value process `V` is equal to the Itô integral of the process `Delta` against the discounted price process `S`. We check this.

        >>> V_discount = V.discount(gross_return=R)
        >>> S_discount = S.discount(gross_return=R)
        >>> np.allclose(V_discount, Delta.ito_integral(S_discount))
        True
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from ..claims.european_option import EuropeanOption

        if claim is not None and not isinstance(claim, EuropeanOption):
            raise TypeError(
                "If passed, the claim must be an instance of EuropeanOption"
            )

        R = self.risk_free_gross_return
        Delta = self.delta(claim=claim)
        Delta_data = Delta.data.drop(columns=self.data.columns[-1])

        powers = pd.Index(range(len(self.time.data)))
        discount_factors = R ** (-powers)
        discounted_price = self.data.multiply(discount_factors, axis=1)

        increments = -1 * discounted_price.diff(periods=-1, axis=1).dropna(axis=1)

        data = (Delta_data * increments).cumsum(axis=1)
        data.insert(loc=0, column="initial", value=0)
        data.columns = self.time.data

        future_value_factors = (1 / R) ** (-powers)
        data = data.multiply(future_value_factors, axis=1)

        V = StochasticProcess._from_validated(
            data=data,
            sig_alg=self.sig_alg,
            measure=self.measure,
            index_kind="Time",
            index_name=self.index_name,
            name="V",
        )

        B = (V - self * Delta).with_name("B")

        return Delta, B, V

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        dt = self._index.data[1] - self._index.data[0]
        initial_time = self._index.data[0]

        increments_arr = self._random_state.normal(
            scale=np.sqrt(dt), size=(self._n_trajectories, len(self._index) - 1)
        )
        increments = pd.DataFrame(increments_arr, columns=self._index.data[1:])

        drift_term = pd.DataFrame(
            self._expected_return * dt,
            index=increments.index,
            columns=increments.columns,
        ).cumsum(axis=1)

        self._driver_data = increments.cumsum(axis=1)
        log_trajectories = drift_term + self._volatility * self._driver_data
        trajectories = np.exp(log_trajectories) * self._initial_price

        self._driver_data.insert(loc=0, column=initial_time, value=0.0)
        trajectories.insert(loc=0, column=initial_time, value=self._initial_price)

        return trajectories
