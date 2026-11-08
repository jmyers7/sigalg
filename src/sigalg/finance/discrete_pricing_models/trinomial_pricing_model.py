"""A class modeling a trinomial pricing model."""

from __future__ import annotations

from itertools import product
from numbers import Real
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .discrete_pricing_model import DiscretePricingModel

if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy as np

    from ...core.measures.parametrized_probability_measure import (
        ParametrizedProbabilityMeasure,
    )
    from ...core.measures.probability_measure import ProbabilityMeasure
    from ...core.spaces.domain import Domain
    from ...typing.index_like import IndexLike


class TrinomialPricingModel(DiscretePricingModel):
    """A class modeling a trinomial pricing model.

    The base constructor is not meant to be called directly by users. Instead, the user should call the `generate` class method. See the Examples section below for usage.

    Examples
    --------
    Given parameters of a trinomial model, generate all length-2 price trajectories in `enum` mode.

    >>> from sigalg.finance import TrinomialPricingModel
    >>> S_0 = 4
    >>> u = 1.2
    >>> m = 1.1
    >>> d = 0.9
    >>> p_u = 0.6
    >>> p_d = 0.1
    >>> r = 0.01
    >>> S = TrinomialPricingModel.generate(
    ...     mode="enum",
    ...     initial_price=S_0,
    ...     up_factor=u,
    ...     middle_factor=m,
    ...     down_factor=d,
    ...     up_prob=p_u,
    ...     down_prob=p_d,
    ...     risk_free_rate=r,
    ...     length=2,
    ... )
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Trinomial price process 'S':
    time    0    1     2
    sample
    0       4  4.8  5.76
    1       4  4.8  5.28
    2       4  4.8  4.32
    3       4  4.4  5.28
    4       4  4.4  4.84
    5       4  4.4  3.96
    6       4  3.6  4.32
    7       4  3.6  3.96
    8       4  3.6  3.24

    Simulate ten length-2 trajectories in `sim` mode.

    >>> S.n_trajectories = 10
    >>> S.random_state = 42
    >>> S.mode = "sim"
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Trinomial price process 'S':
    time    0    1     2
    sample
    0       4  4.4  4.84
    1       4  4.8  4.32
    2       4  4.4  5.28
    3       4  4.8  5.28
    4       4  4.4  5.28
    5       4  4.8  5.76
    6       4  4.4  4.84
    7       4  3.6  3.96
    8       4  4.8  5.76
    9       4  4.8  5.28
    """

    _repr_name = "TrinomialPricingModel"
    _str_name = "Trinomial price process"

    # --------------------- constructors --------------------- #

    @DiscretePricingModel.generator
    def generate(
        cls,
        initial_price: Real,
        up_prob: Real,
        down_prob: Real,
        up_factor: Real,
        middle_factor: Real,
        down_factor: Real,
        risk_free_rate: Real | None = None,
        risk_free_gross_return: Real | None = None,
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        domain_name: Hashable | None = None,
        index: IndexLike | None = None,
        length: int | None = None,
        name: Hashable = "S",
        random_state: int | np.random.Generator | None = None,
    ) -> TrinomialPricingModel:
        """Generate trajectories of the trinomial pricing model by either exhaustive enumeration or Monte Carlo simulation.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        mode : Literal["enum", "sim"]
            Whether to generate trajectories by exhuastive enumeration or Monte Carlo simulation.
        initial_price : Real
            The initial price of the risky asset.
        risk_free_rate : Real
            The risk-free rate of the non-risky asset, which must be positive.
        up_prob : Real
            The probability of an upward move in the price of the risky asset.
        down_prob : Real
            The probability of a downward move in the price of the risky asset.
        up_factor : Real
            The up-factor of the model. Must have `down_factor < middle_factor < up_factor`.
        middle_factor : Real
            The middle-factor of the model. Must have `down_factor < middle_factor < up_factor`.
        down_factor : Real
            The down-factor of the model. Must have `down_factor < middle_factor < up_factor`.
        enum_mode : Literal["dense", "sparse"], default="dense"
            The mode of enumeration. If the generation mode is set to `sim`, this parameter is ignore.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. If the generation mode is set to `enum`, this parameter is ignored.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        name : Hashable, default="S"
            The name of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded. If the generation mode is set to `enum`, this parameter is ignored.

        Returns
        -------
        self : TrinomialPricingModel
            The current instance with generated trajectories.
        """
        if not isinstance(initial_price, Real) or initial_price <= 0:
            raise TypeError("initial_price must be a positive real number.")
        if risk_free_rate is not None and (
            not isinstance(risk_free_rate, Real) or risk_free_rate <= 0
        ):
            raise TypeError("risk_free_rate must be a positive real number.")
        if risk_free_gross_return is not None and (
            not isinstance(risk_free_gross_return, Real) or risk_free_gross_return <= 1
        ):
            raise TypeError(
                "The risk-free gross return must be a real number greater than 1."
            )
        if risk_free_rate is None and risk_free_gross_return is None:
            raise TypeError(
                "One of risk_free_rate or risk_free_gross_return must be given."
            )
        if not isinstance(up_prob, Real) or not (0 < up_prob < 1):
            raise TypeError("up_prob must be a probability in the interval (0, 1).")
        if not isinstance(down_prob, Real) or not (0 < down_prob < 1):
            raise TypeError("down_prob must be a probability in the interval (0, 1).")
        if not isinstance(up_factor, Real) or up_factor <= 1:
            raise TypeError("up_factor must be a real number greater than 1.")
        if not isinstance(middle_factor, Real) or middle_factor <= 0:
            raise TypeError("middle_factor must be a positive real number.")
        if not isinstance(down_factor, Real) or down_factor <= 0:
            raise TypeError("down_factor must be a positive real number.")
        if up_prob + down_prob > 1:
            raise ValueError("The sum of up_prob and down_prob must be at most 1.")
        if not (down_factor < middle_factor < up_factor):
            raise ValueError("We must have down_factor < middle_factor < up_factor.")

        if risk_free_rate is None:
            risk_free_rate = risk_free_gross_return - 1

        return {
            "initial_price": initial_price,
            "risk_free_rate": risk_free_rate,
            "up_prob": up_prob,
            "down_prob": down_prob,
            "up_factor": up_factor,
            "middle_factor": middle_factor,
            "down_factor": down_factor,
        }

    # --------------------- properties --------------------- #

    @property
    def initial_price(self) -> Real | None:
        """Get the initial price of the asset.

        This attribute is settable. See the Examples below.

        Returns
        -------
        initial_price : Real
            The initial price of the asset.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel

        Define the parameters of a binomial pricing model.

        >>> S_0 = 100
        >>> r = 0.05
        >>> p = 0.55
        >>> u = 1.1
        >>> d = 0.9
        >>> T = 2

        Exhaustively generate all price trajectories and print them.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     up_prob=p,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     mode="enum",
        ...     length=T,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  110.0  121.0
        1      100.0  110.0   99.0
        2      100.0   90.0   99.0
        3      100.0   90.0   81.0

        Reset the initial price and regenerate new price trajectories.

        >>> S.initial_price = 120
        >>> S.regenerate()
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      120.0  132.0  145.2
        1      120.0  132.0  118.8
        2      120.0  108.0  118.8
        3      120.0  108.0   97.2
        """
        return self._initial_price

    @initial_price.setter
    def initial_price(self, value: Real) -> None:
        """Set the initial price of the asset."""
        if not isinstance(value, Real) or value <= 0:
            raise TypeError("initial_price must be a positive real number.")

        self._initial_price = value

    @property
    def risk_free_rate(self) -> Real | None:
        """Get the per-period risk-free rate.

        This attribute is settable. See the Examples below.

        Returns
        -------
        risk_free_rate : Real
            The risk-free rate.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Define the parameters of a binomial pricing model.

        >>> S_0 = 100
        >>> r = 0.05
        >>> p = 0.55
        >>> u = 1.1
        >>> d = 0.9
        >>> T = 2

        Exhaustively generate all price trajectories and print them.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     up_prob=p,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     mode="enum",
        ...     length=T,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  110.0  121.0
        1      100.0  110.0   99.0
        2      100.0   90.0   99.0
        3      100.0   90.0   81.0

        Define a European call option.

        >>> Phi = Claim.european_option(strike=100, model=S)
        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        European option 'Phi':
        t        0    1     2
        omega
        0      0.0  0.0  21.0
        1      0.0  0.0   0.0
        2      0.0  0.0   0.0
        3      0.0  0.0   0.0

        Compute the risk-neutral price of the option with respect to the risk-free rate given above.

        >>> print(Phi.price())
        10.714285714285719

        Reset the risk-free rate, regenerate trajectories, and compute the new risk-neutral price.

        >>> S.risk_free_rate = 0.025
        >>> S.regenerate()
        >>> print(Phi.price())
        7.807852468768577
        """
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, value: Real) -> None:
        """Set the per-period risk-free rate."""
        if not isinstance(value, Real) or value <= 0:
            raise TypeError("risk_free_rate must be a positive real number.")
        self._risk_free_rate = value

    @property
    def up_prob(self) -> Real | None:
        """Get the probability of an upward price movement.

        This attribute is settable. See the Examples below.

        Returns
        -------
        up_prob : Real
            The probability of an upward price movement.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel

        Define the parameters of a binomial pricing model.

        >>> S_0 = 100
        >>> r = 0.05
        >>> p = 0.55
        >>> u = 1.1
        >>> d = 0.9
        >>> T = 2

        Exhaustively generate all price trajectories and print them.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     up_prob=p,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     mode="enum",
        ...     length=T,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  110.0  121.0
        1      100.0  110.0   99.0
        2      100.0   90.0   99.0
        3      100.0   90.0   81.0

        Print the real-world probability measure of the model.

        >>> print(S.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                    P
        omega
        0      0.3025
        1      0.2475
        2      0.2475
        3      0.2025

        Reset the up-probability, regenerate new price trajectories, and print the new real-world measure.

        >>> S.up_prob = 0.65
        >>> S.regenerate()
        >>> print(S.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                    P
        omega
        0      0.4225
        1      0.2275
        2      0.2275
        3      0.1225
        """
        return self._up_prob

    @up_prob.setter
    def up_prob(self, value: Real) -> None:
        """Set the probability of an upward price movement."""
        if not isinstance(value, Real) or not (0 < value < 1):
            raise TypeError("up_prob must be a probability in the interval (0, 1).")
        self._up_prob = value

    @property
    def middle_prob(self) -> Real | None:
        """Get the probability of a middle price movement.

        Returns
        -------
        down_prob : Real
            The probability of a middle price movement.
        """
        return 1 - self.up_prob - self.down_prob

    @property
    def down_prob(self) -> Real | None:
        """Get the probability of a downward price movement.

        Returns
        -------
        down_prob : Real
            The probability of a downward price movement.
        """
        return self._down_prob

    @property
    def up_factor(self) -> Real | None:
        """Get the up-factor of the model.

        This attribute is settable. See the Examples below.

        Returns
        -------
        up_factor : Real
            The up-factor of the model.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel

        Define the parameters of a binomial pricing model.

        >>> S_0 = 100
        >>> r = 0.05
        >>> p = 0.55
        >>> u = 1.1
        >>> d = 0.9
        >>> T = 2

        Exhaustively generate all price trajectories and print them.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     up_prob=p,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     mode="enum",
        ...     length=T,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  110.0  121.0
        1      100.0  110.0   99.0
        2      100.0   90.0   99.0
        3      100.0   90.0   81.0

        Reset the up-factor and regenerate new price trajectories.

        >>> S.up_factor = 1.2
        >>> S.regenerate()
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  120.0  144.0
        1      100.0  120.0  108.0
        2      100.0   90.0  108.0
        3      100.0   90.0   81.0
        """
        return self._up_factor

    @up_factor.setter
    def up_factor(self, value: Real) -> None:
        """Set the up-factor of the model."""
        if not isinstance(value, Real) or value <= 1:
            raise TypeError("up_factor must be a real number greater than 1.")
        self._up_factor = value

    @property
    def middle_factor(self) -> Real | None:
        """Pass."""
        return self._middle_factor

    @property
    def down_factor(self) -> Real | None:
        """Get the down-factor of the model.

        This attribute is settable. See the Examples below.

        Returns
        -------
        down_factor : Real
            The down-factor of the model.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel

        Define the parameters of a binomial pricing model.

        >>> S_0 = 100
        >>> r = 0.05
        >>> p = 0.55
        >>> u = 1.1
        >>> d = 0.9
        >>> T = 2

        Exhaustively generate all price trajectories and print them.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     up_prob=p,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     mode="enum",
        ...     length=T,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  110.0  121.0
        1      100.0  110.0   99.0
        2      100.0   90.0   99.0
        3      100.0   90.0   81.0

        Reset the down-factor and regenerate new price trajectories.

        >>> S.down_factor = 0.7
        >>> S.regenerate()
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  110.0  121.0
        1      100.0  110.0   77.0
        2      100.0   70.0   77.0
        3      100.0   70.0   49.0
        """
        return self._down_factor

    @down_factor.setter
    def down_factor(self, value: Real) -> None:
        """Set the down-factor of the model."""
        if not isinstance(value, Real) or not (0 < value < 1):
            raise TypeError("down_factor must be a real number between 0 and 1.")
        self._down_factor = value

    @property
    def enum_mode(self) -> Literal["dense", "sparse"] | None:
        """Get the enumeration mode of the model.

        This attribute is settable. See the Examples below.

        Returns
        -------
        enum_mode : Literal["dense", "sparse"]
            The enumeration mode of the model.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel

        Define the parameters of a binomial pricing model.

        >>> S_0 = 100
        >>> r = 0.05
        >>> p = 0.55
        >>> u = 1.1
        >>> d = 0.9
        >>> T = 3

        Exhaustively generate all price trajectories in `dense` enumeration mode and print them.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     up_prob=p,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     enum_mode="dense",
        ...     mode="enum",
        ...     length=T,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2      3
        omega
        0      100.0  110.0  121.0  133.1
        1      100.0  110.0  121.0  108.9
        2      100.0  110.0   99.0  108.9
        3      100.0  110.0   99.0   89.1
        4      100.0   90.0   99.0  108.9
        5      100.0   90.0   99.0   89.1
        6      100.0   90.0   81.0   89.1
        7      100.0   90.0   81.0   72.9

        Reset the enumeration mode, regenerate the trajectories, and print them.

        >>> S.enum_mode = "sparse"
        >>> S.regenerate()
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2      3
        omega
        0      100.0  110.0  121.0  133.1
        1      100.0   90.0   99.0  108.9
        2      100.0   90.0   81.0   89.1
        3      100.0   90.0   81.0   72.9
        """
        return self._enum_mode

    @enum_mode.setter
    def enum_mode(self, value: Literal["dense", "sparse"]) -> None:
        """Set the enumeration mode of the model."""
        if value not in ["dense", "sparse"]:
            raise ValueError("The enumeration mode must be either 'dense' or 'sparse'.")
        self._enum_mode = value

    @property
    def risk_free_gross_return(self) -> Real | None:
        """Get the per-period risk-free gross return of the model.

        This attribute is settable. See the Examples below.

        Returns
        -------
        risk_free_gross_return : Real
            The per-period risk-free gross return.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Define the parameters of a binomial pricing model.

        >>> S_0 = 100
        >>> r = 0.05
        >>> p = 0.55
        >>> u = 1.1
        >>> d = 0.9
        >>> T = 2

        Exhaustively generate all price trajectories and print them.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=S_0,
        ...     risk_free_rate=r,
        ...     up_prob=p,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     mode="enum",
        ...     length=T,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0      1      2
        omega
        0      100.0  110.0  121.0
        1      100.0  110.0   99.0
        2      100.0   90.0   99.0
        3      100.0   90.0   81.0

        Define a European call option.

        >>> Phi = Claim.european_option(strike=100, model=S)
        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        European option 'Phi':
        t        0    1     2
        omega
        0      0.0  0.0  21.0
        1      0.0  0.0   0.0
        2      0.0  0.0   0.0
        3      0.0  0.0   0.0

        Compute the risk-neutral price of the option with respect to the risk-free rate given above.

        >>> print(Phi.price())
        10.714285714285719

        Reset the risk-free gross return, regenerate trajectories, and compute the new risk-neutral price.

        >>> S.risk_free_gross_return = 1.025
        >>> S.regenerate()
        >>> print(Phi.price())
        7.807852468768577
        """
        return 1 + self.risk_free_rate

    @risk_free_gross_return.setter
    def risk_free_gross_return(self, value: Real) -> None:
        """Set the per-period risk-free gross return of the model."""
        if not isinstance(value, Real) or value <= 1:
            raise ValueError(
                "The risk-free gross return must be a real number greater than 1."
            )
        self._risk_free_rate = value - 1

    # --------------------- finance methods --------------------- #

    def risk_neutral_probs(self, theta: Real) -> tuple[Real, Real, Real]:
        r"""Get the risk-neutral probabilities of the model.

        Parameters
        ----------
        theta : Real
            A real number parameterizing the risk-neutral probabilities.

        Returns
        -------
        risk_neutral_probs : tuple[Real, Real, Real]
            The risk neutral probabilities as a tuple `(q_u, q_m, q_d)`, where `q_u` is the risk-neutral probability of an up move, `q_m` is the risk-neutral probability of a middle move, and `q_d` is the risk-neutral probability of a down move.

        Examples
        --------
        >>> from sigalg.finance import TrinomialPricingModel
        >>> S_0 = 4
        >>> u = 1.2
        >>> m = 1.1
        >>> d = 0.9
        >>> p_u = 0.6
        >>> p_d = 0.1
        >>> r = 0.01
        >>> S = TrinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     middle_factor=m,
        ...     down_factor=d,
        ...     up_prob=p_u,
        ...     down_prob=p_d,
        ...     risk_free_rate=r,
        ...     length=2,
        ... )
        >>> print(S) # doctest: +NORMALIZE_WHITESPACE
        Trinomial price process 'S':
        t        0    1     2
        omega
        0      4.0  4.8  5.76
        1      4.0  4.8  5.28
        2      4.0  4.8  4.32
        3      4.0  4.4  5.28
        4      4.0  4.4  4.84
        5      4.0  4.4  3.96
        6      4.0  3.6  4.32
        7      4.0  3.6  3.96
        8      4.0  3.6  3.24
        >>> risk_neutral_probs = S.risk_neutral_probs(theta=0.42)
        >>> print(risk_neutral_probs)
        (0.15400000000000003, 0.31899999999999973, 0.5270000000000001)
        """
        if not isinstance(theta, Real):
            raise TypeError("The parameter theta must be a real number.")
        if theta <= 0 or theta >= 1:
            raise ValueError("The parameter theta must be in the open interval (0, 1).")

        R = self.risk_free_gross_return
        u = self.up_factor
        m = self.middle_factor
        d = self.down_factor

        if None in [R, u, m, d]:
            raise ValueError(
                "One of the parameters needed to generate the risk-neutral probabilities is None. Be sure to call 'generate' first."
            )
        if R <= d or R >= u:
            raise ValueError(
                "There is arbitrage in the model. The risk-free gross return R must be in the interval [down_factor, up_factor]."
            )

        a = max((m - R) / (m - d), 0)
        b = (u - R) / (u - d)
        q_d = (b - a) * (theta - 1) + b
        q_u = ((m - d) * q_d + (R - m)) / (u - m)
        q_m = ((d - u) * q_d + (u - R)) / (u - m)

        return q_u, q_m, q_d

    def emms(self) -> ParametrizedProbabilityMeasure:
        r"""Return the equivalent martingale measures of the model.

        Raises
        ------
        ValueError
            If one of the parameters used to generate the EMMs is none. This likely means that the `generate` method was not called first.

        Returns
        -------
        EMMs : ParametrizedProbabilityMeasure
            The equivalent martingale measures of the model.

        Examples
        --------
        >>> from sigalg.finance import TrinomialPricingModel
        >>> S_0 = 4
        >>> u = 1.2
        >>> m = 1.1
        >>> d = 0.9
        >>> p_u = 0.6
        >>> p_d = 0.1
        >>> r = 0.01
        >>> S = TrinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     middle_factor=m,
        ...     down_factor=d,
        ...     up_prob=p_u,
        ...     down_prob=p_d,
        ...     risk_free_rate=r,
        ...     length=2,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Trinomial price process 'S':
        time    0    1     2
        sample
        0       4  4.8  5.76
        1       4  4.8  5.28
        2       4  4.8  4.32
        3       4  4.4  5.28
        4       4  4.4  4.84
        5       4  4.4  3.96
        6       4  3.6  4.32
        7       4  3.6  3.96
        8       4  3.6  3.24
        >>> Q = S.EMMs
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q(theta, sample)'
        >>> print(Q(theta=0.5))  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q(theta=0.5)':
                probability
        sample
        0          0.033611
        1          0.050417
        2          0.099306
        3          0.050417
        4          0.075625
        5          0.148958
        6          0.099306
        7          0.148958
        8          0.293403
        >>> print(Q(theta=0.5, sample=4))
        0.07562500000000004
        >>> is_martingale_wrt_real_world_measure = S.discount(r).is_martingale()
        >>> print(is_martingale_wrt_real_world_measure)
        False
        >>> is_martingale_wrt_EMM = S.discount(r).is_martingale(prob_measure=Q(theta=0.5))
        >>> print(is_martingale_wrt_EMM)
        True
        """
        from scipy.stats import multinomial

        from ...core.measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ...processes.types.iid_process import IIDProcess

        if None in [
            self.risk_free_gross_return,
            self.up_factor,
            self.middle_factor,
            self.down_factor,
            self.time,
        ]:
            raise ValueError(
                "One of the parameters needed to generate the EMMs is None. Be sure to call 'generate' first."
            )

        if self._emms is None:
            T = self.time[1:]

            def mapping(*, theta, sample):
                q_u, q_m, q_d = self.risk_neutral_probs(theta=theta)

                prob_measure = IIDProcess.generate(
                    mode="enum",
                    distribution=multinomial(1, [q_u, q_m, q_d]),
                    support=[0, 1, 2],
                    index=T,
                ).prob_measure

                return prob_measure(sample)

            self._emms = ParametrizedProbabilityMeasure(
                domain=self.sample_space, mapping=mapping, name="Q"
            )

        return self._emms

    def price(self, claim):
        """Pass."""
        return super().price(claim)

    def price_process(self, claim):
        """Pass."""
        return super().price_process(claim)

    def delta(self, claim):
        """Pass."""
        return super().delta(claim)

    def replicating_portfolio(self, claim):
        """Pass."""
        return super().replicating_portfolio(claim)

    # --------------------- enumeration methods --------------------- #

    def _enumeration_subclass_hook(self) -> pd.DataFrame:
        """Hook for enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """  # noqa: D401
        time = self._index[1:].data
        initial_time = self._index[0]

        trajectories = pd.DataFrame(
            list(
                product(
                    [self.up_factor, self.middle_factor, self.down_factor],
                    repeat=len(self._index) - 1,
                )
            ),
            columns=time,
        )
        trajectories.insert(loc=0, column=initial_time, value=1)
        trajectories = self.initial_price * trajectories.cumprod(axis=1)

        return trajectories

    def _generate_exact_prob_measure(self, domain: Domain) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated trinomial pricing process.

        Parameters
        ----------
        domain : Domain
            The domain of the underlying probability space.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.
        """
        from ...core.measures.probability_measure import ProbabilityMeasure

        num_steps = len(self._index) - 1
        p_u = self.up_prob
        p_m = self.middle_prob
        p_d = self.down_prob

        probabilities = pd.DataFrame(
            list(product([p_u, p_m, p_d], repeat=num_steps)), index=domain.data
        ).product(axis=1)

        probabilities /= probabilities.sum()

        return ProbabilityMeasure(domain=domain, mapping=probabilities)

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        u = self.up_factor
        m = self.middle_factor
        d = self.down_factor
        p_u = self.up_prob
        p_m = self.middle_prob
        p_d = self.down_prob
        n_trajectories = self._n_trajectories
        time = self._index[1:].data
        initial_time = self._index[0]

        trajectories = pd.DataFrame(
            self._random_state.choice(
                [u, m, d],
                size=(n_trajectories, len(time)),
                p=[p_u, p_m, p_d],
            ),
            columns=time,
        )
        trajectories.insert(loc=0, column=initial_time, value=1)
        trajectories = self.initial_price * trajectories.cumprod(axis=1)

        return trajectories
