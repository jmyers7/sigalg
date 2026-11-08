"""A class modeling a binomial pricing model."""

from __future__ import annotations

from itertools import product
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .discrete_pricing_model import DiscretePricingModel

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...core.measures.probability_measure import ProbabilityMeasure
    from ...core.spaces.domain import Domain
    from ...processes.base.stochastic_process import StochasticProcess
    from ...processes.stopping_times.stopping_time import StoppingTime
    from ...typing.index_like import IndexLike
    from ..claims.claim import Claim


class BinomialPricingModel(DiscretePricingModel):
    r"""A class modeling a binomial pricing model.

    The base constructor is not meant to be called directly by users. Instead, the user should call the `generate` class method. See the Examples section below for usage.

    Examples
    --------
    >>> from sigalg.finance import BinomialPricingModel

    Given parameters of a binomial model, enumerate all length-3 price trajectories in `dense` enumeration mode.

    >>> S = BinomialPricingModel.generate(
    ...     initial_price=100,
    ...     risk_free_rate=0.01,
    ...     up_factor=1.1,
    ...     up_prob=0.7,
    ...     length=3,
    ...     enum_mode="dense",
    ...     mode="enum",
    ... )
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Binomial price process 'S':
    t          0           1           2           3
    omega
    0      100.0  110.000000  121.000000  133.100000
    1      100.0  110.000000  121.000000  110.000000
    2      100.0  110.000000  100.000000  110.000000
    3      100.0  110.000000  100.000000   90.909091
    4      100.0   90.909091  100.000000  110.000000
    5      100.0   90.909091  100.000000   90.909091
    6      100.0   90.909091   82.644628   90.909091
    7      100.0   90.909091   82.644628   75.131480

    Enumerate all length-3 price trajectories in `sparse` enumeration mode.

    >>> S.enum_mode = "sparse"
    >>> S.regenerate()
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Binomial price process 'S':
    t          0           1           2           3
    omega
    0      100.0  110.000000  121.000000  133.100000
    1      100.0   90.909091  100.000000  110.000000
    2      100.0   90.909091   82.644628   90.909091
    3      100.0   90.909091   82.644628   75.131480

    Simulate ten length-3 trajectories in `sim` mode.

    >>> S.n_trajectories = 10
    >>> S.random_state = 42
    >>> S.mode = "sim"
    >>> S.regenerate()
    >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
    Binomial price process 'S':
    t          0           1           2           3
    omega
    0      100.0   90.909091  100.000000   90.909091
    1      100.0  110.000000  121.000000  110.000000
    2      100.0   90.909091   82.644628   90.909091
    3      100.0  110.000000  121.000000  110.000000
    4      100.0  110.000000  100.000000  110.000000
    5      100.0  110.000000  121.000000  133.100000
    6      100.0   90.909091  100.000000   90.909091
    7      100.0  110.000000  100.000000   90.909091
    8      100.0   90.909091  100.000000  110.000000
    9      100.0  110.000000  121.000000  133.100000
    """

    _repr_name = "BinomialPricingModel"
    _str_name = "Binomial price process"

    # --------------------- constructors --------------------- #

    @DiscretePricingModel.generator
    def generate(
        cls,
        initial_price: Real,
        up_prob: Real,
        up_factor: Real,
        down_factor: Real | None = None,
        risk_free_rate: Real | None = None,
        risk_free_gross_return: Real | None = None,
        enum_mode: Literal["dense", "sparse"] = "dense",
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        domain_name: Hashable | None = None,
        index: IndexLike | None = None,
        length: int | None = None,
        name: Hashable = "S",
        random_state: int | np.random.Generator | None = None,
    ) -> BinomialPricingModel:
        r"""Generate trajectories of the binomial pricing model by either exhaustive enumeration (in sparse or dense mode) or Monte Carlo simulation.

        Parameters
        ----------
        initial_price : Real
            The initial price of the asset.
        up_prob : Real
            The probability of an upward move in the price.
        up_factor : Real
            The up-factor of the model.
        down_factor : Real | None, default=None
            The down-factor of the model. If `None`, then it will be set to the reciprocal of the up-factor.
        risk_free_rate : Real | None, default=None
            The per-period risk-free rate. If `None`, the risk-free gross return must be given.
        risk_free_gross_return : Real | None, default=None
            The per-period risk-free gross return. If `None`, the risk-free rate must be given.
        enum_mode : Literal["dense", "sparse"], default="dense"
            The mode of enumeration. If the generation mode is set to `sim`, this parameter is ignore.
        mode : Literal["enum", "sim"]
            Whether to generate trajectories by exhuastive enumeration or Monte Carlo simulation.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. If the generation mode is set to `enum`, this parameter is ignored.
        index : IndexLike | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        name : Hashable, default="S"
            The name of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional random state for reproducibility.

        Returns
        -------
        info : dict[str, object]
            A dictionary containing the names of the BinomialPricingModel-specific parameters and their values.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel

        Given parameters of a binomial model, enumerate all length-3 price trajectories in `dense` enumeration mode.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=100,
        ...     risk_free_rate=0.01,
        ...     up_factor=1.1,
        ...     up_prob=0.7,
        ...     length=3,
        ...     enum_mode="dense",
        ...     mode="enum",
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0           1           2           3
        omega
        0      100.0  110.000000  121.000000  133.100000
        1      100.0  110.000000  121.000000  110.000000
        2      100.0  110.000000  100.000000  110.000000
        3      100.0  110.000000  100.000000   90.909091
        4      100.0   90.909091  100.000000  110.000000
        5      100.0   90.909091  100.000000   90.909091
        6      100.0   90.909091   82.644628   90.909091
        7      100.0   90.909091   82.644628   75.131480

        Enumerate all length-3 price trajectories in `sparse` enumeration mode.

        >>> S.enum_mode = "sparse"
        >>> S.regenerate()
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0           1           2           3
        omega
        0      100.0  110.000000  121.000000  133.100000
        1      100.0   90.909091  100.000000  110.000000
        2      100.0   90.909091   82.644628   90.909091
        3      100.0   90.909091   82.644628   75.131480

        Simulate ten length-3 trajectories in `sim` mode.

        >>> S.n_trajectories = 10
        >>> S.random_state = 42
        >>> S.mode = "sim"
        >>> S.regenerate()
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        t          0           1           2           3
        omega
        0      100.0   90.909091  100.000000   90.909091
        1      100.0  110.000000  121.000000  110.000000
        2      100.0   90.909091   82.644628   90.909091
        3      100.0  110.000000  121.000000  110.000000
        4      100.0  110.000000  100.000000  110.000000
        5      100.0  110.000000  121.000000  133.100000
        6      100.0   90.909091  100.000000   90.909091
        7      100.0  110.000000  100.000000   90.909091
        8      100.0   90.909091  100.000000  110.000000
        9      100.0  110.000000  121.000000  133.100000

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
        if not isinstance(up_factor, Real) or up_factor <= 1:
            raise TypeError("up_factor must be a real number greater than 1.")
        if down_factor is not None:
            if not isinstance(down_factor, Real) or not (0 < down_factor < 1):
                raise TypeError(
                    "down_factor must be a real number in the interval (0, 1)."
                )
        if not isinstance(enum_mode, str) or enum_mode not in {"sparse", "dense"}:
            raise ValueError("enum_mode must be either 'sparse' or 'dense'.")

        if down_factor is None:
            down_factor = 1 / up_factor
        if risk_free_rate is None:
            risk_free_rate = risk_free_gross_return - 1

        return {
            "initial_price": initial_price,
            "risk_free_rate": risk_free_rate,
            "up_prob": up_prob,
            "up_factor": up_factor,
            "down_factor": down_factor,
            "enum_mode": enum_mode,
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
    def down_prob(self) -> Real | None:
        """Get the probability of a downward price movement.

        Returns
        -------
        down_prob : Real
            The probability of a downward price movement.
        """
        return 1 - self.up_prob

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

    def risk_neutral_probs(self) -> tuple[Real, Real]:
        """Get the risk-neutral probabilities of the model.

        Returns
        -------
        risk_neutral_probs : tuple[Real, Real]
            The risk neutral probabilities as a tuple `(q_u, q_d)`, where `q_u` is the risk-neutral probability of an up move, and `q_d` is the risk-neutral probability of a down move.

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

        Get the risk-neutral probabilities of the model.

        >>> S.risk_neutral_probs()
        (0.7499999999999999, 0.2500000000000001)
        """
        R, u, d = self.risk_free_gross_return, self.up_factor, self.down_factor

        if None in [R, u, d]:
            raise ValueError(
                "One of the parameters needed to generate the risk-neutral probabilities is None. Be sure to call 'generate' first."
            )
        if R <= d or R >= u:
            raise ValueError(
                "There is arbitrage in the model. The risk-free gross return R must be in the interval [down_factor, up_factor]."
            )

        q_u = (R - d) / (u - d)
        q_d = 1 - q_u

        return q_u, q_d

    def emms(self) -> ProbabilityMeasure:
        """Get the equivalent martingale measure of the model.

        Returns
        -------
        emm : ProbabilityMeasure
            The equivalent martingale measure of the model.

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

        Print the equivalent martingale measure.

        >>> print(S.emms())  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                    Q
        omega
        0      0.5625
        1      0.1875
        2      0.1875
        3      0.0625
        """
        from ...core.measures.probability_measure import ProbabilityMeasure

        num_steps = len(self._index) - 1
        domain = self.domain
        q = self.risk_neutral_probs()[0]

        if self.enum_mode == "sparse":
            # self._emms = self._generate_prob_measure(
            #     prob=self.risk_neutral_probs()[0], name="Q"
            # )
            pass

        elif self.enum_mode == "dense":
            probabilities = pd.DataFrame(
                list(product([q, 1 - q], repeat=num_steps)), index=domain.data
            ).product(axis=1)

            probabilities /= probabilities.sum()

            return ProbabilityMeasure(
                domain=domain,
                mapping=probabilities,
                name="Q",
            )

    def price(self, claim: Claim) -> Real:
        """Pass."""
        if claim.model is not self and not claim.is_adapted(self.natural_filtration):
            raise ValueError(
                "The claim must be adapated to the natural filtration of the binomial model."
            )

        if not claim.is_early_exercise:
            T = self.time[-1]
            R = self.risk_free_gross_return
            Phi_T = claim.last_rv
            Q = self.emms()

            return Phi_T.expectation(measure=Q).item() / R**T

        else:
            raise NotImplementedError(
                "Have not yet implemented pricing with an early-exercise claim and a binomial model."
            )

    def price_process(self, claim: Claim) -> StochasticProcess:
        """Pass."""
        if claim.model is not self and not claim.is_adapted(self.natural_filtration):
            raise ValueError(
                "The claim must be adapated to the natural filtration of the binomial model."
            )

        if not claim.is_early_exercise:
            _, _, V, _ = self.replicating_portfolio(claim=claim)

            return V.with_name("X")

        else:
            raise NotImplementedError()

    def delta(self, claim: Claim) -> StochasticProcess:
        """Pass."""
        if claim.model is not self and not claim.is_adapted(self.natural_filtration):
            raise ValueError(
                "The claim must be adapated to the natural filtration of the binomial model."
            )

        if not claim.is_early_exercise:
            _, Delta, _, _ = self.replicating_portfolio(claim=claim)

            return Delta

        else:
            raise NotImplementedError(
                "Have not yet implemented delta processes with an early-exercise claim and a binomial model."
            )

    def replicating_portfolio(
        self, claim: Claim
    ) -> tuple[
        StochasticProcess,
        StochasticProcess,
        StochasticProcess,
        StochasticProcess,
        StoppingTime,
    ]:
        """Compute the replicating portfolio for a given contingent claim.

        Parameters
        ----------
        claim : Claim
            The contingent claim for which to compute the replicating portfolio.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Define a binomial pricing model.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=100,
        ...     up_factor=1.1,
        ...     down_factor=0.9,
        ...     up_prob=0.7,
        ...     risk_free_rate=0.05,
        ...     mode="enum",
        ...     length=3,
        ... )
        >>> print(S) # doctest: +NORMALIZE_WHITESPACE
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

        Define a European call option with strike `K=100`.

        >>> K = 100
        >>> euro_call = Claim.european_option(model=S, strike=K, type="call", name="euro_call")
        >>> print(euro_call)  # doctest: +NORMALIZE_WHITESPACE
        European option 'euro_call':
        t        0    1    2     3
        omega
        0      0.0  0.0  0.0  33.1
        1      0.0  0.0  0.0   8.9
        2      0.0  0.0  0.0   8.9
        3      0.0  0.0  0.0   0.0
        4      0.0  0.0  0.0   8.9
        5      0.0  0.0  0.0   0.0
        6      0.0  0.0  0.0   0.0
        7      0.0  0.0  0.0   0.0

        Generate the replicating portfolio for the European call.

        >>> X, Delta, B, V, tau = euro_call.replicating_portfolio()

        Print the price process for the option. Note the initial price.

        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        t              0          1          2     3
        omega
        0      15.306122  19.914966  25.761905  33.1
        1      15.306122  19.914966  25.761905   8.9
        2      15.306122  19.914966   6.357143   8.9
        3      15.306122  19.914966   6.357143   0.0
        4      15.306122   4.540816   6.357143   8.9
        5      15.306122   4.540816   6.357143   0.0
        6      15.306122   4.540816   0.000000   0.0
        7      15.306122   4.540816   0.000000   0.0

        Print the `Delta` and `B` price processes of the hedging portfolio, along with its value process `V`.

        >>> print(Delta) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Delta':
        t             0         1         2
        omega
        0      0.768707  0.882035  1.000000
        1      0.768707  0.882035  1.000000
        2      0.768707  0.882035  0.449495
        3      0.768707  0.882035  0.449495
        4      0.768707  0.353175  0.449495
        5      0.768707  0.353175  0.449495
        6      0.768707  0.353175 -0.000000
        7      0.768707  0.353175 -0.000000
        >>> print(B)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'B':
        t              0          1          2
        omega
        0     -61.564626 -77.108844 -95.238095
        1     -61.564626 -77.108844 -95.238095
        2     -61.564626 -77.108844 -38.142857
        3     -61.564626 -77.108844 -38.142857
        4     -61.564626 -27.244898 -38.142857
        5     -61.564626 -27.244898 -38.142857
        6     -61.564626 -27.244898   0.000000
        7     -61.564626 -27.244898   0.000000
        >>> print(V) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'V':
        t              0          1          2     3
        omega
        0      15.306122  19.914966  25.761905  33.1
        1      15.306122  19.914966  25.761905   8.9
        2      15.306122  19.914966   6.357143   8.9
        3      15.306122  19.914966   6.357143   0.0
        4      15.306122   4.540816   6.357143   8.9
        5      15.306122   4.540816   6.357143   0.0
        6      15.306122   4.540816   0.000000   0.0
        7      15.306122   4.540816   0.000000   0.0

        Print the exercise time `tau`. Since a European option does not allow early exercise, the option will either be exercised at maturity (if it is in the money) or never.

        >>> print(tau)  # doctest: +NORMALIZE_WHITESPACE
        Stopping time 'tau':
               tau
        omega
        0      3.0
        1      3.0
        2      3.0
        3      inf
        4      3.0
        5      inf
        6      inf
        7      inf

        Now define an Asian put option with early-exercise and the same strike.

        >>> asian_put = Claim.asian_option(strike=100, model=S, type="put", early_exercise=True, name="asian_put")
        >>> print(asian_put)  # doctest: +NORMALIZE_WHITESPACE
        Asian option 'asian_put':
        t        0    1         2       3
        omega
        0      0.0  0.0  0.000000   0.000
        1      0.0  0.0  0.000000   0.000
        2      0.0  0.0  0.000000   0.000
        3      0.0  0.0  0.000000   0.475
        4      0.0  5.0  3.666667   0.525
        5      0.0  5.0  3.666667   5.475
        6      0.0  5.0  9.666667   9.975
        7      0.0  5.0  9.666667  14.025

        Compute and print the replicating portfolio.

        >>> X, Delta, B, V, tau = asian_put.replicating_portfolio()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        t            0         1          2       3
        omega
        0      1.23603  0.026927   0.000000   0.000
        1      1.23603  0.026927   0.000000   0.000
        2      1.23603  0.026927   0.113095   0.000
        3      1.23603  0.026927   0.113095   0.475
        4      1.23603  5.110544   3.666667   0.525
        5      1.23603  5.110544   3.666667   5.475
        6      1.23603  5.110544  10.464286   9.975
        7      1.23603  5.110544  10.464286  14.025
        >>> print(Delta)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Delta':
        t             0         1        2
        omega
        0     -0.254181 -0.005141 -0.00000
        1     -0.254181 -0.005141 -0.00000
        2     -0.254181 -0.005141 -0.02399
        3     -0.254181 -0.005141 -0.02399
        4     -0.254181 -0.377646 -0.25000
        5     -0.254181 -0.377646 -0.25000
        6     -0.254181 -0.377646 -0.25000
        7     -0.254181 -0.377646 -0.25000
        >>> print(B)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'B':
        t              0          1          2
        omega
        0      26.654114   0.592404   0.000000
        1      26.654114   0.592404   0.000000
        2      26.654114   0.592404   2.488095
        3      26.654114   0.592404   2.488095
        4      26.654114  39.098639  26.428571
        5      26.654114  39.098639  26.428571
        6      26.654114  39.098639  30.714286
        7      26.654114  39.098639  30.714286
        >>> print(V)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'V':
        t            0         1          2       3
        omega
        0      1.23603  0.026927   0.000000   0.000
        1      1.23603  0.026927   0.000000   0.000
        2      1.23603  0.026927   0.113095   0.000
        3      1.23603  0.026927   0.113095   0.475
        4      1.23603  5.110544   1.678571   0.525
        5      1.23603  5.110544   1.678571   5.475
        6      1.23603  5.110544  10.464286   9.975
        7      1.23603  5.110544  10.464286  14.025
        >>> print(tau)  # doctest: +NORMALIZE_WHITESPACE
        Stopping time 'tau':
               tau
        omega
        0      inf
        1      inf
        2      inf
        3      3.0
        4      2.0
        5      2.0
        6      3.0
        7      3.0
        """
        T = self.time[-1]

        if claim.model is not self and not claim.is_adapted(self.natural_filtration):
            raise ValueError(
                "The claim must be adapated to the natural filtration of the binomial model."
            )

        B_arr, Delta_arr, V_arr, X_arr, tau_arr = self._initialize_replicating_arrays()

        Phi_arr = claim.to_numpy()
        V_arr[:, -1] = claim.last_rv
        X_arr[:, -1] = claim.last_rv
        tau_arr[:, -1] = np.where(X_arr[:, -1] == 0, 0, 1)

        for t in reversed(range(T)):
            V_forward, X_forward, S_forward, S_curr, Phi_curr = (
                self._extract_tree_nodes(
                    t,
                    V_arr,
                    X_arr,
                    Phi_arr,
                )
            )

            B_curr, Delta_curr, V_curr, X_curr, tau_curr = self._backward_induction(
                enum_mode=self.enum_mode,
                V_forward=V_forward,
                X_forward=X_forward,
                S_forward=S_forward,
                S_curr=S_curr,
                Phi_curr=Phi_curr,
            )

            B_arr[:, t], Delta_arr[:, t], V_arr[:, t], X_arr[:, t], tau_arr[:, t] = (
                self._broadcast_node_values(
                    t=t,
                    B_curr=B_curr,
                    Delta_curr=Delta_curr,
                    V_curr=V_curr,
                    X_curr=X_curr,
                    tau_curr=tau_curr,
                )
            )

        tau_arr = np.where(
            tau_arr.max(axis=1) == 0,
            np.inf,
            np.argmax(tau_arr, axis=1),
        )

        B, Delta, V, X, tau = self._convert_replicating_arrays_to_processes(
            B_arr=B_arr,
            Delta_arr=Delta_arr,
            V_arr=V_arr,
            X_arr=X_arr,
            tau_arr=tau_arr,
        )

        return X, Delta, B, V, tau

    def _initialize_replicating_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]

        if self.enum_mode == "dense":
            B_arr = np.zeros(shape=(2**T, T))
            Delta_arr = np.zeros(shape=(2**T, T))
            V_arr = np.zeros(shape=(2**T, T + 1))
            X_arr = np.zeros(shape=(2**T, T + 1))
            tau_arr = np.zeros(shape=(2**T, T + 1))

        elif self.enum_mode == "sparse":
            B_arr = np.zeros(shape=(T + 1, T))
            Delta_arr = np.zeros(shape=(T + 1, T))
            V_arr = np.zeros(shape=(T + 1, T + 1))
            X_arr = np.zeros(shape=(T + 1, T + 1))
            tau_arr = np.zeros(shape=(T + 1, T + 1))

        return B_arr, Delta_arr, V_arr, X_arr, tau_arr

    def _extract_tree_nodes(
        self,
        t: int,
        V_arr: np.ndarray,
        X_arr: np.ndarray,
        Phi_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]
        S_arr = self.data.values

        if self.enum_mode == "dense":
            V_forward = V_arr[:: (2 ** (T - t - 1)), t + 1]  # shape (2^(t+1),)
            X_forward = X_arr[:: (2 ** (T - t - 1)), t + 1]  # shape (2^(t+1),)
            S_forward = S_arr[:: (2 ** (T - t - 1)), t + 1]  # shape (2^(t+1),)
            S_curr = S_arr[:: (2 ** (T - t)), t]  # shape (2^t,)
            Phi_curr = Phi_arr[:: (2 ** (T - t)), t]  # shape (2^t,)

        elif self.enum_mode == "sparse":
            V_forward = V_arr[: t + 2, t + 1]  # shape (t+2,)
            X_forward = X_arr[: t + 2, t + 1]  # shape (t+2,)
            S_forward = self.data.values[: t + 2, t + 1]  # shape(t+2,)
            S_curr = self.data.values[: t + 1, t]  # shape (t+1,)

        return V_forward, X_forward, S_forward, S_curr, Phi_curr

    def _backward_induction(
        self,
        enum_mode: str,
        V_forward: np.ndarray,
        X_forward: np.ndarray,
        S_forward: np.ndarray,
        S_curr: np.ndarray,
        Phi_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R = self.risk_free_gross_return
        q = self.risk_neutral_probs()[0]

        if enum_mode == "dense":
            V_curr = (X_forward.reshape(-1, 2) @ np.array([q, 1 - q])) / R
            X_curr = np.maximum(Phi_curr, V_curr)
            tau_curr = Phi_curr > V_curr
            Delta_curr = (
                np.diff(X_forward.reshape(-1, 2)).squeeze()
                / np.diff(S_forward.reshape(-1, 2)).squeeze()
            )
            B_curr = V_curr - S_curr * Delta_curr

            return B_curr, Delta_curr, V_curr, X_curr, tau_curr

        elif enum_mode == "sparse":
            V_curr = (q * V_forward[:-1] + (1 - q) * V_forward[1:]) / R
            Delta_curr = (V_forward[:-1] - V_forward[1:]) / (
                S_forward[:-1] - S_forward[1:]
            )
            B_curr = V_curr - S_curr * Delta_curr
            tau_curr = np.zeros(shape=(len(V_curr),))

            return B_curr, Delta_curr, V_curr, X_curr, tau_curr

    def _broadcast_node_values(
        self,
        t: int,
        B_curr: np.ndarray,
        Delta_curr: np.ndarray,
        V_curr: np.ndarray,
        X_curr: np.ndarray,
        tau_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.time[-1]

        if self.enum_mode == "dense":
            # all have shape (2^T,)
            B = np.repeat(B_curr, repeats=2 ** (T - t))
            Delta = np.repeat(Delta_curr, repeats=2 ** (T - t))
            V = np.repeat(V_curr, repeats=2 ** (T - t))
            X = np.repeat(X_curr, repeats=2 ** (T - t))
            tau = np.repeat(tau_curr, repeats=2 ** (T - t))

        elif self.enum_mode == "sparse":
            # all have shape (T+1,)
            B = np.concatenate((B_curr, np.repeat(B_curr[-1], T - t)))
            Delta = np.concatenate((Delta_curr, np.repeat(Delta_curr[-1], T - t)))
            V = np.concatenate((V_curr, np.repeat(V_curr[-1], T - t)))
            X = np.concatenate((X_curr, np.repeat(V_curr[-1], T - t)))
            tau = np.concatenate((tau_curr, np.repeat(tau_curr[-1], T - t)))

        return B, Delta, V, X, tau

    def _convert_replicating_arrays_to_processes(
        self,
        B_arr: np.ndarray,
        Delta_arr: np.ndarray,
        V_arr: np.ndarray,
        X_arr: np.ndarray,
        tau_arr: np.ndarray,
    ) -> tuple[
        StochasticProcess,
        StochasticProcess,
        StochasticProcess,
        StochasticProcess,
        StoppingTime,
    ]:
        from ...processes.base.stochastic_process import StochasticProcess
        from ...processes.stopping_times.stopping_time import StoppingTime

        B = StochasticProcess(
            *self.prob_space,
            index=self.time[:-1],
            name="B",
            mapping=B_arr,
        )

        Delta = StochasticProcess(
            *self.prob_space,
            index=self.time[:-1],
            name="Delta",
            mapping=Delta_arr,
        )

        V = StochasticProcess(
            *self.prob_space,
            index=self.time,
            name="V",
            mapping=V_arr,
        )

        X = StochasticProcess(
            *self.prob_space,
            index=self.time,
            name="X",
            mapping=X_arr,
        )

        tau = StoppingTime.from_filtration(process=self, mapping=tau_arr, name="tau")

        return B, Delta, V, X, tau

    # --------------------- enumeration methods --------------------- #

    def _enumeration_subclass_hook(self) -> pd.DataFrame:
        """Hook for enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """  # noqa: D401
        if self.enum_mode == "sparse":
            u = self.up_factor
            d = self.down_factor
            T = self._index[-1]

            total_powers = np.tile(np.array(range(1, T + 1)), reps=[T + 1, 1])
            self._u_powers = np.maximum(
                0, total_powers - np.array(range(T + 1)).reshape(-1, 1)
            )
            self._d_powers = total_powers - self._u_powers

            price_factors = (np.ones(shape=(T + 1, T)) * u**self._u_powers) * (
                np.ones(shape=(T + 1, T)) * d**self._d_powers
            )

            sparse_price_array = self.initial_price * np.insert(
                arr=price_factors,
                obj=[0],
                values=np.ones(shape=(T + 1, 1)),
                axis=1,
            )

            return pd.DataFrame(sparse_price_array)

        elif self.enum_mode == "dense":
            time = self._index[1:].data
            initial_time = self._index[0]

            trajectories = pd.DataFrame(
                list(
                    product(
                        [self.up_factor, self.down_factor], repeat=len(self._index) - 1
                    )
                ),
                columns=time,
            )
            trajectories.insert(loc=0, column=initial_time, value=1)
            trajectories = self.initial_price * trajectories.cumprod(axis=1)

            return trajectories

    def _generate_exact_prob_measure(self, domain: Domain) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated binomial pricing process.

        Parameters
        ----------
        domain : Domain
            The domain of the underlying probability space.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.
        """
        from scipy.stats import binom

        from ...core.measures.probability_measure import ProbabilityMeasure

        num_steps = len(self._index) - 1
        T = self._index[-1]
        p = self.up_prob

        if self.enum_mode == "sparse":
            probabilities = dict(
                zip(
                    self.sample_space,
                    binom(n=T, p=1 - p).pmf(range(T + 1)),
                )
            )

            return ProbabilityMeasure(domain=domain, mapping=probabilities)

        elif self.enum_mode == "dense":
            probabilities = pd.DataFrame(
                list(product([p, 1 - p], repeat=num_steps)), index=domain.data
            ).product(axis=1)

            probabilities /= probabilities.sum()

            return ProbabilityMeasure(domain=domain, mapping=probabilities)

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        u = self.up_factor
        d = self.down_factor
        p = self.up_prob
        n_trajectories = self._n_trajectories
        time = self._index[1:].data
        initial_time = self._index[0]

        trajectories = pd.DataFrame(
            self._random_state.choice(
                [u, d],
                size=(n_trajectories, len(time)),
                p=[p, 1 - p],
            ),
            columns=time,
        )
        trajectories.insert(loc=0, column=initial_time, value=1)
        trajectories = self.initial_price * trajectories.cumprod(axis=1)

        return trajectories
