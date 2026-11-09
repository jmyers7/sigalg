"""Base class for contingent claims."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal

import pandas as pd

from ...processes.base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...core.functions.random_variable import RandomVariable
    from ...processes.stopping_times import StoppingTime
    from ..discrete_pricing_models.discrete_pricing_model import DiscretePricingModel
    from .american_option import AmericanOption
    from .asian_option import AsianOption
    from .european_option import EuropeanOption


class Claim(StochasticProcess):
    """Base class for contingent claims.

    The base constructor is not meant to be called directly by users. Instead, the user should call the `from_stochastic_process` or `from_random_variables_ class methods. See the Examples section below for usage.

    Examples
    --------
    >>> from sigalg.finance import BinomialPricingModel, Claim

    Create a binomial pricing model.

    >>> S = BinomialPricingModel.generate(
    ...     initial_price=100,
    ...     up_prob=0.7,
    ...     up_factor=1.1,
    ...     down_factor=0.9,
    ...     risk_free_rate=0.05,
    ...     enum_mode="dense",
    ...     mode="enum",
    ...     length=3,
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

    Create a stochastic process representing an American-style Asian call option.

    >>> process = (S.cumavg() - 100).max(0)
    >>> print(process)  # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'max((S_cumavg - 100), 0)':
    t        0    1          2       3
    omega
    0      0.0  5.0  10.333333  16.025
    1      0.0  5.0  10.333333   9.975
    2      0.0  5.0   3.000000   4.475
    3      0.0  5.0   3.000000   0.000
    4      0.0  0.0   0.000000   0.000
    5      0.0  0.0   0.000000   0.000
    6      0.0  0.0   0.000000   0.000
    7      0.0  0.0   0.000000   0.000

    Construct the contingent claim.

    >>> Phi = Claim.from_stochastic_process(process=process, model=S, name="Phi")
    >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
    Claim 'Phi':
    t        0    1          2       3
    omega
    0      0.0  5.0  10.333333  16.025
    1      0.0  5.0  10.333333   9.975
    2      0.0  5.0   3.000000   4.475
    3      0.0  5.0   3.000000   0.000
    4      0.0  0.0   0.000000   0.000
    5      0.0  0.0   0.000000   0.000
    6      0.0  0.0   0.000000   0.000
    7      0.0  0.0   0.000000   0.000

    Create the payoff random variable of a European call with strike `K=100`.

    >>> Phi_3 = (S[3] - 100).max(0)
    >>> print(Phi_3)  # doctest: +NORMALIZE_WHITESPACE
    Random variable 'max((S_3 - 100), 0)':
            max((S_3 - 100), 0)
    omega
    0                     33.1
    1                      8.9
    2                      8.9
    3                      0.0
    4                      8.9
    5                      0.0
    6                      0.0
    7                      0.0

    Create an instance of `Claim` from the random variable.

    >>> Phi = Claim.from_random_variables(rvs=[Phi_3], times=[3], model=S)
    >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
    Claim 'Phi':
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

    Create the payoff random variable of a European call with strike `K=100` at time `t=2`.

    >>> Phi_2 = (S[2] - 100).max(0)
    >>> print(Phi_2)  # doctest: +NORMALIZE_WHITESPACE
    Random variable 'max((S_2 - 100), 0)':
            max((S_2 - 100), 0)
    omega
    0                     21.0
    1                     21.0
    2                      0.0
    3                      0.0
    4                      0.0
    5                      0.0
    6                      0.0
    7                      0.0

    Combine the random variables `Phi_3` and `Phi_2` into a single instance of `Claim` that allows exercise at times `t=2` and `t=3`.

    >>> Phi = Claim.from_random_variables(rvs=[Phi_2, Phi_3], times=[2, 3], model=S)
    >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
    Claim 'Phi':
    t        0    1     2     3
    omega
    0      0.0  0.0  21.0  33.1
    1      0.0  0.0  21.0   8.9
    2      0.0  0.0   0.0   8.9
    3      0.0  0.0   0.0   0.0
    4      0.0  0.0   0.0   8.9
    5      0.0  0.0   0.0   0.0
    6      0.0  0.0   0.0   0.0
    7      0.0  0.0   0.0   0.0
    """

    _repr_name = "Claim"
    _str_name = "Claim"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_stochastic_process(
        cls,
        process: StochasticProcess,
        model: DiscretePricingModel,
        name: Hashable | None = None,
    ):
        """Construct a contingent claim from a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process used to generate the contingent claim.
        model : DiscretePricingModel
            The pricing model.
        name : Hashable | None, default=None
            The name for the claim. If `None`, the pre-existing name of the stochastic process will be used.

        Returns
        -------
        claim : Claim
            The contingent claim created from the stochastic process

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Create a binomial pricing model.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=100,
        ...     up_prob=0.7,
        ...     up_factor=1.1,
        ...     down_factor=0.9,
        ...     risk_free_rate=0.05,
        ...     enum_mode="dense",
        ...     mode="enum",
        ...     length=3,
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

        Create a stochastic process representing an American-style Asian call option.

        >>> process = (S.cumavg() - 100).max(0)
        >>> print(process)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'max((S_cumavg - 100), 0)':
        t        0    1          2       3
        omega
        0      0.0  5.0  10.333333  16.025
        1      0.0  5.0  10.333333   9.975
        2      0.0  5.0   3.000000   4.475
        3      0.0  5.0   3.000000   0.000
        4      0.0  0.0   0.000000   0.000
        5      0.0  0.0   0.000000   0.000
        6      0.0  0.0   0.000000   0.000
        7      0.0  0.0   0.000000   0.000

        Construct the contingent claim.

        >>> Phi = Claim.from_stochastic_process(process=process, model=S, name="Phi")
        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        Claim 'Phi':
        t        0    1          2       3
        omega
        0      0.0  5.0  10.333333  16.025
        1      0.0  5.0  10.333333   9.975
        2      0.0  5.0   3.000000   4.475
        3      0.0  5.0   3.000000   0.000
        4      0.0  0.0   0.000000   0.000
        5      0.0  0.0   0.000000   0.000
        6      0.0  0.0   0.000000   0.000
        7      0.0  0.0   0.000000   0.000
        """
        from ..discrete_pricing_models.discrete_pricing_model import (
            DiscretePricingModel,
        )

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(model, DiscretePricingModel):
            raise TypeError("model must be an instance of DiscretePricingModel.")
        if process.domain != model.domain or process.index != model.index:
            raise ValueError(
                "The process and the model must have the same domain and index."
            )
        if not process.is_adapted(model.natural_filtration):
            raise ValueError(
                "The claim is not adapted to the natural filtration of the model."
            )

        claim = Claim.__new__(Claim)
        claim.__dict__.update(process.__dict__)
        claim.model = model

        if name is not None:
            claim.name = name

        return claim

    @classmethod
    def from_random_variables(
        cls,
        rvs: list[RandomVariable],
        times: list[Real],
        model: DiscretePricingModel,
        name: Hashable = "Phi",
    ) -> Claim:
        """Construct a contingent claim from a list of random variables and a list of times.

        The random variables are inserted into the claim at the specified times. Any unspecified times are filled with the `0` random variable. See the Examples section below.

        Parameters
        ----------
        rvs : list[RandomVariable]
            The list of random variables to include as the components of the contingent claim.
        times : list[Real]
            The list of times corresponding to the random variables.
        model : DiscretePricingModel
            The pricing model.
        name : Hashable, default="Phi"
            The name for the claim.

        Returns
        -------
        claim : Claim
            The contingent claim created from the random variables.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Create a binomial pricing model.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=100,
        ...     up_prob=0.7,
        ...     up_factor=1.1,
        ...     down_factor=0.9,
        ...     risk_free_rate=0.05,
        ...     enum_mode="dense",
        ...     mode="enum",
        ...     length=3,
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

        Create the payoff random variable of a European call with strike `K=100`.

        >>> Phi_3 = (S[3] - 100).max(0)
        >>> print(Phi_3)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'max((S_3 - 100), 0)':
               max((S_3 - 100), 0)
        omega
        0                     33.1
        1                      8.9
        2                      8.9
        3                      0.0
        4                      8.9
        5                      0.0
        6                      0.0
        7                      0.0

        Create an instance of `Claim` from the random variable.

        >>> Phi = Claim.from_random_variables(rvs=[Phi_3], times=[3], model=S)
        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        Claim 'Phi':
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

        Create the payoff random variable of a European call with strike `K=100` at time `t=2`.

        >>> Phi_2 = (S[2] - 100).max(0)
        >>> print(Phi_2)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'max((S_2 - 100), 0)':
               max((S_2 - 100), 0)
        omega
        0                     21.0
        1                     21.0
        2                      0.0
        3                      0.0
        4                      0.0
        5                      0.0
        6                      0.0
        7                      0.0

        Combine the random variables `Phi_3` and `Phi_2` into a single instance of `Claim` that allows exercise at times `t=2` and `t=3`.

        >>> Phi = Claim.from_random_variables(rvs=[Phi_2, Phi_3], times=[2, 3], model=S)
        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        Claim 'Phi':
        t        0    1     2     3
        omega
        0      0.0  0.0  21.0  33.1
        1      0.0  0.0  21.0   8.9
        2      0.0  0.0   0.0   8.9
        3      0.0  0.0   0.0   0.0
        4      0.0  0.0   0.0   8.9
        5      0.0  0.0   0.0   0.0
        6      0.0  0.0   0.0   0.0
        7      0.0  0.0   0.0   0.0
        """
        from ...core.functions.random_variable import RandomVariable
        from ...core.indices.time import Time
        from ..discrete_pricing_models.discrete_pricing_model import (
            DiscretePricingModel,
        )

        if not all(isinstance(rv, RandomVariable) for rv in rvs):
            raise TypeError(
                "The list of random variables must only contain instances of RandomVariable."
            )
        if not isinstance(model, DiscretePricingModel):
            raise TypeError("model must be an instance of DiscretePricingModel.")
        if not all(isinstance(t, Real) for t in times):
            raise TypeError("All times must be real numbers.")
        if not all(t in model.index.data for t in times):
            raise ValueError("All times must be in the time index of the model.")
        if not all(rv.domain == model.domain for rv in rvs):
            raise ValueError(
                "The domain of every random variable must match the domain of the model."
            )
        if len(rvs) != len(times):
            raise ValueError(
                "The number of random variables must match the number of times."
            )

        missing_times = Time(
            indices=[t for t in model.index if t not in times],
            variable_names=model.index.variable_names,
            name=model.index.name,
        )
        data = pd.DataFrame(
            data=0, index=model.domain.data, columns=missing_times.data, dtype="float64"
        )
        rvs = pd.DataFrame({t: rv.data for t, rv in zip(times, rvs)}).reindex(
            data.index
        )
        data = pd.concat([data, rvs], axis=1).sort_index(axis=1)
        data.columns = model.index.data

        claim = Claim._from_validated(
            data=data,
            sig_alg=model.sig_alg,
            measure=model.measure,
            index_kind="Time",
            index_name=model.index_name,
            name=name,
        )

        claim.model = model

        if not claim.is_adapted(model.natural_filtration):
            raise ValueError(
                "The claim is not adapted to the natural filtration of the model."
            )

        return claim

    # --------------------- properties --------------------- #

    @property
    def is_early_exercise(self) -> bool:
        """Return whether the claim allows early exercise or not.

        Returns
        -------
        is_early_exercise : bool
            Whether the claim allows early exercise or not.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Create a binomial pricing model.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=100,
        ...     up_prob=0.7,
        ...     up_factor=1.1,
        ...     down_factor=0.9,
        ...     risk_free_rate=0.05,
        ...     enum_mode="dense",
        ...     mode="enum",
        ...     length=3,
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

        Create the payoff random variable of a European call with strike `K=100`.

        >>> Phi_3 = (S[3] - 100).max(0)
        >>> print(Phi_3)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'max((S_3 - 100), 0)':
                max((S_3 - 100), 0)
        omega
        0                     33.1
        1                      8.9
        2                      8.9
        3                      0.0
        4                      8.9
        5                      0.0
        6                      0.0
        7                      0.0

        Create an instance of `Claim` from the random variable and check whether it allows early exercise.

        >>> Phi = Claim.from_random_variables(rvs=[Phi_3], times=[3], model=S)
        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        Claim 'Phi':
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
        >>> Phi.is_early_exercise
        False

        Create the payoff random variable of a European call with strike `K=100` at time `t=2`.

        >>> Phi_2 = (S[2] - 100).max(0)
        >>> print(Phi_2)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'max((S_2 - 100), 0)':
                max((S_2 - 100), 0)
        omega
        0                     21.0
        1                     21.0
        2                      0.0
        3                      0.0
        4                      0.0
        5                      0.0
        6                      0.0
        7                      0.0

        Combine the random variables `Phi_3` and `Phi_2` into a single instance of `Claim` that allows exercise at times `t=2` and `t=3`.

        >>> Phi = Claim.from_random_variables(rvs=[Phi_2, Phi_3], times=[2, 3], model=S)
        >>> print(Phi)  # doctest: +NORMALIZE_WHITESPACE
        Claim 'Phi':
        t        0    1     2     3
        omega
        0      0.0  0.0  21.0  33.1
        1      0.0  0.0  21.0   8.9
        2      0.0  0.0   0.0   8.9
        3      0.0  0.0   0.0   0.0
        4      0.0  0.0   0.0   8.9
        5      0.0  0.0   0.0   0.0
        6      0.0  0.0   0.0   0.0
        7      0.0  0.0   0.0   0.0
        >>> Phi.is_early_exercise
        True
        """
        return not bool((self.data.iloc[:, :-1] == 0).all().all())

    # --------------------- standard types --------------------- #

    @classmethod
    def european_option(
        cls,
        strike: Real,
        model: DiscretePricingModel,
        type: Literal["call", "put"] = "call",
        name: Hashable = "Phi",
    ) -> EuropeanOption:
        """Create a European option.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Define a binomial pricing model.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=100,
        ...     up_prob=0.7,
        ...     up_factor=1.1,
        ...     down_factor=0.9,
        ...     risk_free_rate=0.05,
        ...     enum_mode="dense",
        ...     mode="enum",
        ...     length=3,
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

        Create a European call option with strike `K=100`.

        >>> K = 100
        >>> Phi_call = Claim.european_option(strike=K, model=S, type="call", name="Phi_call")
        >>> print(Phi_call)  # doctest: +NORMALIZE_WHITESPACE
        European option 'Phi_call':
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

        Create a European put option with the same strike.

        >>> Phi_put = Claim.european_option(strike=K, model=S, type="put", name="Phi_put")
        >>> print(Phi_put)  # doctest: +NORMALIZE_WHITESPACE
        European option 'Phi_put':
        t        0    1    2     3
        omega
        0      0.0  0.0  0.0   0.0
        1      0.0  0.0  0.0   0.0
        2      0.0  0.0  0.0   0.0
        3      0.0  0.0  0.0  10.9
        4      0.0  0.0  0.0   0.0
        5      0.0  0.0  0.0  10.9
        6      0.0  0.0  0.0  10.9
        7      0.0  0.0  0.0  27.1
        """
        from .european_option import EuropeanOption

        if type == "call":
            rv = (model.last_rv - strike).max(0)
        else:
            rv = (strike - model.last_rv).max(0)

        claim = cls.from_random_variables(
            rvs=[rv], times=[model.time[-1]], model=model, name=name
        )
        claim.__class__ = EuropeanOption
        claim._strike = strike
        claim._option_type = type

        return claim

    @classmethod
    def asian_option(
        cls,
        strike: Real,
        model: DiscretePricingModel,
        type: Literal["call", "put"] = "call",
        early_exercise: bool = False,
        name: Hashable = "Phi",
    ) -> AsianOption:
        """Pass."""
        from .asian_option import AsianOption

        if type == "call":
            if early_exercise:
                process = (model.cumavg() - strike).max(0)
            else:
                rv = (model.cumavg().last_rv - strike).max(0)
        else:
            if early_exercise:
                process = (strike - model.cumavg()).max(0)
            else:
                rv = (strike - model.cumavg().last_rv).max(0)

        if early_exercise:
            claim = cls.from_stochastic_process(process=process, model=model, name=name)
        else:
            claim = cls.from_random_variables(
                rvs=[rv], times=[model.time[-1]], model=model, name=name
            )

        claim.__class__ = AsianOption
        claim._strike = strike
        claim._option_type = type

        return claim

    @classmethod
    def american_option(
        cls,
        strike: Real,
        model: DiscretePricingModel,
        type: Literal["call", "put"] = "call",
        name: Hashable = "Phi",
    ) -> AmericanOption:
        """Create an Asian option.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel, Claim

        Define a binomial pricing model.

        >>> S = BinomialPricingModel.generate(
        ...     initial_price=100,
        ...     up_prob=0.7,
        ...     up_factor=1.1,
        ...     down_factor=0.9,
        ...     risk_free_rate=0.05,
        ...     enum_mode="dense",
        ...     mode="enum",
        ...     length=3,
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

        Create an Asian call option with strike `K=100` and no early exercise.

        >>> K = 100
        >>> Phi_call_no_early = Claim.asian_option(
        ...     strike=100, model=S, type="call", early_exercise=False, name="Phi_call_no_early"
        ... )
        >>> print(Phi_call_no_early)  # doctest: +NORMALIZE_WHITESPACE
        Asian option 'Phi_call_no_early':
        t        0    1    2       3
        omega
        0      0.0  0.0  0.0  16.025
        1      0.0  0.0  0.0   9.975
        2      0.0  0.0  0.0   4.475
        3      0.0  0.0  0.0   0.000
        4      0.0  0.0  0.0   0.000
        5      0.0  0.0  0.0   0.000
        6      0.0  0.0  0.0   0.000
        7      0.0  0.0  0.0   0.000

        Create an Asian call with the same strike and with early exercise.

        >>> Phi_call_early = Claim.asian_option(
        ...     strike=100, model=S, type="call", early_exercise=True, name="Phi_call_early"
        ... )
        >>> print(Phi_call_early)  # doctest: +NORMALIZE_WHITESPACE
        Asian option 'Phi_call_early':
        t        0    1          2       3
        omega
        0      0.0  5.0  10.333333  16.025
        1      0.0  5.0  10.333333   9.975
        2      0.0  5.0   3.000000   4.475
        3      0.0  5.0   3.000000   0.000
        4      0.0  0.0   0.000000   0.000
        5      0.0  0.0   0.000000   0.000
        6      0.0  0.0   0.000000   0.000
        7      0.0  0.0   0.000000   0.000

        Create an Asian put with the same strike and no early exercise.

        >>> Phi_put_no_early = Claim.asian_option(
        ...     strike=100, model=S, type="put", early_exercise=False, name="Phi_put_no_early"
        ... )
        >>> print(Phi_put_no_early)  # doctest: +NORMALIZE_WHITESPACE
        Asian option 'Phi_put_no_early':
        t        0    1    2       3
        omega
        0      0.0  0.0  0.0   0.000
        1      0.0  0.0  0.0   0.000
        2      0.0  0.0  0.0   0.000
        3      0.0  0.0  0.0   0.475
        4      0.0  0.0  0.0   0.525
        5      0.0  0.0  0.0   5.475
        6      0.0  0.0  0.0   9.975
        7      0.0  0.0  0.0  14.025

        Create an Asian put with the same strike and early exercise.

        >>> Phi_put_early = Claim.asian_option(
        ...     strike=100, model=S, type="put", early_exercise=True, name="Phi_put_early"
        ... )
        >>> print(Phi_put_early)  # doctest: +NORMALIZE_WHITESPACE
        Asian option 'Phi_put_early':
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
        """
        from .american_option import AmericanOption

        if type == "call":
            process = (model - strike).max(0)
        else:
            process = (strike - model).max(0)

        claim = cls.from_stochastic_process(process=process, model=model, name=name)
        claim.__class__ = AmericanOption
        claim._strike = strike
        claim._option_type = type

        return claim

    # --------------------- finance methods --------------------- #

    @dataclass
    class PricingData:
        """Data class for pricing data for contingent claims."""

        price: Real | None = None
        price_process: StochasticProcess | None = None
        continuation_process: StochasticProcess | None = None
        Delta: StochasticProcess | None = None
        B: StochasticProcess | None = None
        exercise_time: StoppingTime | None = None

    def price(self, **kwargs) -> Real:
        """Get the price of the claim.

        Returns
        -------
        price : Real
            The price of the claim relative to the underlying price model.

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

        Get the risk-neutral price of the call option.

        >>> euro_call.price()
        15.306122448979611

        Define an American put option with the same strike.

        >>> american_put = Claim.american_option(model=S, strike=K, type="put", name="american_put")
        >>> print(american_put)  # doctest: +NORMALIZE_WHITESPACE
        American option 'american_put':
        t        0     1     2     3
        omega
        0      0.0   0.0   0.0   0.0
        1      0.0   0.0   0.0   0.0
        2      0.0   0.0   1.0   0.0
        3      0.0   0.0   1.0  10.9
        4      0.0  10.0   1.0   0.0
        5      0.0  10.0   1.0  10.9
        6      0.0  10.0  19.0  10.9
        7      0.0  10.0  19.0  27.1

        Get the risk-neutral price of the put option.

        >>> american_put.price()
        2.822319403952058
        """
        return self.model.price(claim=self, **kwargs)

    def price_process(self, **kwargs) -> StochasticProcess:
        """Get the price process of the claim.

        Returns
        -------
        price_process : StochasticProcess
            The price process of the claim relative to the underlying price model.

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

        Get the risk-neutral price process of the call option.

        >>> print(euro_call.price_process())  # doctest: +NORMALIZE_WHITESPACE
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

        Define an American put option with the same strike.

        >>> american_put = Claim.american_option(model=S, strike=K, type="put", name="american_put")
        >>> print(american_put)  # doctest: +NORMALIZE_WHITESPACE
        American option 'american_put':
        t        0     1     2     3
        omega
        0      0.0   0.0   0.0   0.0
        1      0.0   0.0   0.0   0.0
        2      0.0   0.0   1.0   0.0
        3      0.0   0.0   1.0  10.9
        4      0.0  10.0   1.0   0.0
        5      0.0  10.0   1.0  10.9
        6      0.0  10.0  19.0  10.9
        7      0.0  10.0  19.0  27.1

        Get the risk-neutral price process of the put option.

        >>> print(american_put.price_process())  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        t             0          1          2     3
        omega
        0      2.822319   0.617914   0.000000   0.0
        1      2.822319   0.617914   0.000000   0.0
        2      2.822319   0.617914   2.595238   0.0
        3      2.822319   0.617914   2.595238  10.9
        4      2.822319  10.000000   2.595238   0.0
        5      2.822319  10.000000   2.595238  10.9
        6      2.822319  10.000000  19.000000  10.9
        7      2.822319  10.000000  19.000000  27.1
        """
        return self.model.price_process(claim=self, **kwargs)

    def continuation_process(self, **kwargs) -> StoppingTime:
        """Get the continuation process of the claim.

        Returns
        -------
        continuation_process : StochasticProcess
            The continuation process of the claim relative to the underlying price model.

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

        Get the risk-neutral continuation process of the call option.

        >>> print(euro_call.continuation_process())  # doctest: +NORMALIZE_WHITESPACE
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

        Define an American put option with the same strike.

        >>> american_put = Claim.american_option(model=S, strike=K, type="put", name="american_put")
        >>> print(american_put)  # doctest: +NORMALIZE_WHITESPACE
        American option 'american_put':
        t        0     1     2     3
        omega
        0      0.0   0.0   0.0   0.0
        1      0.0   0.0   0.0   0.0
        2      0.0   0.0   1.0   0.0
        3      0.0   0.0   1.0  10.9
        4      0.0  10.0   1.0   0.0
        5      0.0  10.0   1.0  10.9
        6      0.0  10.0  19.0  10.9
        7      0.0  10.0  19.0  27.1

        Get the risk-neutral continuation process of the put option.

        >>> print(american_put.continuation_process())  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'V':
        t             0         1          2     3
        omega
        0      2.822319  0.617914   0.000000   0.0
        1      2.822319  0.617914   0.000000   0.0
        2      2.822319  0.617914   2.595238   0.0
        3      2.822319  0.617914   2.595238  10.9
        4      2.822319  6.377551   2.595238   0.0
        5      2.822319  6.377551   2.595238  10.9
        6      2.822319  6.377551  14.238095  10.9
        7      2.822319  6.377551  14.238095  27.1
        """
        return self.model.continuation_process(claim=self, **kwargs)

    def exercise_time(self, **kwargs) -> StoppingTime:
        """Get the exercise time of the claim.

        Returns
        -------
        exercise_time : StoppingTime
            The exercise time of the claim relative to the underlying price model.

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

        Get the exercise time of the call option.

        >>> print(euro_call.exercise_time())  # doctest: +NORMALIZE_WHITESPACE
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

        Define an American put option with the same strike.

        >>> american_put = Claim.american_option(model=S, strike=K, type="put", name="american_put")
        >>> print(american_put)  # doctest: +NORMALIZE_WHITESPACE
        American option 'american_put':
        t        0     1     2     3
        omega
        0      0.0   0.0   0.0   0.0
        1      0.0   0.0   0.0   0.0
        2      0.0   0.0   1.0   0.0
        3      0.0   0.0   1.0  10.9
        4      0.0  10.0   1.0   0.0
        5      0.0  10.0   1.0  10.9
        6      0.0  10.0  19.0  10.9
        7      0.0  10.0  19.0  27.1

        Get the exercise time of the put option.

        >>> print(american_put.exercise_time())  # doctest: +NORMALIZE_WHITESPACE
        Stopping time 'tau':
               tau
        omega
        0      inf
        1      inf
        2      inf
        3      3.0
        4      1.0
        5      1.0
        6      1.0
        7      1.0
        """
        return self.model.exercise_time(claim=self, **kwargs)

    def delta(self, **kwargs) -> StochasticProcess:
        """Get the delta process of the claim.

        Returns
        -------
        delta : StochasticProcess
            The delta process of the claim relative to the underlying price model.

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

        Get the exercise time of the call option.

        >>> print(euro_call.delta())  # doctest: +NORMALIZE_WHITESPACE
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

        Define an American put option with the same strike.

        >>> american_put = Claim.american_option(model=S, strike=K, type="put", name="american_put")
        >>> print(american_put)  # doctest: +NORMALIZE_WHITESPACE
        American option 'american_put':
        t        0     1     2     3
        omega
        0      0.0   0.0   0.0   0.0
        1      0.0   0.0   0.0   0.0
        2      0.0   0.0   1.0   0.0
        3      0.0   0.0   1.0  10.9
        4      0.0  10.0   1.0   0.0
        5      0.0  10.0   1.0  10.9
        6      0.0  10.0  19.0  10.9
        7      0.0  10.0  19.0  27.1

        Get the exercise time of the put option.

        >>> print(american_put.delta())  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Delta':
        t             0         1         2
        omega
        0     -0.469104 -0.117965 -0.000000
        1     -0.469104 -0.117965 -0.000000
        2     -0.469104 -0.117965 -0.550505
        3     -0.469104 -0.117965 -0.550505
        4     -0.469104 -0.911376 -0.550505
        5     -0.469104 -0.911376 -0.550505
        6     -0.469104 -0.911376 -1.000000
        7     -0.469104 -0.911376 -1.000000
        """
        return self.model.delta(claim=self, **kwargs)

    def B(self, **kwargs) -> StochasticProcess:
        """Get the price process of the non-risky asset of the claim.

        Returns
        -------
        B : StochasticProcess
            The price process of the non-risky asset of the claim.

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

        Get the exercise time of the call option.

        >>> print(euro_call.B())  # doctest: +NORMALIZE_WHITESPACE
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

        Define an American put option with the same strike.

        >>> american_put = Claim.american_option(model=S, strike=K, type="put", name="american_put")
        >>> print(american_put)  # doctest: +NORMALIZE_WHITESPACE
        American option 'american_put':
        t        0     1     2     3
        omega
        0      0.0   0.0   0.0   0.0
        1      0.0   0.0   0.0   0.0
        2      0.0   0.0   1.0   0.0
        3      0.0   0.0   1.0  10.9
        4      0.0  10.0   1.0   0.0
        5      0.0  10.0   1.0  10.9
        6      0.0  10.0  19.0  10.9
        7      0.0  10.0  19.0  27.1

        Get the exercise time of the put option.

        >>> print(american_put.B())  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'B':
        t             0          1          2
        omega
        0      49.73275  13.594104   0.000000
        1      49.73275  13.594104   0.000000
        2      49.73275  13.594104  57.095238
        3      49.73275  13.594104  57.095238
        4      49.73275  88.401361  57.095238
        5      49.73275  88.401361  57.095238
        6      49.73275  88.401361  95.238095
        7      49.73275  88.401361  95.238095
        """
        return self.model.B(claim=self, **kwargs)

    def replicating_portfolio(self, **kwargs):
        """Compute the replicating portfolio of the claim.

        Returns
        -------
        replicating_portfolio : tuple[StochasticProcess, StochasticProcess, StochasticProcess,]
            Three processes `Delta`, `B`, and `V`. The first gives the number of units of the risky asset, the second gives the cash value of the position in the non-risky asset, and the third is the value process of the two-asset portfolio.

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

        >>> Delta, B, V = euro_call.replicating_portfolio()

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

        >>> Delta, B, V = asian_put.replicating_portfolio()
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
        """
        return self.model.replicating_portfolio(claim=self, **kwargs)
