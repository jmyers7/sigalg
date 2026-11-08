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
    """Base class for contingent claims."""

    _repr_name = "Claim"
    _str_name = "Claim"

    @dataclass
    class PricingData:
        """Pass."""

        price: Real | None = None
        price_process: StochasticProcess | None = None
        continuation_process: StochasticProcess | None = None
        Delta: StochasticProcess | None = None
        B: StochasticProcess | None = None
        exercise_time: StoppingTime | None = None

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
        """Pass."""
        return not bool((self.data.iloc[:, :-1] == 0).all().all())

    @property
    def pricing_data(self) -> PricingData | dict[Real, PricingData] | None:
        """Pass."""
        if not hasattr(self, "_pricing_data"):
            raise ValueError("Pricing data has not yet been computed for this claim.")
        else:
            return self._pricing_data

    # --------------------- types --------------------- #

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

    def price(self, **kwargs) -> Real:
        """Pass."""
        if hasattr(self, "_price"):
            return self._price
        else:
            return self.model.price(claim=self, **kwargs)

    def price_process(self, **kwargs) -> StochasticProcess:
        """Pass."""
        return self.model.price_process(claim=self, **kwargs)

    def exercise_time(self, **kwargs) -> StoppingTime:
        """Pass."""
        return self.model.exercise_time(claim=self, **kwargs)

    def delta(self, **kwargs) -> StochasticProcess:
        """Pass."""
        return self.model.delta(claim=self, **kwargs)

    def replicating_portfolio(self, **kwargs):
        """Pass."""
        return self.model.replicating_portfolio(claim=self, **kwargs)
