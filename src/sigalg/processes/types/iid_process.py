"""A class representing an independent and identically distributed (IID) stochastic process."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess, generator

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import rv_frozen
    from scipy.stats._multivariate import multinomial_frozen

    from ...core.indices.time import Time
    from ...core.measures.probability_measure import ProbabilityMeasure
    from ...core.spaces.domain import Domain
    from ...typing.index_like import IndexLike


class IIDProcess(StochasticProcess):
    """A class representing an independent and identically distributed (IID) stochastic process.

    The constructor is not intended for direct usage. Instead, user's should call the `generate` method. See the Examples section below.

    Examples
    --------
    Enumerate all length-2 trajectories of an IID process consisting of Bernoulli random variables.

    >>> from scipy.stats import bernoulli, poisson
    >>> from sigalg.core import Time
    >>> from sigalg.processes import IIDProcess
    >>> T = Time.discrete(length=2)
    >>> X = IIDProcess.generate(
    ...     mode="enum",
    ...     distribution=bernoulli(p=0.25),
    ...     support=[0, 1],
    ...     index=T,
    ... )
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    IID process 'X':
    time    0  1  2
    sample
    0       0  0  0
    1       0  0  1
    2       0  1  0
    3       0  1  1
    4       1  0  0
    5       1  0  1
    6       1  1  0
    7       1  1  1

    Simulate 10 length-2 trajectories from an IID process consisting of Poisson random variables.

    >>> Y = IIDProcess.generate(
    ...     mode="sim",
    ...     distribution=poisson(mu=1.0),
    ...     n_trajectories=10,
    ...     index=T,
    ...     name="Y",
    ...     random_state=42,
    ... )
    >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
    IID process 'Y':
    time    0  1  2
    sample
    0       1  2  3
    1       1  3  0
    2       1  3  3
    3       1  0  3
    4       1  0  0
    5       1  1  2
    6       2  0  1
    7       0  0  3
    8       1  0  0
    9       1  2  1
    """

    _repr_name = "IIDProcess"
    _str_name = "IID process"

    # --------------------- constructors --------------------- #

    @generator
    def generate(
        cls,
        distribution: rv_frozen | multinomial_frozen,
        support: list | dict | None = None,
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        index: Time | IndexLike | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> dict[str, object]:
        """Generate trajectories of the IID process by either exhaustive enumeration or Monte Carlo simulation.

        Parameters
        ----------
        distribution : rv_frozen | multinomial_frozen
            A frozen random variable from scipy.stats representing the common distribution of the IID process.
        support : list | dict | None, default=None
            Either a list containing the support of `distribution`, a dictionary mapping the support to a "new" support, or `None` if the support of the distribution is infinite. See the Examples section below for usage.
        mode : Literal["enum", "sim"], default="sim"
            Whether to generate trajectories by exhaustive enumeration or Monte Carlo simulation.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. This parameter is ignored if the generation mode is set to `enum`.
        index : Time | IndexLike | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        random_state : int | np.random.Generator | None, default=None
            An optional random state for reproducibility.
        name : Hashable, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `distribution` is not a `rv_frozen` or `multinomial_frozen`, or if `support` is not a list or dictionary (if given).

        Returns
        -------
        self : IIDProcess
            The current instance with generated trajectories.

        Examples
        --------
        Enumerate all length-2 trajectories of an IID process consisting of Bernoulli random variables.

        >>> from scipy.stats import bernoulli, poisson
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(length=2)
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.25),
        ...     support=[0, 1],
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  0  0
        1       0  0  1
        2       0  1  0
        3       0  1  1
        4       1  0  0
        5       1  0  1
        6       1  1  0
        7       1  1  1

        Enumerate all length-2 trajectories of an IID process consisting of Bernoulli random variables supported on {a, b}.

        >>> Y = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.25),
        ...     support={0: "a", 1: "b"},
        ...     index=T,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'Y':
        time    0  1  2
        sample
        0       a  a  a
        1       a  a  b
        2       a  b  a
        3       a  b  b
        4       b  a  a
        5       b  a  b
        6       b  b  a
        7       b  b  b

        Simulate 10 length-2 trajectories from an IID process consisting of Poisson random variables.

        >>> Y = IIDProcess.generate(
        ...     mode="sim",
        ...     distribution=poisson(mu=1.0),
        ...     n_trajectories=10,
        ...     index=T,
        ...     name="Y",
        ...     random_state=42,
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'Y':
        time    0  1  2
        sample
        0       1  2  3
        1       1  3  0
        2       1  3  3
        3       1  0  3
        4       1  0  0
        5       1  1  2
        6       2  0  1
        7       0  0  3
        8       1  0  0
        9       1  2  1
        """
        from scipy.stats._distn_infrastructure import rv_frozen
        from scipy.stats._multivariate import multinomial_frozen

        if not isinstance(distribution, rv_frozen | multinomial_frozen):
            raise TypeError(
                "distribution must be an instance of rv_frozen or multinomial_frozen from scipy.stats."
            )
        if support is not None and not isinstance(support, list | dict):
            raise TypeError("If given, support must be a list or dict.")

        return {"distribution": distribution, "support": support}

    # --------------------- properties --------------------- #

    @property
    def distribution(self) -> rv_frozen | multinomial_frozen:
        """Get the distribution of the IID process.

        Returns
        -------
        distribution : rv_frozen | multinomial_frozen
            The distribution of the IID process.
        """
        return self._distribution

    @property
    def support(self) -> list | dict | None:
        """Get the support of the IID process.

        Returns
        -------
        support : list | dict | None
            The support of the IID process. If the support is infinite, this property will be `None`.
        """
        return self._support

    # --------------------- enumeration methods --------------------- #

    def _enumeration_subclass_hook(self) -> pd.DataFrame:
        """Hook for enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """  # noqa: D401
        if isinstance(self.support, dict):
            support = self.support.values()
        else:
            support = self.support

        trajectories = list(product(support, repeat=len(self.time)))
        return pd.DataFrame(data=trajectories, columns=self.time.data)

    def _generate_exact_prob_measure(self, domain: Domain) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated IID process.

        Parameters
        ----------
        domain : Domain
            The domain of the underlying probability space.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(length=2)
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.25),
        ...     support=[0, 1],
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  0  0
        1       0  0  1
        2       0  1  0
        3       0  1  1
        4       1  0  0
        5       1  0  1
        6       1  1  0
        7       1  1  1
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0          0.421875
        1          0.140625
        2          0.140625
        3          0.046875
        4          0.140625
        5          0.046875
        6          0.046875
        7          0.015625
        """
        from scipy.stats._multivariate import multinomial_frozen

        from ...core.measures.probability_measure import ProbabilityMeasure

        if isinstance(self.support, dict):
            inverse_support = {y: x for x, y in self.support.items()}
            values = self.data.map(lambda x: inverse_support[x]).values
        else:
            values = self.data.values

        if isinstance(self.distribution, multinomial_frozen):
            element_wise_probabilities = self.distribution.p[values]
        else:
            element_wise_probabilities = self.distribution.pmf(values)
        probabilities = pd.Series(
            data=np.prod(element_wise_probabilities, axis=1),
            index=domain.data,
        )

        probabilities /= probabilities.sum()
        return ProbabilityMeasure(
            domain=domain,
            mapping=probabilities,
        )

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        """Generate simulated data for the IID process.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        from scipy.stats._multivariate import multinomial_frozen

        trajectories = self.distribution.rvs(
            size=(self.n_trajectories, len(self.time)),
            random_state=self.random_state,
        )
        if isinstance(self.distribution, multinomial_frozen):
            trajectories = trajectories.argmax(axis=-1)

        result = pd.DataFrame(data=trajectories, columns=self.time.data)

        if self.support is not None and isinstance(self.support, dict):
            return result.map(lambda x: self.support[x])
        else:
            return result

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the IID process.

        Returns
        -------
        repr_str : str
            The string representation of the IID process.
        """
        if self.data is None:
            return type(self)._repr_name + "(empty)"
        if self.measure is not None:
            return (
                type(self)._repr_name + f"(domain={self.domain.name}, "
                f"sig_alg={self.sig_alg.name}, "
                f"measure={self.measure.name}, "
                f"distribution={self.distribution.dist.name}, "
                f"support={self.support}, "
                f"name={self.name})"
            )
