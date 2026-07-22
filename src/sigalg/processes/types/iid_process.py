"""A class representing an independent and identically distributed (IID) stochasti process."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import rv_frozen
    from scipy.stats._multivariate import multinomial_frozen

    from ...core.base.index import Index
    from ...core.measures.probability_measure import ProbabilityMeasure


class IIDProcess(StochasticProcess):
    """A class representing an independent and identically distributed (IID) stochastic process.

    The constructor is not intended for direct usage. Instead, user's should call the `generate` class method. See the Examples section below.

    See also the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
        The sample space of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space.
    index : Index | None, default=None
        The index of the random vector.
    name : Hashable, default="X"
        The name of the random vector.
    **kwargs
        Additional keyword arguments for subclass constructors.

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

    _repr_name = "IID process"

    # --------------------- constructors --------------------- #

    @classmethod
    def generate(
        cls,
        mode: Literal["enum", "sim"],
        distribution: rv_frozen | multinomial_frozen,
        support: list | dict | None = None,
        n_trajectories: int | None = None,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "X",
        random_state: int | np.random.Generator | None = None,
    ) -> IIDProcess:
        """Generate trajectories of the IID process by either exhaustive enumeration or Monte Carlo simulation.

        Parameters
        ----------
        mode : Literal["enum", "sim"]
            Whether to generate trajectories by exhuastive enumeration or Monte Carlo simulation.
        distribution : rv_frozen | multinomial_frozen
            A frozen random variable from scipy.stats representing the common distribution of the IID process.
        support : list | dict | None, default=None
            Either a list containing the support of `distribution`, a dictionary mapping the support to a "new" support, or `None` if the support of the distribution is infinite. See the Examples section below for usage.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. This parameter is ignored if the generation mode is set to `enum`.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        name : Hashable, default="X"
            The name of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded. If the generation mode is set to `enum`, this parameter is ignore.

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

        index, random_state = cls._validate_and_return_generation_params(
            index=index,
            length=length,
            n_trajectories=n_trajectories,
            mode=mode,
            random_state=random_state,
        )
        process = cls(index=index, name=name)
        process._mode = mode
        process._n_trajectories = n_trajectories
        process._random_state = random_state

        process.distribution = distribution
        process.support = support

        if mode == "enum":
            return process._enumeration_logic()
        else:
            return process._simulation_logic()

    # --------------------- enumeration methods --------------------- #

    def _enumeration_hook(self) -> pd.DataFrame:
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

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated IID process.

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
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

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
            index=self.sample_space.data,
        )

        probabilities /= probabilities.sum()
        return ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(self.sample_space),
            mapping=probabilities,
        )

    # --------------------- simulation methods --------------------- #

    def _simulation_hook(self) -> pd.DataFrame:
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
