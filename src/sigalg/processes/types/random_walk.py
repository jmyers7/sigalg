"""A class representing a random walk stochastic process."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from ...core.base.index import Index
    from ...core.probability_measures.probability_measure import ProbabilityMeasure


class RandomWalk(StochasticProcess):
    """A class representing a random walk stochastic process.

    The constructor is not intended for direct usage. Instead, user's should call one of either class methods `from_enumeration` or `from_simulation`. See the Examples section below.

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
    Define a random walk with probability p=0.75 of stepping right one unit, and 0.25 of stepping left one unit.

    >>> from math import comb
    >>> from sigalg.core import Time
    >>> from sigalg.processes import RandomWalk
    >>> time = Time.discrete(length=3)
    >>> X = RandomWalk.from_enumeration(p=0.75, initial_state=0, name="X", index=time)
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Random walk 'X':
    time    0  1  2  3
    sample
    0       0 -1 -2 -3
    1       0 -1 -2 -1
    2       0 -1  0 -1
    3       0 -1  0  1
    4       0  1  0 -1
    5       0  1  0  1
    6       0  1  2  1
    7       0  1  2  3

    Print the values of the X_3 component random variable and its corresponding law.

    >>> print(X[3].range.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P_X_3':
         probability
    X_3
    -3      0.015625
    -1      0.140625
     1      0.421875
     3      0.421875

    Print binomial probabilities and note they match the law of X_3.

    >>> for k in range(4):
    ...     print(comb(3, k) * (0.75**k) * (0.25 ** (3 - k)))
    0.015625
    0.140625
    0.421875
    0.421875
    """

    _repr_name = "Random walk"

    # --------------------- enumeration methods --------------------- #

    @classmethod
    def from_enumeration(
        cls,
        p: Real,
        initial_state: int,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Generate all trajectories of the random walk by exhaustive enumeration.

        Parameters
        ----------
        p : Real
            The probability of stepping "to the right".
        initial_state : int
            The initial state of the random walk.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `p` is not a real number, or if `initial_state` is not an integer.
        ValueError
            If `p` is not between 0 and 1.

        Returns
        -------
        self : StochasticProcess
            The current instance with all trajectories enumerated.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> time = Time.discrete(length=3)
        >>> X = RandomWalk.from_enumeration(p=0.75, initial_state=0, name="X", index=time)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2  3
        sample
        0       0 -1 -2 -3
        1       0 -1 -2 -1
        2       0 -1  0 -1
        3       0 -1  0  1
        4       0  1  0 -1
        5       0  1  0  1
        6       0  1  2  1
        7       0  1  2  3
        """
        if not isinstance(p, Real):
            raise TypeError("p must be a real number.")
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1.")
        if not isinstance(initial_state, int):
            raise TypeError("initial_state must be a real number.")

        index = cls._validate_and_return_index(index=index, length=length)
        process = cls(index=index, name=name)

        process.p = p
        process.initial_state = initial_state

        return process._enumeration_logic()

    def _enumeration_hook(self) -> pd.DataFrame:
        """Hook for enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """  # noqa: D401
        from scipy.stats import bernoulli

        from ...core.random_objects.random_variable import RandomVariable
        from .iid_process import IIDProcess

        if len(self.time) == 1:
            return pd.DataFrame(data=[self.initial_state], columns=self.time.data)

        step_indicators = IIDProcess.from_enumeration(
            distribution=bernoulli(p=self.p),
            support=[0, 1],
            index=self.time[1:],
            name="step_indicators",
        )
        self.step_indicators = step_indicators

        displacements = (2 * step_indicators - 1).with_name("displacements")
        initial_state = RandomVariable.from_constant(
            sample_space=step_indicators.sample_space, constant=0
        )

        S = (
            displacements.cumsum(name="S").insert_rv(
                rv=initial_state,
                time=self.time[0],
            )
            + self.initial_state
        )

        return S.data

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated IID process.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> time = Time.discrete(length=3)
        >>> X = RandomWalk.from_enumeration(p=0.75, initial_state=0, name="X", index=time)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2  3
        sample
        0       0 -1 -2 -3
        1       0 -1 -2 -1
        2       0 -1  0 -1
        3       0 -1  0  1
        4       0  1  0 -1
        5       0  1  0  1
        6       0  1  2  1
        7       0  1  2  3
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0          0.015625
        1          0.046875
        2          0.046875
        3          0.140625
        4          0.046875
        5          0.140625
        6          0.140625
        7          0.421875
        """
        return self.step_indicators._generate_exact_prob_measure()

    # --------------------- simulation methods --------------------- #

    @classmethod
    def from_simulation(
        cls,
        p: Real,
        initial_state: int,
        n_trajectories: int,
        index: Index | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Simulate trajectories of the random walk.

        Parameters
        ----------
        p : Real
            The probability of stepping "to the right".
        initial_state : int
            The initial state of the random walk.
        n_trajectories : int
            The number of trajectories to simulate.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Returns
        -------
        self : StochasticProcess
            The current instance with simulated trajectories.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> time = Time.discrete(length=3)
        >>> X = RandomWalk.from_simulation(
        ...     p=0.75,
        ...     initial_state=0,
        ...     n_trajectories=10_000,
        ...     name="X",
        ...     index=time,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2  3
        sample
        0       0 -1  0 -1
        1       0  1  2  1
        2       0 -1 -2 -1
        3       0  1  2  1
        4       0  1  0  1
        ...    .. .. .. ..
        9995    0  1  2  3
        9996    0 -1 -2 -1
        9997    0 -1  0  1
        9998    0 -1  0  1
        9999    0  1  0  1
        <BLANKLINE>
        [10000 rows x 4 columns]
        """
        if not isinstance(p, Real):
            raise TypeError("p must be a real number.")
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1.")
        if not isinstance(initial_state, int):
            raise TypeError("initial_state must be a real number.")

        index = cls._validate_and_return_index(index=index, length=length)
        random_state = cls._validate_simulation_parameters_and_return_random_state(
            n_trajectories=n_trajectories, random_state=random_state
        )
        process = cls(index=index, name=name)

        process.n_trajectories = n_trajectories
        process.random_state = random_state
        process.p = p
        process.initial_state = initial_state

        return process._simulation_logic()

    def _simulation_hook(self) -> pd.DataFrame:
        """Generate simulated data for the random walk.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        from scipy.stats import bernoulli

        from ...core.random_objects.random_variable import RandomVariable
        from .iid_process import IIDProcess

        if len(self.time) == 1:
            return pd.DataFrame(data=[self.initial_state], columns=self.time.data)

        step_indicators = IIDProcess.from_simulation(
            distribution=bernoulli(p=self.p),
            index=self.time[1:],
            name="step_indicators",
            n_trajectories=self.n_trajectories,
            random_state=self.random_state,
        )

        displacements = (2 * step_indicators - 1).with_name("displacements")
        initial_state = RandomVariable.from_constant(
            sample_space=step_indicators.sample_space, constant=0
        )

        S = (
            displacements.cumsum(name="S").insert_rv(
                rv=initial_state,
                time=self.time[0],
            )
            + self.initial_state
        )

        return S.data
