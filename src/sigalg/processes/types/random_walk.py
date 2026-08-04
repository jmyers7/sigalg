"""A class representing a random walk stochastic process."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

from ..base.stochastic_process import StochasticProcess, generator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from ...core.indices.time import Time
    from ...core.measures.probability_measure import ProbabilityMeasure
    from ...core.spaces.domain import Domain
    from ...typing.index_like import IndexLike


# TODO: Add Notes section to class docstring
class RandomWalk(StochasticProcess):
    """A class representing a random walk stochastic process.

    The constructor is not intended for direct usage. Instead, user's should call the `generate` method. See the Examples section below.

    See the Notes section below for the mathematical details.

    Examples
    --------
    Generate all trajectories of a random walk with probability p=0.75 of stepping right one unit, and 0.25 of stepping left one unit.

    >>> from math import comb
    >>> from sigalg.core import Time
    >>> from sigalg.processes import RandomWalk
    >>> T = Time.discrete(length=3)
    >>> X = RandomWalk.generate(mode="enum", p=0.75, index=T)
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

    >>> print(X[3].pushforward())  # doctest: +NORMALIZE_WHITESPACE
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

    _repr_name = "RandomWalk"
    _str_name = "Random walk"

    # --------------------- constructors --------------------- #

    @generator
    def generate(
        cls,
        p: Real,
        initial_state: Real = 0,
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        index: Time | IndexLike | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> dict[str, object]:
        """Generate trajectories of the random walk by either exhaustive enumeration or Monte Carlo simulation.

        Parameters
        ----------
        p : Real
            The probability of stepping "to the right".
        initial_state : Real, default=0
            The initial state of the random walk.
        mode : Literal["enum", "sim"], default="sim"
            Whether to generate trajectories by exhaustive enumeration or Monte Carlo simulation.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. If the generation mode is set to `enum`, this parameter is ignored.
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
            If `p` is not a `Real` or `initial_state` is not a `Real`.
        ValueError
            If `p` is not between 0 and 1.

        Returns
        -------
        info : dict[str, object]
            A dictionary containing the parameters `p` and `initial_state` of the random walk.

        Examples
        --------
        Generate all length-3 trajectories of a random walk with a probability of 0.75 of stepping right.

        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=3)
        >>> X = RandomWalk.generate(
        ...     mode="enum",
        ...     p=0.75,
        ...     index=T,
        ... )
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

        Simulate ten length-3 trajectories of a random walk that begins at 2 and has probability 0.4 of stepping right.

        >>> Y = RandomWalk.generate(
        ...     mode="sim",
        ...     p=0.4,
        ...     initial_state=2,
        ...     n_trajectories=10,
        ...     index=T,
        ...     name="Y",
        ...     random_state=42,
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'Y':
        time    0  1  2  3
        sample
        0       2  3  2  3
        1       2  3  2  3
        2       2  3  4  3
        3       2  1  0  1
        4       2  3  4  3
        5       2  1  0 -1
        6       2  3  4  5
        7       2  1  2  3
        8       2  3  2  1
        9       2  1  0  1
        """
        if not isinstance(p, Real):
            raise TypeError("p must be a real number.")
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1.")
        if not isinstance(initial_state, Real):
            raise TypeError("initial_state must be a real number.")

        return {"p": p, "initial_state": initial_state}

    # --------------------- properties --------------------- #

    @property
    def p(self) -> Real:
        """Get the `p` parameter of the random walk.

        The `p` parameter is settable. See the Examples below.

        Returns
        -------
        p : Real
            The probability of stepping "to the right" in the random walk.

        Examples
        --------
        Generate a symmetric random walk (i.e., with `p=0.5`) and print its trajectories and probability measure.

        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=0, index=T)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    1  2  3
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        >>> print(X.measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0              0.25
        1              0.25
        2              0.25
        3              0.25

        Set the `p` parameter to `0.7`, regenerate the trajectories, and print the new probability measure.

        >>> X.p = 0.7
        >>> X.regenerate()
        >>> print(X.measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0              0.09
        1              0.21
        2              0.21
        3              0.49
        """
        return self._p

    @p.setter
    def p(self, value: Real) -> None:
        """Set the `p` parameter of the random walk.

        Parameters
        ----------
        value : Real
            The new value for the `p` parameter.

        Raises
        ------
        TypeError
            If `value` is not a `Real`.
        ValueError
            If `value` is not between 0 and 1.
        """
        if not isinstance(value, Real):
            raise TypeError("p must be a real number.")
        if value < 0 or value > 1:
            raise ValueError("p must be between 0 and 1.")
        self._p = value
        self._erase_generated_data()

    @property
    def initial_state(self) -> Real:
        """Get the initial state of the random walk.

        The `initial_state` parameter is settable. See the Examples below.

        Returns
        -------
        initial_state : Real
            The initial state of the random walk.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: Real) -> None:
        """Set the initial state of the random walk.

        Parameters
        ----------
        value : Real
            The new initial state of the random walk.

        Raises
        ------
        TypeError
            If `value` is not a `Real`.

        Examples
        --------
        Generate a symmetric random walk (i.e., with `p=0.5`) with initial state `0` and print its trajectories.

        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=0, index=T)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    1  2  3
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2

        Set the `initial_state` parameter to `0.5`, regenerate the trajectories, and print the new trajectories.

        >>> X.initial_state = 0.5
        >>> X.regenerate()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    1    2    3
        sample
        0     0.5 -0.5 -1.5
        1     0.5 -0.5  0.5
        2     0.5  1.5  0.5
        3     0.5  1.5  2.5
        """
        if not isinstance(value, Real):
            raise TypeError("initial_state must be a real number.")
        self._initial_state = value
        self._erase_generated_data()

    # --------------------- enumeration methods --------------------- #

    def _enumeration_subclass_hook(self) -> pd.DataFrame:
        """Hook for enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """  # noqa: D401
        from scipy.stats import bernoulli

        from ...core.functions.random_variable import RandomVariable
        from .iid_process import IIDProcess

        if len(self.time) == 1:
            return pd.DataFrame(data=[self.initial_state], columns=self.time.data)

        step_indicators = IIDProcess.generate(
            mode="enum",
            distribution=bernoulli(p=self.p),
            support=[0, 1],
            index=self.time[1:],
            name="step_indicators",
        )
        self.step_indicators = step_indicators

        displacements = (2 * step_indicators - 1).with_name("displacements")
        initial_state = RandomVariable.from_constant(
            *step_indicators.measure_space, constant=0
        )

        S = (
            displacements.cumsum(name="S").insert_rv(
                rv=initial_state,
                time=self.time[0],
            )
            + self.initial_state
        )

        return S.data

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
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=3)
        >>> X = RandomWalk.generate(mode="enum", p=0.75, index=T)
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
        return self.step_indicators._generate_exact_prob_measure(domain)

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        """Generate simulated data for the random walk.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        from scipy.stats import bernoulli

        from ...core.functions.random_variable import RandomVariable
        from .iid_process import IIDProcess

        if len(self.time) == 1:
            return pd.DataFrame(data=[self.initial_state], columns=self.time.data)

        step_indicators = IIDProcess.generate(
            mode="sim",
            distribution=bernoulli(p=self.p),
            support=[0, 1],
            index=self.time[1:],
            name="step_indicators",
            n_trajectories=self.n_trajectories,
            random_state=self.random_state,
        )

        displacements = (2 * step_indicators - 1).with_name("displacements")
        initial_state = RandomVariable.from_constant(
            *step_indicators.measure_space, constant=0
        )

        S = (
            displacements.cumsum(name="S").insert_rv(
                rv=initial_state,
                time=self.time[0],
            )
            + self.initial_state
        )

        return S.data

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the random walk.

        Returns
        -------
        repr_str : str
            The string representation of the random walk.
        """
        if self.data is None:
            return type(self)._repr_name + "(empty)"
        if self.measure is not None:
            return (
                type(self)._repr_name + f"(domain={self.domain.name}, "
                f"sig_alg={self.sig_alg.name}, "
                f"measure={self.measure.name}, "
                f"p={self.p}, "
                f"initial_state={self.initial_state}, "
                f"name={self.name})"
            )
