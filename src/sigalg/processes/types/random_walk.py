"""A class representing a random walk stochastic process."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    import numpy as np

    from ...core.base.index import Index
    from ...core.probability_measures.probability_measure import ProbabilityMeasure


# TODO: Update docstrings
class RandomWalk(StochasticProcess):
    """A class representing a random walk stochastic process.

    Parameters
    ----------
    p : Real
        The probability that the particle takes a step to the right, so `1-p` is the probability that it steps left. Must be between `0` and `1`.
    initial_state : Real, default=0
        The initial state of the random walk at the first time point.
    time : Time | None, default=None
        The time index of the stochastic process. If `None`, then the `is_discrete_time` property must be provided.
    is_discrete_time : bool | None, default=None
        Whether the stochastic process is a discrete-time process. If `None`, then `time` parameter must be provided.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.

    Raises
    ------
    TypeError
        If `p` is not a real number between `0` and `1`.

    Examples
    --------
    >>> from math import comb
    >>> from sigalg.processes import RandomWalk
    >>> # Define a random walk with probability p=0.75 of stepping right one unit, and 0.25 of stepping left one unit
    >>> time = Time.discrete(length=3)
    >>> X = RandomWalk(p=0.75, name="X", time=time).from_enumeration()
    >>> # Print the trajectories and their probabilities
    >>> X.print_trajectories_and_probabilities() # doctest: +NORMALIZE_WHITESPACE
                0  1  2  3  probability
    trajectory
    0           0 -1 -2 -3     0.015625
    1           0 -1 -2 -1     0.046875
    2           0 -1  0 -1     0.046875
    3           0 -1  0  1     0.140625
    4           0  1  0 -1     0.046875
    5           0  1  0  1     0.140625
    6           0  1  2  1     0.140625
    7           0  1  2  3     0.421875
    >>> # Print the values of the X_3 random variable and their corresponding probabilities
    >>> X.at[3].range.prob_measure # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P_X_3':
        probability
    output
    -3         0.015625
    -1         0.140625
    1          0.421875
    3          0.421875
    >>> # Print binomial probabilities and note they match the law of X_3
    >>> for k in range(4):
    ...     print(comb(3, k) * (0.75**k) * (0.25**(3-k)))
    0.015625
    0.140625
    0.421875
    0.421875
    """

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
        """Later."""
        if not isinstance(p, Real):
            raise TypeError("p must be a real number.")
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1.")
        if not isinstance(initial_state, Real):
            raise TypeError("initial_state must be a real number.")

        index = cls._validate_and_return_index(index=index, length=length)
        process = cls(index=index, name=name)

        process.p = p
        process.initial_state = initial_state

        return process._enumeration_logic()

    def _enumeration_hook(self) -> pd.DataFrame:
        """Generate the enumerated trajectories for the random walk based on the trajectory length.

        Parameters
        ----------
        **kwargs
            Not needed for Markov chain enumeration, but included for consistency with the base class.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the enumerated trajectories as rows and time points as columns.
        """
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
        """Generate the exact probability measure for the random walk process.

        Parameters
        ----------
        name : Hashable | None, default="P"
            The name of the generated probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The generated probability measure.
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
        """Later."""
        if not isinstance(p, Real):
            raise TypeError("p must be a real number.")
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1.")
        if not isinstance(initial_state, Real):
            raise TypeError("initial_state must be a real number.")

        index = cls._validate_and_return_index(index=index, length=length)
        process = cls(index=index, name=name)

        process.n_trajectories = n_trajectories
        process.random_state = random_state
        process.p = p
        process.initial_state = initial_state

        return process._simulation_logic()

    def _simulation_hook(self) -> pd.DataFrame:
        """Later."""
        from scipy.stats import bernoulli

        from ...core.random_objects.random_variable import RandomVariable
        from .iid_process import IIDProcess

        if len(self.time) == 1:
            return pd.DataFrame(data=[self.initial_state], columns=self.time.data)

        step_indicators = IIDProcess.from_simulation(
            distribution=bernoulli(p=self.p),
            support=[0, 1],
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
