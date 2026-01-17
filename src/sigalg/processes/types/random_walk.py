"""Random walk module."""

from collections.abc import Hashable
from numbers import Real

import pandas as pd
from scipy.stats import bernoulli

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ...core.random_objects.random_vector import RandomVector
from ..base.stochastic_process import StochasticProcess
from .iid_process import IIDProcess


class RandomWalk(StochasticProcess):
    """A class representing a random walk stochastic process.

    Parameters
    ----------
    p : Real
        The probability that the particle takes a step to the right, so `1-p` is the probability that it steps left. Must be between `0` and `1`.
    initial_state : int, default=0
        The initial state of the random walk at the first time point. Must be an integer.
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
    >>> X = RandomWalk(p=0.75, name="X", is_discrete_time=True).from_enumeration(length=4)
    >>> # Print the trajectories and their probabilities
    >>> X.range.print_trajectories_and_probabilities() # doctest: +NORMALIZE_WHITESPACE
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
    >>> X.at[3].range.print_values_and_probabilities() # doctest: +NORMALIZE_WHITESPACE
            X_3  probability
    output
    x_3_0    -3     0.015625
    x_3_1    -1     0.140625
    x_3_2     1     0.421875
    x_3_3     3     0.421875
    >>> # Print binomial probabilities and note they match the law of X_3
    >>> for k in range(4):
    ...     print(comb(3, k) * (0.75**k) * (0.25**(3-k)))
    0.015625
    0.140625
    0.421875
    0.421875
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        p: Real,
        initial_state: int = 0,
        time: Time | None = None,
        is_discrete_time: bool | None = None,
        domain: SampleSpace | None = None,
        name: Hashable | None = "X",
    ) -> None:
        if not isinstance(p, Real) or (p < 0 or p > 1):
            raise TypeError("p must be a real number between 0 and 1.")
        if not isinstance(initial_state, int):
            raise TypeError("initial_state must be an integer.")

        super().__init__(
            domain=domain,
            time=time,
            is_discrete_time=is_discrete_time,
            is_discrete_state=True,
            name=name,
        )

        self.p = p
        self.initial_state = initial_state

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
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
        if len(self.time) == 1:
            return pd.DataFrame(data=[self.initial_state], columns=self.time.data)

        initial_time = self.time[0]
        step_times = Time().from_pandas(self.time.data[1:])
        step_times.is_discrete = self.time.is_discrete

        step_indicators = IIDProcess(
            distribution=bernoulli(p=self.p),
            support=[0, 1],
            time=step_times,
            name="step_indicators",
        ).from_enumeration()
        self.step_indicators = step_indicators

        displacements = (2 * step_indicators - 1).with_name("displacements")

        initial_state = RandomVector(
            domain=step_indicators.domain, name=initial_time
        ).from_constant(0)

        S = (
            displacements.cumsum(name="S").add_initial_state(initial_state)
            + self.initial_state
        )

        return S.data

    def _simulation_logic(
        self,
        n_trajectories: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Generate simulated data for the random walk.

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories to simulate.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        if len(self.time) == 1:
            return pd.DataFrame(data=[self.initial_state], columns=self.time.data)

        initial_time = self.time[0]
        step_times = Time().from_pandas(self.time.data[1:])
        step_times.is_discrete = self.time.is_discrete

        step_indicators = IIDProcess(
            distribution=bernoulli(p=self.p),
            time=step_times,
            name="step_indicators",
        ).from_simulation(
            n_trajectories=n_trajectories,
            random_state=random_state,
        )

        displacements = (2 * step_indicators - 1).with_name("displacements")

        initial_state = RandomVector(
            domain=step_indicators.domain, name=initial_time
        ).from_constant(0)

        S = (
            displacements.cumsum(name="S").add_initial_state(initial_state)
            + self.initial_state
        )

        return S.data

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
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
        return self.step_indicators._generate_exact_prob_measure(name=name)

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated random walk" if self.is_enumerated else "Random walk"
        return f"{prefix} process '{self.name}'"
