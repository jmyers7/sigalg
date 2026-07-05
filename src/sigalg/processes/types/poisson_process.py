"""A class representing a Poisson stochastic process."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ...core.base.index import Index


class PoissonProcess(StochasticProcess):
    """A class representing a Poisson stochastic process.

    The constructor is not intended for direct usage. Instead, user's should call the class method `from_simulation`. See the Examples section below.

    See also the Notes section below for the mathematical details.

    Examples
    --------
    >>> from math import ceil, sqrt
    >>> from scipy.stats import poisson
    >>> from sigalg.core import Time
    >>> from sigalg.processes import PoissonProcess

    Parameters for the continuous time index. We select a very coarse time grid for printing purposes in the docstrings.

    >>> start = 0.0
    >>> stop = 6.25
    >>> num_points = 5
    >>> time = Time.continuous(
    ...     start=start,
    ...     stop=stop,
    ...     num_points=num_points,
    ... )

    Parameters for the Poisson process. The max_count parameter follows the suggested rule of thumb described above.

    >>> rate = 9.5
    >>> max_count = ceil(rate * stop + 3 * sqrt(rate * stop))

    Simulate 10 trajectories of the Poisson process with the specified parameters and print them.

    >>> X = PoissonProcess.from_simulation(
    ...     rate=rate,
    ...     max_count=max_count,
    ...     index=time,
    ...     n_trajectories=10,
    ...     random_state=42,
    ... )
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Poisson process 'X':
    time    0.0000  1.5625  3.1250  4.6875  6.2500
    sample
    0          0.0    11.0    32.0    54.0    64.0
    1          0.0    17.0    32.0    50.0    63.0
    2          0.0    14.0    27.0    44.0    62.0
    3          0.0    23.0    42.0    60.0    75.0
    4          0.0    11.0    20.0    37.0    45.0
    5          0.0    11.0    25.0    37.0    54.0
    6          0.0    14.0    33.0    48.0    60.0
    7          0.0     9.0    19.0    28.0    42.0
    8          0.0    19.0    26.0    41.0    62.0
    9          0.0     7.0    21.0    37.0    55.0

    Simulate a Poisson process using 50,000 trajectories.

    >>> Y = PoissonProcess.from_simulation(
    ...     rate=rate,
    ...     max_count=max_count,
    ...     index=time,
    ...     n_trajectories=50_000,
    ...     random_state=42,
    ...     name="Y",
    ... )

    Extract the simulated values of the final random variable and their empirical probabilities.

    >>> simulated_outputs = Y.last_rv.range.sample_space
    >>> simulated_probabilities = Y.last_rv.range.prob_measure

    Get the final time point, compute the theoretical probabilities of the final random variable and compare to the empirical ones.

    >>> final_time = Y.time[-1]
    >>> theoretical_probabilities = poisson(mu=rate * final_time).pmf(simulated_outputs)
    >>> round(float(abs(simulated_probabilities.data - theoretical_probabilities).sum()), 4)
    0.02

    Notes
    -----
    The Poisson process is a process `{X_t}` where `X_t` counts the number of events that have occurred by time `t`. The `rate` parameter represents the average number of events per unit time.
    """

    _repr_name = "Poisson process"

    # --------------------- simulation methods --------------------- #

    @classmethod
    def from_simulation(
        cls,
        rate: Real,
        max_count: int,
        n_trajectories: int,
        index: Index | None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Simulate trajectories of the Poisson process.

        In this implementation, trajectories are simulated until one trajectory reaches the specified `max_count` of events, and then the (required) user-provided index is truncated to the length of this shortest complete trajectory.

        If `t_stop` is the last time value in the time index, then a good choice for `max_count` is approximately `rate * t_stop + 3 * sqrt(rate * t_stop)`, which is the mean of `X_{rate * t_stop}` (a Poisson random variable) plus 3 times its standard deviation.

        The trajectories of Poisson processes are right-continuous step functions that jump by 1 at each event time. In order to plot these trajectories accurately, the user should select a continuous time index with a sufficiently large number of points.

        Parameters
        ----------
        rate : Real
            The rate (lambda) of the Poisson process, which must be a positive real number.
        max_count : int
            The maximum count of events to simulate, which must be a positive integer.
        n_trajectories : int
            The number of trajectories to simulate.
        index : Index | None, default=None
            The index of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `rate` is not a `Real`, or if `max_count` is not an integer.
        ValueError
            If either `rate` or `max_cont` is negative.

        Examples
        --------
        >>> from math import ceil, sqrt
        >>> from scipy.stats import poisson
        >>> from sigalg.core import Time
        >>> from sigalg.processes import PoissonProcess

        Parameters for the continuous time index. We select a very coarse time grid for printing purposes in the docstrings.

        >>> start = 0.0
        >>> stop = 6.25
        >>> num_points = 5
        >>> time = Time.continuous(
        ...     start=start,
        ...     stop=stop,
        ...     num_points=num_points,
        ... )

        Parameters for the Poisson process. The max_count parameter follows the suggested rule of thumb described above.

        >>> rate = 9.5
        >>> max_count = ceil(rate * stop + 3 * sqrt(rate * stop))

        Simulate 10 trajectories of the Poisson process with the specified parameters and print them.

        >>> X = PoissonProcess.from_simulation(
        ...     rate=rate,
        ...     max_count=max_count,
        ...     index=time,
        ...     n_trajectories=10,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Poisson process 'X':
        time    0.0000  1.5625  3.1250  4.6875  6.2500
        sample
        0          0.0    11.0    32.0    54.0    64.0
        1          0.0    17.0    32.0    50.0    63.0
        2          0.0    14.0    27.0    44.0    62.0
        3          0.0    23.0    42.0    60.0    75.0
        4          0.0    11.0    20.0    37.0    45.0
        5          0.0    11.0    25.0    37.0    54.0
        6          0.0    14.0    33.0    48.0    60.0
        7          0.0     9.0    19.0    28.0    42.0
        8          0.0    19.0    26.0    41.0    62.0
        9          0.0     7.0    21.0    37.0    55.0
        """
        if not isinstance(rate, Real):
            raise TypeError("rate must be a real number.")
        if rate <= 0:
            raise ValueError("rate must be positive.")
        if not isinstance(max_count, int):
            raise TypeError("max_count must be an integer.")
        if max_count <= 0:
            raise ValueError("max_count must be positive.")

        index = cls._validate_and_return_index(index=index, length=None)
        random_state = cls._validate_simulation_parameters_and_return_random_state(
            n_trajectories=n_trajectories, random_state=random_state
        )
        process = cls(index=index, name=name)

        process.n_trajectories = n_trajectories
        process.random_state = random_state
        process.rate = rate
        process.max_count = max_count

        return process._simulation_logic()

    def _simulation_hook(self) -> pd.DataFrame:
        """Generate simulated data for the Poisson process.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        from scipy.stats import expon

        from ...core.base.time import Time
        from .iid_process import IIDProcess

        counts = Time.discrete(
            length=self.max_count, start=1, variable_name="count", name=None
        )

        interarrival_times = IIDProcess.from_simulation(
            distribution=expon(scale=1 / self.rate),
            name="interarrival_times",
            index=counts,
            n_trajectories=self.n_trajectories,
            random_state=self.random_state,
        )

        arrival_times = interarrival_times.cumsum().with_name("arrival_times")

        shortest_complete_arrival_time = arrival_times.data.iloc[:, -1].min()
        self._index = Time(
            self.time.data[self.time <= shortest_complete_arrival_time + 1e-3]
        )

        poisson = arrival_times.to_counting_process(
            time=self._index,
        ).with_name("poisson")
        trajectories = poisson.data

        return trajectories
