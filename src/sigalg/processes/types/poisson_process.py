"""Poisson process module."""

from collections.abc import Hashable
from numbers import Real

import pandas as pd
from scipy.stats import expon

from sigalg.processes.types.iid_process import IIDProcess

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ..base.stochastic_process import StochasticProcess


class PoissonProcess(StochasticProcess):
    """A class representing a Poisson process.

    The Poisson process is a process `{X_t}` where `X_t` counts the number of events that have occurred by time `t`. The `rate` parameter represents the average number of events per unit time.

    In this implementation, trajectories are simulated until one trajectory reaches the specified `max_count` of events, and then the (required) user-provided time index is truncated to the length of this shortest complete trajectory.

    If `t_stop` is the last time value in the time index, then a good choice for `max_count` is approximately `rate * t_stop + 3 * sqrt(rate * t_stop)`, which is the mean of `X_{rate * t_stop}` (a Poisson random variable) plus 3 times its standard deviation.

    The trajectories of Poisson processes are right-continuous step functions that jump by 1 at each event time. In order to plot these trajectories accurately, the user should select a continuous time index with a sufficiently large number of points.

    The `from_enumeration` method is not implemented for `PoissonProcess` since it is a continuous-time process.

    Parameters
    ----------
    rate : Real
        The rate (lambda) of the Poisson process, which must be a positive real number.
    max_count : int
        The maximum count of events to simulate, which must be a positive integer.
    time : Time
        The time index of the stochastic process.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.

    Raises
    ------
    TypeError
        If `rate` is not a positive real number or if `max_count` is not a positive integer.

    Examples
    --------
    >>> from math import ceil, sqrt
    >>> from scipy.stats import poisson
    >>> from sigalg.core import Time
    >>> from sigalg.processes import PoissonProcess
    >>> # Parameters for the continuous time index. We select a very coarse time grid for printing purposes in the docstrings.
    >>> start = 0.0
    >>> stop = 6.25
    >>> num_points = 5
    >>> time = Time.continuous(
    ...     start=start,
    ...     stop=stop,
    ...     num_points=num_points,
    ... )
    >>> # Parameters for the Poisson process. The max_count parameter follows the suggested rule of thumb described above.
    >>> rate = 9.5
    >>> max_count = ceil(rate * stop + 3 * sqrt(rate * stop))
    >>> max_count
    83
    >>> # Simulate 10 trajectories of the Poisson process with the specified parameters and print them.
    >>> X = PoissonProcess(rate=rate, max_count=max_count, time=time).from_simulation(
    ...     n_trajectories=10, random_state=42
    ... )
    >>> X # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    time        0.0000  1.5625  3.1250  4.6875  6.2500
    trajectory
    0              0.0    11.0    32.0    54.0    64.0
    1              0.0    18.0    33.0    51.0    64.0
    2              0.0    15.0    29.0    45.0    60.0
    3              0.0    16.0    40.0    53.0    73.0
    4              0.0    11.0    21.0    35.0    46.0
    5              0.0     9.0    24.0    37.0    47.0
    6              0.0    15.0    26.0    48.0    63.0
    7              0.0     9.0    20.0    29.0    42.0
    8              0.0    13.0    29.0    38.0    62.0
    9              0.0     9.0    18.0    32.0    52.0
    >>> # Simulate a Poisson process using 50,000 trajectories
    >>> Y = PoissonProcess(
    ...     rate=rate, max_count=max_count, time=time, name="Y"
    ... ).from_simulation(n_trajectories=50_000, random_state=42)
    >>> # Extract the simulated values of the final random variable Y_last
    >>> final_counts = Y.last_rv.range
    >>> simulated_outputs = final_counts.data
    >>> # Extract the empirical probabilities of the final random variable Y_last
    >>> simulated_probabilities = final_counts.probability_measure.data
    >>> # Get the final time point, compute the theoretical probabilities of the final random variable Y_last, a Poisson random variable
    >>> final_time = Y.time[-1]
    >>> theoretical_probabilities = poisson(mu=rate * final_time).pmf(simulated_outputs)
    >>> # Compare the simulated probabilities with the theoretical probabilities
    >>> round(float(abs(simulated_probabilities - theoretical_probabilities).sum()), 4)
    0.0169
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        rate: Real,
        max_count: int,
        time: Time,
        domain: SampleSpace | None = None,
        name: Hashable | None = "X",
    ) -> None:
        if not isinstance(rate, Real) or rate <= 0:
            raise TypeError("rate must be a positive real number.")
        if not isinstance(max_count, int) or max_count <= 0:
            raise TypeError("max_count must be a positive integer.")
        self.rate = rate
        self.max_count = max_count

        super().__init__(
            domain=domain,
            time=time,
            is_discrete_state=True,
            is_discrete_time=False,
            name=name,
        )

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
        """Not implemented for PoissonProcess."""
        raise NotImplementedError("Not implemented for PoissonProcess.")

    def _simulation_logic(
        self, n_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        """Simulate Poisson process trajectories."""
        counts = Time.discrete(
            length=self.max_count, start=1, data_name="count", name=None
        )

        interarrival_times = IIDProcess(
            distribution=expon(scale=1 / self.rate),
            name="interarrival_times",
            time=counts,
        ).from_simulation(n_trajectories=n_trajectories, random_state=random_state)

        arrival_times = interarrival_times.cumsum().with_name("arrival_times")

        shortest_complete_arrival_time = arrival_times.data.iloc[:, -1].min()
        self._index = Time().from_pandas(
            data=self.time.data[self.time <= shortest_complete_arrival_time + 1e-3]
        )
        self._index.is_discrete = False

        poisson = arrival_times.to_counting_process(
            time=self._index,
        ).with_name("poisson")
        trajectories = poisson.data

        return trajectories

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        """Not implemented for PoissonProcess."""
        raise NotImplementedError(
            "Exact probability measure generation is not implemented for PoissonProcess."
        )

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        return f"Poisson process '{self.name}'"
