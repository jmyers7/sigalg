"""Brownian motion module."""

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.stats import norm

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ...core.random_objects.random_variable import RandomVariable
from ..base.stochastic_process import StochasticProcess
from .iid_process import IIDProcess


class BrownianMotion(StochasticProcess):
    """A class representing a Brownian motion.

    Parameters
    ----------
    time : Time | None, default=None
        The time index of the stochastic process. If `None`, then it will be generated in the `from_simulation` method.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, then it will be generated in the `from_simulation` method.
    name : Hashable | None, default="X"
        The name of the stochastic process.

    Examples
    --------
    >>> from sigalg.core import Time
    >>> from sigalg.processes import BrownianMotion
    >>> T = Time.continuous(start=0.1, stop=1.1, dt=0.35)
    >>> X = BrownianMotion(time=T).from_simulation(n_trajectories=4, random_state=42)
    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    time        0.100000  0.433333  0.766667  1.100000
    trajectory
    0                0.0  0.175928 -0.424507  0.008767
    1                0.0  0.543035 -0.583395 -1.335209
    2                0.0  0.073809 -0.108774 -0.118474
    3                0.0 -0.492505  0.015216  0.464274
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: SampleSpace | None = None,
        time: Time | None = None,
        name: Hashable | None = "X",
    ) -> None:
        super().__init__(
            time=time,
            is_discrete_time=False,
            domain=domain,
            is_discrete_state=False,
            name=name,
        )

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
        """Not implemented for BrownianMotion."""
        raise NotImplementedError("Not implemented for BrownianMotion.")

    def _simulation_logic(
        self, n_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        """Simulate Brownian motion trajectories."""
        dt = self.time.data[1] - self.time.data[0]
        initial_time = self.time.data[0]
        increments_time = self.time.remove_time(pos=0)

        increments = IIDProcess(
            distribution=norm(loc=0.0, scale=np.sqrt(dt)),
            time=increments_time,
        ).from_simulation(n_trajectories=n_trajectories, random_state=random_state)

        initial_value = RandomVariable(
            domain=increments.domain, name=initial_time
        ).from_constant(0.0)

        return increments.insert_rv(rv=initial_value, time=initial_time).cumsum().data

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        """Not implemented for BrownianMotion."""
        raise NotImplementedError(
            "Exact probability measure generation is not implemented for BrownianMotion."
        )

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        return f"Brownian motion '{self.name}'"
