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
    """A class representing a Brownian motion."""

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

        increments = IIDProcess(
            distribution=norm(loc=0.0, scale=np.sqrt(dt)),
            time=self.time[1:],
            name="increments",
        ).from_simulation(n_trajectories=n_trajectories, random_state=random_state)

        initial_value = RandomVariable(
            domain=increments.domain, name=initial_time
        ).from_constant(0.0)
        increments.add_initial_state(initial_value)

        return increments.cumsum().data

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
