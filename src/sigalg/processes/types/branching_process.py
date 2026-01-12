"""Branching process module."""

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ...core.base.index import Index
from ...core.base.sample_space import SampleSpace
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ..base.stochastic_process import StochasticProcess


class BranchingProcess(StochasticProcess):
    """A class representing a branching process."""

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        offspring_dist: rv_frozen,
        initial_population: int = 1,
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
    ) -> None:
        super().__init__(
            domain=domain,
            index=index,
            name=name,
        )
        if not isinstance(offspring_dist, rv_frozen):
            raise TypeError(
                "offspring_dist must be an instance of rv_frozen from scipy.stats."
            )

        self.offspring_dist = offspring_dist
        self.initial_population = initial_population
        self._is_discrete_state = True

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
        """Not implemented for BranchingProcess."""
        raise NotImplementedError("Not implemented for BranchingProcess.")

    def _simulation_logic(
        self, n_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        """Simulate branching process trajectories."""
        if self.offspring_dist is None:
            raise ValueError("offspring_dist must be provided for simulation")

        rng = np.random.default_rng(random_state)
        length = len(self.time)

        trajectories = np.zeros((n_trajectories, length), dtype=int)
        trajectories[:, 0] = self.initial_population

        for t in range(1, length):
            for i in range(n_trajectories):
                current_pop = trajectories[i, t - 1]
                if current_pop == 0:
                    trajectories[i, t] = 0
                else:
                    if current_pop > 0:
                        trajectories[i, t] = self.offspring_dist.rvs(
                            size=current_pop, random_state=rng
                        ).sum()

        return pd.DataFrame(trajectories, columns=self.time.data)

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        """Not implemented for BranchingProcess."""
        raise NotImplementedError(
            "Exact probability measure generation is not implemented for BranchingProcess."
        )

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        return f"Branching process '{self.name}'"
