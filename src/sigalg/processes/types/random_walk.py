"""Random walk module."""

from collections.abc import Hashable
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import bernoulli

from ...core.base.index import Index
from ...core.base.sample_space import SampleSpace
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ...core.random_objects.random_vector import RandomVector
from ..base.stochastic_process import StochasticProcess


class RandomWalk(StochasticProcess):
    """A class representing a random walk stochastic process.

    Parameters
    ----------
    p : Real
        The probability that the particle takes a step to the right, so `1-p` is the probability that it steps left. Must be between `0` and `1`.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    index : Index | None, default=None
        The index of the stochastic process. If `None`, it will be generated later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.

    Raises
    ------
    TypeError
        If `p` is not a real number between `0` and `1`.
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        p: Real,
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
    ) -> None:
        if not isinstance(p, Real) or (p < 0 or p > 1):
            raise TypeError("p must be a real number between 0 and 1.")

        super().__init__(
            domain=domain,
            index=index,
            name=name,
        )

        self.p = p
        self._is_discrete_state = True

    # --------------------- data generation methods --------------------- #

    def _simulation_logic(
        self,
        max_trajectories: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Generate simulated data for the random walk.

        Parameters
        ----------
        max_trajectories : int
            The maximum number of trajectories to simulate.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        from .iid_process import IIDProcess

        step_indicators = IIDProcess(
            distribution=bernoulli(p=self.p),
            name="step_indicators",
        ).from_simulation(
            n_trajectories=max_trajectories,
            length=len(self.time),
            random_state=random_state,
        )

        displacements = (2 * step_indicators - 1).with_name("displacements")
        initial_state = RandomVector(
            domain=step_indicators.domain, name=0
        ).from_constant(0)
        S = displacements.cumsum(name="S").add_initial_state(initial_state)
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
        rv = bernoulli(p=self.p)

        displacements = self.data.diff(axis=1)

        return displacements

        # new_step_indicator = ((new_displacement + 1) / 2).with_name(
        #     "new_step_indicator"
        # )

        # element_wise_probabilities = rv.pmf(self.data.values)
        # probabilities = pd.Series(
        #     data=np.prod(element_wise_probabilities, axis=1),
        #     index=self.domain.data,
        # )
        # return ProbabilityMeasure(sample_space=self.domain, name=name).from_pandas(
        #     probabilities
        # )

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated random walk" if self.is_enumerated else "Random walk"
        return f"{prefix} process '{self.name}'"
