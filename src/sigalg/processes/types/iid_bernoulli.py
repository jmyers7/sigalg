import numpy as np
import pandas as pd
from scipy.stats import bernoulli

from ..base import StochasticProcess


class IIDBernoulli(StochasticProcess):

    def __init__(
        self,
        probability,
        max_trajectories=1000,
        length=10,
        initial_time=0,
        name="X",
        random_state=None,
    ):
        self._probability = probability
        self._max_trajectories = max_trajectories
        self._length = length
        self._initial_time = initial_time
        self._name = name
        self._process_trajectories = None
        self._random_state = random_state

    @property
    def probability(self):
        return self._probability

    def _simulate(self):
        rng = np.random.default_rng(self._random_state)
        simulated_trajectories = bernoulli.rvs(
            p=self._probability,
            size=(self._max_trajectories, self._length),
            random_state=rng,
        )
        time_index = list(range(self._initial_time, self._length + self._initial_time))
        return pd.DataFrame(data=simulated_trajectories, columns=time_index)

    def _plot_title(self):
        return "IID Bernoulli process"
