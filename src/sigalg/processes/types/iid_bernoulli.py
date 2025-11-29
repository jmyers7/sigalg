from scipy.stats import bernoulli

from ..base import ProcessTrajectories, StochasticProcess


class IIDBernoulli(StochasticProcess):

    def __init__(
        self, probability, n_trajectories=1000, length=10, initial_time=0, name="X"
    ):
        self._probability = probability
        self._n_trajectories = n_trajectories
        self._length = length
        self._initial_time = initial_time
        self._name = name
        self._trajectories = None

    @property
    def probability(self):
        return self._probability

    def _generate(self):
        trajectories = bernoulli.rvs(
            p=self._probability, size=(self._n_trajectories, self._length)
        )
        time_index = list(range(self._initial_time, self._length + self._initial_time))
        self._trajectories = ProcessTrajectories(
            features=trajectories, feature_index=time_index
        )

    def _plot_title(self):
        return "IID Bernoulli process"
