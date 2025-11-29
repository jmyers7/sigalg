from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ..core.featurized_spaces.featurized_sample_space import FeaturizedSampleSpace
from ..core.featurized_spaces.sample_point_features import SamplePointFeatures
from ..core.random_objects.random_variable import RandomVariable

__all__ = ["StochasticProcess", "ProcessTrajectories", "Trajectory"]


class StochasticProcess(ABC):

    # --------------------- properties --------------------- #

    @property
    def trajectories(self):
        if self._trajectories is None:
            self._generate()
        return self._trajectories

    @property
    def n_trajectories(self):
        return self._n_trajectories

    @property
    def length(self):
        return self._length

    @property
    def initial_time(self):
        return self._initial_time

    @property
    def name(self):
        return self._name

    @property
    def time_index(self):
        return self.trajectories.feature_index

    # --------------------- setter methods --------------------- #

    def set_name(self, name):
        if not isinstance(name, str):
            raise ValueError("Name must be a string")
        self._name = name
        return self

    # --------------------- data access methods --------------------- #

    def rv_at(self, time):
        if time not in self.trajectories.feature_index:
            raise ValueError(f"Time {time} not in process time index")
        values = self.trajectories.values[time]
        return RandomVariable.from_values(
            domain=self.trajectories.sample_space,
            values=values,
            name=f"{self._name}_{time}",
        )

    @property
    def trajectory_at(self):
        return self.trajectories.trajectory_at

    # --------------------- generation methods --------------------- #

    @abstractmethod
    def _generate(self):
        pass

    # --------------------- utility methods --------------------- #

    def _integer_check(self, values):
        return np.allclose(values, np.round(values))

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        return f"{self.__class__.__name__} trajectories"

    def plot_trajectories(
        self,
        ax: Axes = None,
        colors: list = None,
        plot_kwargs: dict = None,
        x_label: str = "time",
        y_label: str = "state",
        title: str = None,
    ):
        columns = self.trajectories.feature_index
        n_trajectories = self.n_trajectories

        if ax is None:
            _, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise ValueError("ax must be a matplotlib Axes object")

        if plot_kwargs is None:
            plot_kwargs = {}

        if colors is not None:
            if not isinstance(colors, list):
                raise ValueError("colors must be a list")
            if len(colors) == 1:
                colors = [colors[0]] * n_trajectories
            else:
                custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                if n_trajectories == 1:
                    colors = [custom_cmap(0)]
                else:
                    colors = [
                        custom_cmap(i / (n_trajectories - 1))
                        for i in range(n_trajectories)
                    ]

        for i, (_, row) in enumerate(self.trajectories.iter_sample_features()):
            if colors is not None:
                ax.plot(columns, row.values, color=colors[i], **plot_kwargs)
            else:
                ax.plot(columns, row.values, **plot_kwargs)

        is_time_integer = self._integer_check(columns.values)
        is_trajectory_integer = self._integer_check(
            self.trajectories.values.to_numpy().flatten()
        )
        if is_time_integer:
            time_values = columns.values.astype(int)
            if len(time_values) <= 20:
                ax.set_xticks(time_values)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if is_trajectory_integer:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title is None:
            title = self._plot_title()
        ax.set_title(title)

        return ax


class Trajectory(SamplePointFeatures):

    def __init__(self, features):
        super().__init__(features=features)


class ProcessTrajectories(FeaturizedSampleSpace):

    def __init__(self, features, feature_index):
        super().__init__(features=features, feature_index=feature_index)

    @property
    def trajectory_at(self):
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key):
            features = self.parent._values.iloc[key]
            return Trajectory(features=features)
