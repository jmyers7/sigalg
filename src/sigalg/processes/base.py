from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ..core.featurized_spaces.featurized_probability_space import (
    FeaturizedProbabilitySpace,
)
from ..core.featurized_spaces.featurized_sample_space import FeaturizedSampleSpace
from ..core.featurized_spaces.sample_point_features import SamplePointFeatures
from ..core.random_objects.random_variable import RandomVariable
from ..core.spaces.probability_space import ProbabilitySpace
from ..core.spaces.sample_space import SampleSpace

__all__ = ["StochasticProcess", "ProcessTrajectories", "Trajectory"]


class ProcessTrajectoriesMethods:

    @property
    def trajectory_at(self):
        return self.process_trajectories.trajectory_at

    @property
    def rv_at(self):
        return self.process_trajectories.rv_at


class StochasticProcess(ABC, ProcessTrajectoriesMethods):

    # --------------------- generation methods --------------------- #

    @abstractmethod
    def _simulate(self):
        pass

    def _generate_trajectories(self):
        self._sampled_trajectories = self._simulate()
        prob_series = self._sampled_trajectories.apply(
            lambda row: tuple(row), axis=1
        ).value_counts(normalize=True)
        self._n_trajectories = len(prob_series)
        sample_space = SampleSpace(
            indices=[f"omega{i}" for i in range(self._n_trajectories)]
        )
        probabilities = dict(zip(sample_space.values, prob_series.values))
        probability_space = ProbabilitySpace(
            sample_space=sample_space, probabilities=probabilities
        )
        feature_index = list(
            range(
                self._initial_time,
                self._initial_time + self._length,
            )
        )
        fss = FeaturizedSampleSpace(
            features=pd.DataFrame(prob_series.index.tolist()),
            feature_index=feature_index,
        )
        fss._values.columns.name = "time"
        fss._values.index.name = "trajectory"
        self._process_trajectories = ProcessTrajectories(
            probability_space=probability_space,
            fss=fss,
            name=self.name,
        )
        self._probability_measure = probability_space.probability_measure

    # --------------------- properties --------------------- #

    @property
    def process_trajectories(self):
        if self._process_trajectories is None:
            self._generate_trajectories()
        return self._process_trajectories

    @property
    def n_trajectories(self):
        if self._process_trajectories is None:
            self._generate_trajectories()
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
        if self._process_trajectories is None:
            self._generate_trajectories()
        return self.process_trajectories.feature_index

    @property
    def probability_measure(self):
        if self._process_trajectories is None:
            self._generate_trajectories()
        return self._probability_measure

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return (
            f"Stochastic process '{self.name}'\n"
            f"Number of trajectories: {self.n_trajectories}\n"
            f"Length of each trajectory: {self.length}\n"
            f"Initial time: {self.initial_time}\n\n"
            f"{self.process_trajectories}"
        )

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
        super().__init__(features=features, overwrite_default_feature_index=False)
        self._values.index.name = "time"
        self._values.name = self.sample_index

    @property
    def value_at(self):
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key):
            if key not in self.parent.feature_index:
                raise ValueError(f"Time {key} not in trajectory time index")
            return self.parent.values[key]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        series_repr = repr(self._values)
        lines = series_repr.split("\n")
        data_lines = [
            line
            for line in lines
            if not line.startswith(("Name:", "Length:", "dtype:"))
        ]
        data_str = "\n".join(data_lines)
        return (
            f"Trajectory '{self.sample_index}'\n"
            f"Length: {len(self)}\n\n"
            f"{data_str}"
        )


class ProcessTrajectories(FeaturizedProbabilitySpace):

    def __init__(self, probability_space, fss, name):
        super().__init__(
            probability_space=probability_space,
            fss=fss,
        )
        self._name = name
        self._values.index.name = "trajectory"
        self._values.columns.name = "time"

    # --------------------- data access methods --------------------- #

    @property
    def trajectory_at(self):
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key):
            features = self.parent._values.iloc[key]
            return Trajectory(features=features)

    @property
    def rv_at(self):
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, time):
            if time not in self.parent.feature_index:
                raise ValueError(f"Time {time} not in process time index")
            values = self.parent.values[time]
            rv = RandomVariable.from_values(
                values=values,
                probability_space=self.parent.probability_space,
                name=f"{self.parent._name}{time}",
            )
            rv._values.index.name = "trajectory"
            return rv

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"{self._values}"
