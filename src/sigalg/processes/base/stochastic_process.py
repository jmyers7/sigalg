from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ...core import FeaturizedProbabilitySpace
from .trajectories import TrajectoriesMethods

if TYPE_CHECKING:
    from ...core import FeaturizedProbabilitySpace, RandomVariable
    from ...core.base.time import Time
    from .trajectories import Trajectories


class StochasticProcess(FeaturizedProbabilitySpace, TrajectoriesMethods):

    # --------------------- properties --------------------- #

    @property
    def trajectories(self) -> Trajectories:
        return self.feature_embedding

    @property
    def time(self) -> Time:
        return self.trajectories.time

    @property
    def initial_time(self) -> int:
        return self.time.data[0]

    @property
    def name(self) -> str:
        if not hasattr(self, "_name"):
            raise AttributeError("name attribute not set.")
        else:
            return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name
        self.trajectories._name = name

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectories)

    # # --------------------- data access methods --------------------- #

    @property
    def rv_at(self):
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, stochastic_process):
            self.stochastic_process = stochastic_process

        def __getitem__(self, time) -> RandomVariable:
            from ...core.random_objects.random_variable import RandomVariable

            if time not in self.stochastic_process.time:
                raise ValueError(f"Time {time} not in process time index")
            values = self.stochastic_process.trajectories.values[time]
            rv = RandomVariable(
                values=values, name=f"{self.stochastic_process.trajectories.name}{time}"
            )
            rv.add_probability_measure_to_domain(
                self.stochastic_process.probability_measure
            )
            # rv = RandomVariable.from_values(
            #     values=values,
            #     probability_space=self.stochastic_process.probability_space,
            #     name=f"{self.stochastic_process.trajectories.name}{time}",
            # )
            rv.values.index.name = "trajectory"
            return rv

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return (
            "StochasticProcess("
            f"type={self.__class__.__name__}, "
            f"name='{self._name}', "
            f"initial_time={self.initial_time}, "
            f"n_trajectories={self.n_trajectories}, "
        )

    def __str__(self) -> str:
        header = f"Stochastic Process '{self._name}'"
        separator = "=" * len(header)
        result = (
            header
            + "\n"
            + separator
            + f"\n\n* Type: {self.__class__.__name__}"
            + f"\n* Initial time: {self.initial_time}"
            + f"\n* Number of trajectories: {self.n_trajectories}"
        )
        if self._enumerate:
            result += f"\n* Trajectories:\n\n{self.trajectories.values}"
        return result

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        if not isinstance(other, StochasticProcess):
            return False
        return super().__eq__(other)

    # --------------------- plotting methods --------------------- #

    def plot_trajectories(
        self,
        ax: Axes = None,
        colors: list = None,
        plot_kwargs: dict = None,
        x_label: str = "time",
        y_label: str = "state",
        title: str = None,
    ):
        columns = self.time.data
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
                ax.plot(columns, row.data, color=colors[i], **plot_kwargs)
            else:
                ax.plot(columns, row.data, **plot_kwargs)

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

    def _integer_check(self, values):
        """Utility method for `plot_trajectories`."""
        try:
            return np.allclose(values, np.round(values))
        except (TypeError, AttributeError):
            return False

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_general_parameters(fps: FeaturizedProbabilitySpace) -> None:
        from ...core.featurized_spaces.featurized_probability_space import (
            FeaturizedProbabilitySpace,
        )
        from .trajectories import Trajectories

        if not isinstance(fps, FeaturizedProbabilitySpace):
            raise TypeError("fps must be a FeaturizedProbabilitySpace object.")
        if not isinstance(fps.feature_embedding, Trajectories):
            raise TypeError("fps.feature_embedding must be a Trajectories object.")
