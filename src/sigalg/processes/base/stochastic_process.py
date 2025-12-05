from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from ...core.featurized_spaces.featurized_probability_space import (
        FeaturizedProbabilitySpace,
    )
    from ...core.probability_measures.probability_measure import ProbabilityMeasure
    from ...core.random_objects.random_variable import RandomVariable
    from .trajectories import Trajectories
    from .trajectory import Trajectory


class StochasticProcess(ABC):

    def __init__(
        self,
        *,
        max_trajectories: int = 1000,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        self._validate_general_parameters(
            max_trajectories=max_trajectories,
            length=length,
            initial_time=initial_time,
            name=name,
            random_state=random_state,
            enumerate=enumerate,
        )
        self._max_trajectories = max_trajectories
        self._length = length
        self._initial_time = initial_time
        self._name = name
        self._random_state = random_state
        self._enumerate = enumerate

        if self._enumerate:
            self._decide_if_enumeration_feasible()

        trajectories_df = (
            self._enumerate_raw_trajectories()
            if self._enumerate
            else self._simulate_raw_trajectories()
        )
        self._trajectories, self._fps = self._generate_trajectories(trajectories_df)
        self._sample_space = self._fps.sample_space
        self._sigma_algebra = self._fps
        self._probability_measure = self._fps.probability_measure

    # --------------------- properties --------------------- #

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def fps(self):
        return self._fps

    @property
    def time_index(self):
        return self.trajectories.time_index

    @property
    def sigma_algebra(self):
        return self._sigma_algebra

    @property
    def probability_measure(self):
        return self._probability_measure

    @property
    def n_trajectories(self):
        return len(self.trajectories)

    @property
    def max_trajectories(self) -> int:
        return self._max_trajectories

    @property
    def length(self):
        return self._length

    @property
    def initial_time(self):
        return self._initial_time

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    @property
    def enumerate(self) -> bool:
        return self._enumerate

    # --------------------- data access methods --------------------- #

    @property
    def trajectory_at(self):
        return self._TrajectoryIndexer(self)

    class _TrajectoryIndexer:
        def __init__(self, stochastic_process) -> None:
            self.stochastic_process = stochastic_process

        def __getitem__(self, key) -> Trajectory:
            from .trajectory import Trajectory

            features = self.stochastic_process.trajectories.values.iloc[key]
            return Trajectory(values=features, name=features.name)

    @property
    def rv_at(self):
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, stochastic_process):
            self.stochastic_process = stochastic_process

        def __getitem__(self, time) -> RandomVariable:
            from ...core.random_objects.random_variable import RandomVariable

            if time not in self.stochastic_process.time_index:
                raise ValueError(f"Time {time} not in process time index")
            values = self.stochastic_process.trajectories.values[time]
            rv = RandomVariable.from_values(
                values=values,
                probability_space=self.stochastic_process.fps.probability_space,
                name=f"{self.stochastic_process.trajectories.name}{time}",
            )
            rv._values.index.name = "trajectory"
            return rv

    # --------------------- generation methods --------------------- #

    @abstractmethod
    def _simulate_raw_trajectories(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _enumerate_raw_trajectories(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _compute_exact_probabilities(
        self, trajectories_df: pd.DataFrame
    ) -> ProbabilityMeasure:
        pass

    def _compute_empirical_probabilities(
        self, trajectories_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, ProbabilityMeasure]:
        from ...core.probability_measures.probability_measure import ProbabilityMeasure
        from ...core.spaces.sample_space import SampleSpace

        prob_series = trajectories_df.apply(
            lambda row: tuple(row), axis=1
        ).value_counts(normalize=True)
        sample_space_indices = [f"omega{i}" for i in range(len(prob_series))]
        sample_space = SampleSpace(indices=sample_space_indices)
        probabilities = dict(zip(sample_space, prob_series))
        df = pd.DataFrame(prob_series.index.tolist())
        probability_measure = ProbabilityMeasure(
            sample_space=sample_space,
            probabilities=probabilities,
        )
        return df, probability_measure

    def _generate_trajectories(
        self, trajectories_df: pd.DataFrame
    ) -> tuple[Trajectories, FeaturizedProbabilitySpace]:
        from ...core.featurized_spaces.featurized_probability_space import (
            FeaturizedProbabilitySpace,
        )
        from .trajectories import Trajectories

        if self._enumerate:
            probability_measure = self._compute_exact_probabilities(trajectories_df)
        else:
            trajectories_df, probability_measure = (
                self._compute_empirical_probabilities(trajectories_df)
            )

        sample_space = probability_measure.sample_space
        time_index = range(
            self._initial_time,
            self._initial_time + self._length,
        )
        trajectories_df.columns = time_index
        trajectories_df.index = sample_space
        trajectories_df.index.name = "trajectory"
        trajectories_df.columns.name = "time"

        trajectories = Trajectories(
            sample_space=sample_space, values=trajectories_df, name=self._name
        )
        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=trajectories,
            probability_measure=probability_measure,
        )

        return trajectories, fps

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return (
            self._plot_title() + " ("
            f"name={self.name}, "
            f"n_trajectories={self.n_trajectories}, "
            f"length={self.length}, "
            f"initial_time={self.initial_time})"
        )

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        if not isinstance(other, StochasticProcess):
            return False
        return (
            self.name == other.name
            and self.n_trajectories == other.n_trajectories
            and self.length == other.length
            and self.initial_time == other.initial_time
            and self.probability_measure == other.probability_measure
            and self.trajectories == other.trajectories
        )

    # --------------------- plotting methods --------------------- #

    @abstractmethod
    def _plot_title(self):
        pass

    def plot_trajectories(
        self,
        ax: Axes = None,
        colors: list = None,
        plot_kwargs: dict = None,
        x_label: str = "time",
        y_label: str = "state",
        title: str = None,
    ):
        columns = self.trajectories.time_index
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

    def _integer_check(self, values):
        """Utility method for `plot_trajectories`."""
        try:
            return np.allclose(values, np.round(values))
        except (TypeError, AttributeError):
            return False

    # --------------------- validation methods --------------------- #

    @abstractmethod
    def _decide_if_enumeration_feasible(self) -> bool:
        pass

    @staticmethod
    def _validate_general_parameters(
        max_trajectories: int,
        length: int,
        initial_time: int,
        name: str,
        random_state: int | None,
        enumerate: bool,
    ) -> None:
        if not isinstance(max_trajectories, int) or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer.")
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer.")
        if not isinstance(initial_time, int):
            raise TypeError("initial_time must be an integer.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if random_state is not None and (
            not isinstance(random_state, int) or random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None.")
        if not isinstance(enumerate, bool):
            raise TypeError("enumerate must be a boolean.")
