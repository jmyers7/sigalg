from typing import TYPE_CHECKING

import pandas as pd

from ...core import FeatureEmbedding

if TYPE_CHECKING:
    from ...core import RandomVariable, SampleSpace
    from .time import Time
    from .trajectory import Trajectory


class Trajectories(FeatureEmbedding):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        *,
        sample_space: SampleSpace,
        values: pd.DataFrame,
        time: Time,
        name: str = "X",
    ) -> None:
        super().__init__(sample_space=sample_space, values=values, name=name)
        self._time = time

    # --------------------- properties --------------------- #

    @property
    def time(self) -> Time:
        return self._time

    # --------------------- data access methods --------------------- #

    @property
    def trajectory_at(self):
        return self._TrajectoryIndexer(self)

    class _TrajectoryIndexer:
        def __init__(self, trajectories) -> None:
            self.trajectories = trajectories

        def __getitem__(self, key) -> Trajectory:
            from .trajectory import Trajectory

            features = self.trajectories.values.iloc[key]
            return Trajectory(values=features, name=features.name)

    @property
    def rv_at(self):
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, trajectories):
            self.trajectories = trajectories

        def __getitem__(self, time) -> RandomVariable:
            from ...core.random_objects.random_variable import RandomVariable

            if time not in self.trajectories.values.columns:
                raise ValueError(f"Time {time} not in process time index")
            values = self.trajectories.values[time]
            rv = RandomVariable.from_values(
                values=values,
                domain=self.trajectories.sample_space,
                name=f"{self.trajectories.name}{time}",
            )
            rv._values.index.name = "trajectory"
            return rv

    def iter_trajectories(self):
        for i in range(len(self.values)):
            yield self.trajectory_at[i]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"{self.values}"


class TrajectoriesMethods:

    @property
    def trajectory_at(self):
        return self.trajectories.trajectory_at

    @property
    def rv_at(self):
        return self.trajectories.rv_at
