from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.random_objects.random_variable import RandomVariable
    from .trajectory import Trajectory

from ...core.featurized_spaces.featurized_probability_space import (
    FeaturizedProbabilitySpace,
)


class Trajectories(FeaturizedProbabilitySpace):

    # --------------------- properties --------------------- #

    @property
    def values(self):
        return self.feature_embedding.values

    @property
    def time_index(self):
        return self.feature_embedding.values.columns

    # --------------------- data access methods --------------------- #

    @property
    def trajectory_at(self):
        return self._TrajectoryIndexer(self)

    class _TrajectoryIndexer:
        def __init__(self, trajectories) -> None:
            self.parent = trajectories

        def __getitem__(self, key) -> Trajectory:
            from .trajectory import Trajectory

            features = self.parent.values.iloc[key]
            return Trajectory(values=features, name=features.name)

    @property
    def rv_at(self):
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, trajectories):
            self.trajectories = trajectories

        def __getitem__(self, time) -> RandomVariable:
            from ...core.random_objects.random_variable import RandomVariable

            if time not in self.trajectories.feature_embedding.values.columns:
                raise ValueError(f"Time {time} not in process time index")
            values = self.trajectories.values[time]
            rv = RandomVariable.from_values(
                values=values,
                probability_space=self.trajectories.probability_space,
                name=f"{self.trajectories.feature_embedding.name}{time}",
            )
            rv._values.index.name = "trajectory"
            return rv

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
