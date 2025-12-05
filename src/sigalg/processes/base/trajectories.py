from typing import TYPE_CHECKING

from ...core.featurized_spaces.feature_embedding import FeatureEmbedding

if TYPE_CHECKING:
    from ...core.random_objects.random_variable import RandomVariable
    from .trajectory import Trajectory


class Trajectories(FeatureEmbedding):

    # --------------------- properties --------------------- #

    @property
    def time_index(self):
        return self.values.columns

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
                # probability_space=self.trajectories.probability_space,
                name=f"{self.trajectories.name}{time}",
            )
            rv._values.index.name = "trajectory"
            return rv

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"{self.values}"
