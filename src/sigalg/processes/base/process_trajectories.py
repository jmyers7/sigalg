from ...core.featurized_spaces.featurized_probability_space import (
    FeaturizedProbabilitySpace,
)


class ProcessTrajectories(FeaturizedProbabilitySpace):

    def __init__(self, sample_space, feature_embedding, probability_measure, name):
        super().__init__(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=probability_measure,
        )
        self._name = name

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
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key):
            from .trajectory import Trajectory

            features = self.parent.values.iloc[key]
            return Trajectory(features=features)

    @property
    def rv_at(self):
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, time):
            from ...core.random_objects.random_variable import RandomVariable

            if time not in self.parent.feature_embedding.values.columns:
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


class ProcessTrajectoriesMethods:

    @property
    def trajectory_at(self):
        return self.process_trajectories.trajectory_at

    @property
    def rv_at(self):
        return self.process_trajectories.rv_at
