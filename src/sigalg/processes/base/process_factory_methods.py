from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ...core import FeaturizedProbabilitySpace
    from .time import Time


class ProcessFactoryMethods(ABC):

    def _generate_fps(self) -> FeaturizedProbabilitySpace:
        from ...core import (
            ProbabilityMeasure,
            SampleSpace,
            SigmaAlgebra,
        )
        from .trajectories import Trajectories

        if self.enumerate:
            self._decide_if_enumeration_feasible()

        raw_trajectories = (
            self._enumerate_raw_trajectories()
            if self.enumerate
            else self._simulate_raw_trajectories()
        )

        if self.enumerate:
            prob_series = self._compute_exact_probabilities(raw_trajectories)
        else:
            prob_series = self._compute_empirical_probabilities(raw_trajectories)

        grouped_trajectories, grouped_probabilities = self._groupby_trajectories(
            raw_trajectories, prob_series
        )

        sample_space_indices = [f"omega{i}" for i in range(len(grouped_trajectories))]
        sample_space = SampleSpace(indices=sample_space_indices)

        probabilities = dict(zip(sample_space, grouped_probabilities))
        probability_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )

        grouped_trajectories = grouped_trajectories.reindex(columns=self.time)
        grouped_trajectories.index = sample_space
        grouped_trajectories.index.name = "trajectory"

        trajectories = Trajectories(
            sample_space=sample_space,
            values=grouped_trajectories,
            feature_index=self.time,
            name=self.name,
        )

        sigma_algebra = SigmaAlgebra.power_set(sample_space)

        output_dict = {
            "sample_space": sample_space,
            "sigma_algebra": sigma_algebra,
            "probability_measure": probability_measure,
            "feature_embedding": trajectories,
        }

        return output_dict

    # --------------------- abstract properties and methods --------------------- #

    @property
    @abstractmethod
    def time(self) -> Time:
        pass

    @property
    @abstractmethod
    def max_trajectories(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def support(self) -> list | None:
        pass

    @property
    @abstractmethod
    def random_state(self) -> int | None:
        pass

    @property
    @abstractmethod
    def enumerate(self) -> bool:
        pass

    @abstractmethod
    def _simulate_raw_trajectories(
        self, max_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def _compute_exact_probabilities(self, raw_trajectories: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def _decide_if_enumeration_feasible(self) -> bool:
        pass

    @abstractmethod
    def _plot_title(self):
        pass

    # --------------------- trajectories logic --------------------- #

    def _enumerate_raw_trajectories(self) -> pd.DataFrame:
        from itertools import product

        all_trajectories = list(product(self.support, repeat=len(self.time)))
        return pd.DataFrame(data=all_trajectories)

    def _compute_empirical_probabilities(
        self, raw_trajectories: pd.DataFrame
    ) -> pd.Series:
        prob_series = pd.Series(1 / len(raw_trajectories))
        return prob_series

    def _groupby_trajectories(
        self, raw_trajectories: pd.DataFrame, prob_series: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        prob_series.name = "probability"
        df = pd.concat([raw_trajectories, prob_series], axis=1)
        grouped = df.groupby(list(df.columns[:-1]))
        result = grouped["probability"].sum().reset_index()
        result["probability"] = result["probability"] / result["probability"].sum()
        grouped_trajectories = result.drop(columns=["probability"])
        grouped_probabilities = result["probability"]
        return grouped_trajectories, grouped_probabilities
