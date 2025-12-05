from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ...core.probability_measures.probability_measure import ProbabilityMeasure


class IIDProcess(StochasticProcess):

    _required_type_parameters = {"rv"}

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        *,
        rv: rv_frozen,
        max_trajectories: int = 1000,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        support: list | None = None,
        random_state: int | None = None,
        enumerate: bool = False,
    ) -> None:
        self._validate_parameters(rv=rv)
        self._rv = rv
        self._support = support

        super().__init__(
            max_trajectories=max_trajectories,
            length=length,
            initial_time=initial_time,
            name=name,
            support=support,
            random_state=random_state,
            enumerate=enumerate,
        )

    # --------------------- properties --------------------- #

    @property
    def rv(self) -> rv_frozen:
        return self._rv

    # --------------------- trajectories logic --------------------- #

    def _simulate_raw_trajectories(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._random_state)

        raw_trajectories = self._rv.rvs(
            size=(self._max_trajectories, self._length),
            random_state=rng,
        )

        time_index = list(range(self._initial_time, self._length + self._initial_time))
        raw_trajectories = pd.DataFrame(data=raw_trajectories, columns=time_index)
        raw_trajectories.columns.name = "time"
        return raw_trajectories

    def _compute_exact_probabilities(
        self, raw_trajectories: pd.DataFrame
    ) -> ProbabilityMeasure:
        from ...core.probability_measures.probability_measure import ProbabilityMeasure
        from ...core.spaces.sample_space import SampleSpace

        sample_space_indices = [f"omega{i}" for i in range(len(raw_trajectories))]
        sample_space = SampleSpace(indices=sample_space_indices)

        probabilities = {}
        for idx, (_, trajectory) in enumerate(raw_trajectories.iterrows()):
            prob = 1.0
            for value in trajectory:
                if hasattr(self._rv, "pmf"):
                    prob *= self._rv.pmf(value)
                else:
                    prob *= self._rv.pdf(value)
            probabilities[sample_space_indices[idx]] = prob

        total = sum(probabilities.values())
        if not np.isclose(total, 1.0):
            probabilities = {k: v / total for k, v in probabilities.items()}

        return ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        prefix = "Enumerated IID" if self._enumerate else "IID"
        return (
            f"{prefix} {self._rv.dist.name.capitalize()} Process '{self._name}': "
            f"length={self._length}, "
            f"initial_time={self._initial_time}, "
            f"n_trajectories={self.n_trajectories}"
        )

    def __str__(self) -> str:
        prefix = "Enumerated IID" if self._enumerate else "IID"
        header = f"{prefix} {self._rv.dist.name.capitalize()} Process {self._name}"
        separator = "=" * len(header)
        result = (
            header
            + "\n"
            + separator
            + f"\n\n* Length: {self._length}"
            + f"\n* Initial time: {self._initial_time}"
            + f"\n* Number of trajectories: {self.n_trajectories}"
            + f"\n* Random state: {self._random_state}"
            + f"\n* Distribution: {self._rv.dist.name}"
        )

        if self._enumerate:
            result += f"\n\n* Trajectories:\n{self.process_trajectories.values}"

        return result

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated IID" if self._enumerate else "IID"
        return f"{prefix} {self._rv.dist.name.capitalize()} Process"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(rv: rv_frozen):
        if not isinstance(rv, rv_frozen):
            raise TypeError(
                "rv must be an instance of scipy.stats._distn_infrastructure.rv_frozen."
            )

    def _decide_if_enumeration_feasible(self) -> None:
        if self._support is None:
            raise ValueError(
                "Cannot enumerate trajectories without explicit support. "
                "Please provide the 'support' parameter."
            )

        n_trajectories = len(self._support) ** self._length

        if n_trajectories > 1_000_000:
            raise ValueError(
                "The number of possible trajectories is too large to enumerate."
            )
        if n_trajectories > self._max_trajectories:
            raise ValueError(
                f"The number of possible trajectories {n_trajectories} is greater than max_trajectories {self._max_trajectories}. "
            )
