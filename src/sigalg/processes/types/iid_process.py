from itertools import product
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
        random_state: int | None = None,
        enumerate: bool = False,
    ) -> None:
        self._validate_parameters(rv=rv)
        self._rv = rv

        super().__init__(
            max_trajectories=max_trajectories,
            length=length,
            initial_time=initial_time,
            name=name,
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
        simulated_trajectories = self._rv.rvs(
            size=(self._max_trajectories, self._length),
            random_state=rng,
        )
        time_index = list(range(self._initial_time, self._length + self._initial_time))
        return pd.DataFrame(data=simulated_trajectories, columns=time_index)

    def _enumerate_raw_trajectories(self) -> pd.DataFrame:
        support = self._get_discrete_support()
        time_index = list(range(self._initial_time, self._length + self._initial_time))

        all_trajectories = list(product(support, repeat=self._length))

        n_possible = len(all_trajectories)
        if n_possible > self._max_trajectories:
            rng = np.random.default_rng(self._random_state)
            indices = rng.choice(n_possible, size=self._max_trajectories, replace=False)
            all_trajectories = [all_trajectories[i] for i in sorted(indices)]

        return pd.DataFrame(data=all_trajectories, columns=time_index)

    def _compute_exact_probabilities(
        self, trajectories_df: pd.DataFrame
    ) -> ProbabilityMeasure:
        from ...core.probability_measures.probability_measure import ProbabilityMeasure
        from ...core.spaces.sample_space import SampleSpace

        sample_space_indices = [f"omega{i}" for i in range(len(trajectories_df))]
        sample_space = SampleSpace(indices=sample_space_indices)

        probabilities = {}
        for idx, (_, trajectory) in enumerate(trajectories_df.iterrows()):
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

    # --------------------- discrete distribution helpers --------------------- #

    def _is_discrete(self) -> bool:
        discrete_dists = [
            "bernoulli",
            "binom",
            "poisson",
            "geom",
            "hypergeom",
            "nbinom",
            "randint",
            "zipf",
            "dlaplace",
            "yulesimon",
        ]
        return self._rv.dist.name in discrete_dists

    def _get_discrete_support(self) -> list:
        dist_name = self._rv.dist.name

        if dist_name == "bernoulli":
            return [0, 1]

        elif dist_name == "binom":
            if hasattr(self._rv, "kwds") and "n" in self._rv.kwds:
                n = self._rv.kwds["n"]
            else:
                n = int(self._rv.args[0])
            return list(range(n + 1))

        elif dist_name == "poisson":
            if hasattr(self._rv, "kwds") and "mu" in self._rv.kwds:
                lam = self._rv.kwds["mu"]
            else:
                lam = self._rv.args[0] if self._rv.args else 1.0
            max_val = int(lam + 5 * np.sqrt(lam))
            return list(range(max_val + 1))

        elif dist_name == "geom":
            return list(range(1, 51))

        elif dist_name == "randint":
            if hasattr(self._rv, "kwds"):
                low = self._rv.kwds.get("low", 0)
                high = self._rv.kwds.get("high", 1)
            else:
                low = int(self._rv.args[0]) if len(self._rv.args) > 0 else 0
                high = int(self._rv.args[1]) if len(self._rv.args) > 1 else 1
            return list(range(low, high))

        elif dist_name == "nbinom":
            if hasattr(self._rv, "kwds") and "n" in self._rv.kwds:
                n = self._rv.kwds["n"]
            else:
                n = self._rv.args[0] if self._rv.args else 1
            return list(range(100))

        else:
            raise ValueError(
                f"Don't know how to get support for distribution '{dist_name}'. "
                f"Cannot enumerate trajectories."
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
        if not self._is_discrete():
            raise ValueError(
                f"Cannot enumerate trajectories for continuous distribution "
                f"'{self._rv.dist.name}'. Set enumerate=False."
            )

        n_trajectories = len(self._get_discrete_support()) ** self._length

        if n_trajectories > 1_000_000:
            raise ValueError(
                "The number of possible trajectories is too large to enumerate."
            )
