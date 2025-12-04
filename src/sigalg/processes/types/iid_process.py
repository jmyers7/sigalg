import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ..base.stochastic_process import StochasticProcess


class IIDProcess(StochasticProcess):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        rv: rv_frozen,
        max_trajectories: int = 1000,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        self._validate_parameters(
            rv=rv,
            max_trajectories=max_trajectories,
            length=length,
            initial_time=initial_time,
            name=name,
            random_state=random_state,
            enumerate=enumerate,
        )
        self._rv = rv
        self._max_trajectories = max_trajectories
        self._length = length
        self._initial_time = initial_time
        self._name = name
        self._random_state = random_state
        self._enumerate = enumerate

        if self._enumerate:
            self._validate_enumeration_feasible()

        super().__init__()

    # --------------------- properties --------------------- #

    @property
    def rv(self) -> rv_frozen:
        return self._rv

    @property
    def max_trajectories(self) -> int:
        return self._max_trajectories

    @property
    def length(self) -> int:
        return self._length

    @property
    def initial_time(self) -> int:
        return self._initial_time

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    @property
    def random_state(self) -> int | None:
        return self._random_state

    @property
    def enumerate(self) -> bool:
        return self._enumerate

    @property
    def n_possible_trajectories(self) -> int | float:
        if not self._is_discrete():
            return float("inf")
        support_size = len(self._get_discrete_support())
        return support_size**self._length

    @property
    def is_complete_enumeration(self) -> bool:
        if not self._enumerate:
            return False
        n_possible = self.n_possible_trajectories
        if n_possible == float("inf"):
            return False
        return self._max_trajectories >= n_possible

    # --------------------- simulation logic --------------------- #

    def _simulate(self) -> pd.DataFrame:
        if self._enumerate:
            return self._enumerate_trajectories()
        else:
            return self._simulate_trajectories()

    def _simulate_trajectories(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._random_state)
        simulated_trajectories = self._rv.rvs(
            size=(self._max_trajectories, self._length),
            random_state=rng,
        )
        time_index = list(range(self._initial_time, self._length + self._initial_time))
        return pd.DataFrame(data=simulated_trajectories, columns=time_index)

    def _enumerate_trajectories(self) -> pd.DataFrame:
        support = self._get_discrete_support()
        time_index = list(range(self._initial_time, self._length + self._initial_time))

        all_trajectories = list(product(support, repeat=self._length))

        n_possible = len(all_trajectories)
        if n_possible > self._max_trajectories:
            rng = np.random.default_rng(self._random_state)
            indices = rng.choice(n_possible, size=self._max_trajectories, replace=False)
            all_trajectories = [all_trajectories[i] for i in sorted(indices)]

        return pd.DataFrame(data=all_trajectories, columns=time_index)

    # --------------------- generation methods override --------------------- #

    def _generate_trajectories(self):
        if self._enumerate:
            self._generate_enumerated_trajectories()
        else:
            super()._generate_trajectories()

    def _generate_enumerated_trajectories(self):
        from ...core.featurized_spaces.feature_embedding import FeatureEmbedding
        from ...core.spaces.probability_space import ProbabilitySpace
        from ...core.spaces.sample_space import SampleSpace
        from ..base.trajectories import Trajectories

        self._simulated_trajectories = self._simulate()

        self._n_trajectories = len(self._simulated_trajectories)

        sample_space = SampleSpace(
            indices=[f"omega{i}" for i in range(self._n_trajectories)]
        )

        probabilities = self._compute_exact_probabilities(self._simulated_trajectories)

        probability_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )

        time_index = range(
            self._initial_time,
            self._initial_time + self._length,
        )
        df = pd.DataFrame(
            self._simulated_trajectories.values,
            index=sample_space,
            columns=time_index,
        )
        df.index.name = "trajectory"
        df.columns.name = "time"
        feature_embedding = FeatureEmbedding(
            values=df, name=self._name, sample_space=sample_space
        )

        self._trajectories = Trajectories(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=probability_space.probability_measure,
        )

        self._probability_measure = probability_space.probability_measure

    def _compute_exact_probabilities(self, trajectories_df: pd.DataFrame) -> dict:
        probabilities = {}
        sample_space_indices = [f"omega{i}" for i in range(len(trajectories_df))]

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

        return probabilities

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
    def _validate_parameters(
        rv: rv_frozen,
        max_trajectories: int,
        length: int,
        initial_time: int,
        name: str,
        random_state: int | None,
        enumerate: bool,
    ) -> None:
        if not isinstance(rv, rv_frozen):
            raise TypeError(
                "rv must be an instance of scipy.stats._distn_infrastructure.rv_frozen."
            )
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

    def _validate_enumeration_feasible(self) -> None:
        if not self._is_discrete():
            raise ValueError(
                f"Cannot enumerate trajectories for continuous distribution "
                f"'{self._rv.dist.name}'. Set enumerate=False."
            )

        n_trajectories = self.n_possible_trajectories

        if n_trajectories > 1_000_000:
            warnings.warn(
                f"Enumerating {n_trajectories:,} trajectories may be computationally "
                f"expensive and memory-intensive. Consider reducing length or using "
                f"enumerate=False for simulation-based approach.",
                RuntimeWarning,
                stacklevel=2,
            )

        if n_trajectories > self._max_trajectories:
            warnings.warn(
                f"Total possible trajectories ({n_trajectories:,}) exceeds "
                f"max_trajectories ({self._max_trajectories}). Will sample "
                f"{self._max_trajectories} trajectories from the complete enumeration.",
                RuntimeWarning,
                stacklevel=2,
            )
