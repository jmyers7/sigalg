import warnings
from itertools import product

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess


class MarkovChain(StochasticProcess):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        transition_matrix: np.ndarray | pd.DataFrame,
        initial_distribution: np.ndarray | pd.Series | dict | None = None,
        states: list | None = None,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        max_trajectories: int = 1000,
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        self._validate_parameters(
            transition_matrix=transition_matrix,
            initial_distribution=initial_distribution,
            states=states,
            length=length,
            initial_time=initial_time,
            name=name,
            max_trajectories=max_trajectories,
            random_state=random_state,
            enumerate=enumerate,
        )

        self._transition_matrix = self._process_transition_matrix(
            transition_matrix, states
        )
        self._states = list(self._transition_matrix.index)
        self._n_states = len(self._states)

        self._initial_distribution = self._process_initial_distribution(
            initial_distribution, self._states
        )

        self._length = length
        self._initial_time = initial_time
        self._name = name
        self._max_trajectories = max_trajectories
        self._random_state = random_state
        self._enumerate = enumerate

        if self._enumerate:
            self._validate_enumeration_feasible()

        super().__init__()

    # --------------------- properties --------------------- #

    @property
    def transition_matrix(self) -> pd.DataFrame:
        return self._transition_matrix

    @property
    def initial_distribution(self) -> pd.Series:
        return self._initial_distribution

    @property
    def states(self) -> list:
        return self._states

    @property
    def n_states(self) -> int:
        return self._n_states

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
    def max_trajectories(self) -> int:
        return self._max_trajectories

    @property
    def random_state(self) -> int | None:
        return self._random_state

    @property
    def enumerate(self) -> bool:
        return self._enumerate

    @property
    def n_possible_trajectories(self) -> int:
        return self._n_states**self._length

    @property
    def is_complete_enumeration(self) -> bool:
        if not self._enumerate:
            return False
        return self._max_trajectories >= self.n_possible_trajectories

    @property
    def stationary_distribution(self) -> pd.Series:
        P = self._transition_matrix.values
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, stationary_idx])
        stationary = stationary / stationary.sum()
        return pd.Series(stationary, index=self._states)

    @property
    def is_irreducible(self) -> bool:
        P = self._transition_matrix.values
        n = len(P)
        reachability = np.linalg.matrix_power(P > 0, n)
        return np.all(reachability > 0)

    @property
    def is_aperiodic(self) -> bool:
        P = self._transition_matrix.values
        n = len(P)
        for i in range(n):
            powers_sum = sum(
                np.linalg.matrix_power(P, k)[i, i] > 0 for k in range(1, n + 1)
            )
            if powers_sum > 1:
                return True
        return False

    # --------------------- simulation logic --------------------- #

    def _simulate(self) -> pd.DataFrame:
        if self._enumerate:
            return self._enumerate_trajectories()
        else:
            return self._simulate_trajectories()

    def _simulate_trajectories(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._random_state)

        trajectories = []
        time_index = list(range(self._initial_time, self._length + self._initial_time))

        for _ in range(self._max_trajectories):
            trajectory = []
            current_state = rng.choice(
                self._states, p=self._initial_distribution.values
            )
            trajectory.append(current_state)

            for _ in range(self._length - 1):
                current_idx = self._states.index(current_state)
                transition_probs = self._transition_matrix.iloc[current_idx].values
                current_state = rng.choice(self._states, p=transition_probs)
                trajectory.append(current_state)

            trajectories.append(trajectory)

        return pd.DataFrame(data=trajectories, columns=time_index)

    def _enumerate_trajectories(self) -> pd.DataFrame:
        time_index = list(range(self._initial_time, self._length + self._initial_time))

        all_trajectories = list(product(self._states, repeat=self._length))

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
        from ..base.process_trajectories import ProcessTrajectories

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

        self._process_trajectories = ProcessTrajectories(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=probability_space.probability_measure,
        )

        self._probability_measure = probability_space.probability_measure

    def _compute_exact_probabilities(self, trajectories_df: pd.DataFrame) -> dict:
        probabilities = {}
        sample_space_indices = [f"omega{i}" for i in range(len(trajectories_df))]

        for idx, (_, trajectory) in enumerate(trajectories_df.iterrows()):
            initial_state = trajectory.iloc[0]
            prob = self._initial_distribution[initial_state]

            for t in range(len(trajectory) - 1):
                current_state = trajectory.iloc[t]
                next_state = trajectory.iloc[t + 1]
                transition_prob = self._transition_matrix.loc[current_state, next_state]
                prob *= transition_prob

            probabilities[sample_space_indices[idx]] = prob

        total = sum(probabilities.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            probabilities = {k: v / total for k, v in probabilities.items()}

        return probabilities

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        prefix = "Enumerated" if self._enumerate else "Simulated"
        return (
            f"{prefix} Markov Chain '{self._name}': "
            f"{self._n_states} states, length={self._length}, "
            f"n_trajectories={self.n_trajectories}"
        )

    def __str__(self) -> str:
        prefix = "Enumerated" if self._enumerate else "Simulated"
        header = f"{prefix} Markov Chain {self._name}"
        separator = "=" * len(header)
        result = (
            header
            + "\n"
            + separator
            + f"\n\n* States: {self._states}"
            + f"\n* Number of states: {self._n_states}"
            + f"\n* Length: {self._length}"
            + f"\n* Initial time: {self._initial_time}"
            + f"\n* Number of trajectories: {self.n_trajectories}"
            + f"\n* Random state: {self._random_state}"
            + "\n\n* Initial Distribution:"
            + f"\n{self._initial_distribution.to_string()}"
            + "\n\n* Transition Matrix:"
            + f"\n{self._transition_matrix.to_string()}"
        )

        if self._enumerate and hasattr(self, "_process_trajectories"):
            result += (
                "\n\n* Trajectories:"
                + f"\n{self._process_trajectories.feature_embedding.values.to_string()}"
            )

        return result

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated" if self._enumerate else "Simulated"
        return f"{prefix} Markov Chain"

    # --------------------- factory methods --------------------- #

    @classmethod
    def random_walk(
        cls,
        p: float = 0.5,
        states: list | None = None,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        max_trajectories: int = 1000,
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        if states is None:
            states = [-1, 0, 1]

        if len(states) != 3:
            raise ValueError("Random walk requires exactly 3 states.")

        sorted_states = sorted(states)
        transition_matrix = pd.DataFrame(
            [[0, 1 - p, p], [1 - p, 0, p], [1 - p, p, 0]],
            index=sorted_states,
            columns=sorted_states,
        )

        initial_distribution = pd.Series(
            [0, 1, 0], index=sorted_states, name="initial_distribution"
        )

        return cls(
            transition_matrix=transition_matrix,
            initial_distribution=initial_distribution,
            states=sorted_states,
            length=length,
            initial_time=initial_time,
            name=name,
            max_trajectories=max_trajectories,
            random_state=random_state,
            enumerate=enumerate,
        )

    @classmethod
    def birth_death(
        cls,
        birth_rate: float,
        death_rate: float,
        max_population: int = 10,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        max_trajectories: int = 1000,
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        states = list(range(max_population + 1))
        n = len(states)

        P = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                P[i, i] = 1 - birth_rate
                P[i, i + 1] = birth_rate
            elif i == n - 1:
                P[i, i - 1] = death_rate
                P[i, i] = 1 - death_rate
            else:
                P[i, i - 1] = death_rate
                P[i, i] = 1 - birth_rate - death_rate
                P[i, i + 1] = birth_rate

        transition_matrix = pd.DataFrame(P, index=states, columns=states)

        initial_distribution = pd.Series(
            [1.0] + [0.0] * (n - 1), index=states, name="initial_distribution"
        )

        return cls(
            transition_matrix=transition_matrix,
            initial_distribution=initial_distribution,
            states=states,
            length=length,
            initial_time=initial_time,
            name=name,
            max_trajectories=max_trajectories,
            random_state=random_state,
            enumerate=enumerate,
        )

    @classmethod
    def ehrenfest_urn(
        cls,
        n_balls: int = 10,
        length: int = 10,
        initial_time: int = 0,
        name: str = "X",
        max_trajectories: int = 1000,
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        states = list(range(n_balls + 1))
        n = len(states)

        P = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                P[i, i + 1] = 1.0
            elif i == n - 1:
                P[i, i - 1] = 1.0
            else:
                P[i, i - 1] = i / n_balls
                P[i, i + 1] = (n_balls - i) / n_balls

        transition_matrix = pd.DataFrame(P, index=states, columns=states)

        initial_distribution = pd.Series(
            [0.0] * (n // 2) + [1.0] + [0.0] * (n - n // 2 - 1),
            index=states,
            name="initial_distribution",
        )

        return cls(
            transition_matrix=transition_matrix,
            initial_distribution=initial_distribution,
            states=states,
            length=length,
            initial_time=initial_time,
            name=name,
            max_trajectories=max_trajectories,
            random_state=random_state,
            enumerate=enumerate,
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        transition_matrix: np.ndarray | pd.DataFrame,
        initial_distribution: np.ndarray | pd.Series | dict | None,
        states: list | None,
        length: int,
        initial_time: int,
        name: str,
        max_trajectories: int,
        random_state: int | None,
        enumerate: bool,
    ) -> None:
        if not isinstance(transition_matrix, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                "transition_matrix must be a numpy array or pandas DataFrame."
            )

        if isinstance(transition_matrix, np.ndarray):
            if transition_matrix.ndim != 2:
                raise ValueError("transition_matrix must be a 2D array.")
            if transition_matrix.shape[0] != transition_matrix.shape[1]:
                raise ValueError("transition_matrix must be square.")
        else:
            if transition_matrix.shape[0] != transition_matrix.shape[1]:
                raise ValueError("transition_matrix must be square.")

        if initial_distribution is not None and not isinstance(
            initial_distribution, (np.ndarray, pd.Series, dict)
        ):
            raise TypeError(
                "initial_distribution must be a numpy array, pandas Series, dict, or None."
            )

        if states is not None and not isinstance(states, list):
            raise TypeError("states must be a list or None.")

        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer.")

        if not isinstance(initial_time, int):
            raise TypeError("initial_time must be an integer.")

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if not isinstance(max_trajectories, int) or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer.")

        if random_state is not None and (
            not isinstance(random_state, int) or random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None.")

        if not isinstance(enumerate, bool):
            raise TypeError("enumerate must be a boolean.")

    def _validate_enumeration_feasible(self) -> None:
        n_trajectories = self.n_possible_trajectories

        if n_trajectories > 1_000_000:
            warnings.warn(
                f"Enumerating {n_trajectories:,} trajectories may be computationally "
                f"expensive and memory-intensive. Consider reducing length or number "
                f"of states, or use enumerate=False for simulation-based approach.",
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

    @staticmethod
    def _process_transition_matrix(
        transition_matrix: np.ndarray | pd.DataFrame, states: list | None
    ) -> pd.DataFrame:
        if isinstance(transition_matrix, pd.DataFrame):
            P = transition_matrix
        else:
            n = len(transition_matrix)
            if states is None:
                states = list(range(n))
            elif len(states) != n:
                raise ValueError(
                    f"Length of states ({len(states)}) must match "
                    f"transition_matrix dimension ({n})."
                )
            P = pd.DataFrame(transition_matrix, index=states, columns=states)

        if not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1.")

        if np.any(P.values < 0):
            raise ValueError("All entries in transition_matrix must be non-negative.")

        return P

    @staticmethod
    def _process_initial_distribution(
        initial_distribution: np.ndarray | pd.Series | dict | None, states: list
    ) -> pd.Series:
        n = len(states)

        if initial_distribution is None:
            return pd.Series([1.0 / n] * n, index=states, name="initial_distribution")

        if isinstance(initial_distribution, dict):
            pi = pd.Series(initial_distribution, name="initial_distribution")
            if not all(s in pi.index for s in states):
                missing = [s for s in states if s not in pi.index]
                raise ValueError(f"initial_distribution missing states: {missing}")
            pi = pi.reindex(states, fill_value=0.0)
        elif isinstance(initial_distribution, pd.Series):
            pi = initial_distribution.copy()
            pi.name = "initial_distribution"
            if len(pi) != n:
                raise ValueError(
                    f"Length of initial_distribution ({len(pi)}) must match "
                    f"number of states ({n})."
                )
            pi.index = states
        else:
            if len(initial_distribution) != n:
                raise ValueError(
                    f"Length of initial_distribution ({len(initial_distribution)}) "
                    f"must match number of states ({n})."
                )
            pi = pd.Series(
                initial_distribution, index=states, name="initial_distribution"
            )

        if not np.isclose(pi.sum(), 1.0, atol=1e-6):
            raise ValueError("initial_distribution must sum to 1.")

        if np.any(pi.values < 0):
            raise ValueError(
                "All entries in initial_distribution must be non-negative."
            )

        return pi
