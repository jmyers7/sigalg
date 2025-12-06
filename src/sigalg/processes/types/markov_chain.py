import numpy as np
import pandas as pd

from ..base.process_factory_methods import ProcessFactoryMethods
from ..base.stochastic_process import StochasticProcess
from ..base.time import Time


class MarkovChain(StochasticProcess, ProcessFactoryMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        *,
        transition_matrix: np.ndarray | pd.DataFrame,
        initial_distribution: np.ndarray | pd.Series | dict | None = None,
        states: list | None = None,
        time: Time,
        max_trajectories: int = 1000,
        name: str = "X",
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        self._validate_parameters(
            transition_matrix=transition_matrix,
            initial_distribution=initial_distribution,
            states=states,
            time=time,
            max_trajectories=max_trajectories,
            name=name,
            random_state=random_state,
            enumerate=enumerate,
        )
        self._transition_matrix = self._process_transition_matrix(
            transition_matrix, states
        )
        self._states = list(self._transition_matrix.index)
        self._time = time
        self._name = name
        self._n_states = len(self._states)
        self._initial_distribution = self._process_initial_distribution(
            initial_distribution, self._states
        )
        self._max_trajectories = max_trajectories
        self._length = len(time)
        self._random_state = random_state
        self._enumerate = enumerate
        fps = self._generate_fps()
        super().__init__(fps=fps)

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
    def time(self) -> Time:
        return self._time

    @property
    def support(self) -> list:
        return self._states

    @property
    def n_states(self) -> int:
        return self._n_states

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
    def length(self) -> int:
        return self._length

    # --------------------- Markov-specific properties --------------------- #

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

    # --------------------- trajectories logic --------------------- #

    def _simulate_raw_trajectories(self) -> pd.DataFrame:
        rng = np.random.default_rng(self._random_state)
        max_trajectories = self._max_trajectories
        P = self._transition_matrix.values
        n_states = self._n_states
        initial_distribution = self._initial_distribution

        initial_state_indices = rng.choice(
            n_states, size=max_trajectories, p=initial_distribution.values
        )

        trajectory_indices = np.empty((max_trajectories, self.length), dtype=int)
        trajectory_indices[:, 0] = initial_state_indices

        for t in range(self.length - 1):
            current_states = trajectory_indices[:, t]
            transition_probs = P[current_states]
            random_vals = rng.random(max_trajectories)
            cumprobs = np.cumsum(transition_probs, axis=1)
            trajectory_indices[:, t + 1] = (cumprobs < random_vals[:, None]).sum(axis=1)

        raw_trajectories = np.array(self._states)[trajectory_indices]
        return pd.DataFrame(data=raw_trajectories)

    def _compute_exact_probabilities(self, raw_trajectories: pd.DataFrame) -> pd.Series:
        trajectories_array = raw_trajectories.values
        state_to_idx = {state: idx for idx, state in enumerate(self._states)}
        trajectories_indices = np.vectorize(state_to_idx.get)(trajectories_array)

        initial_probs = self._initial_distribution.loc[trajectories_array[:, 0]].values
        transition_probs = self._transition_matrix.values[
            trajectories_indices[:, :-1], trajectories_indices[:, 1:]
        ]
        prob_values = initial_probs * np.prod(transition_probs, axis=1)

        return pd.Series(prob_values, index=raw_trajectories.index)

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated" if self._enumerate else "Simulated"
        return f"{prefix} Markov Chain"

    # --------------------- factory methods --------------------- #

    @classmethod
    def random_walk(
        cls,
        p: float = 0.5,
        support: list | None = None,
        time: Time | None = None,
        name: str = "X",
        max_trajectories: int = 1000,
        random_state: int | None = None,
        enumerate: bool = False,
    ):
        if support is None:
            support = [-1, 0, 1]

        if len(support) != 3:
            raise ValueError("Random walk requires exactly 3 states.")

        if time is None:
            time = Time.discrete(start=0, length=10)

        sorted_states = sorted(support)
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
            time=time,
            initial_distribution=initial_distribution,
            states=sorted_states,
            name=name,
            max_trajectories=max_trajectories,
            random_state=random_state,
            enumerate=enumerate,
        )

    # --------------------- parameter generation methods --------------------- #

    @staticmethod
    def _process_transition_matrix(
        transition_matrix: np.ndarray | pd.DataFrame, support: list | None
    ) -> pd.DataFrame:
        if isinstance(transition_matrix, pd.DataFrame):
            P = transition_matrix
        else:
            n = len(transition_matrix)
            if support is None:
                support = list(range(n))
            elif len(support) != n:
                raise ValueError(
                    f"Length of support ({len(support)}) must match "
                    f"transition_matrix dimension ({n})."
                )
            P = pd.DataFrame(transition_matrix, index=support, columns=support)

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

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        transition_matrix: np.ndarray | pd.DataFrame,
        initial_distribution: np.ndarray | pd.Series | dict | None,
        states: list | None,
        time: Time,
        max_trajectories: int,
        name: str,
        random_state: int | None,
        enumerate: bool,
    ):
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
        if not isinstance(time, Time):
            raise TypeError("time must be a Time object.")
        if not isinstance(max_trajectories, int) or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if random_state is not None:
            if not isinstance(random_state, int) or random_state < 0:
                raise TypeError("random_state must be a non-negative integer or None.")
        if not isinstance(enumerate, bool):
            raise TypeError("enumerate must be a boolean.")

    def _decide_if_enumeration_feasible(self) -> None:
        n_trajectories = self.n_states**self.length

        if n_trajectories > 1_000_000:
            raise ValueError(
                "The number of possible trajectories is too large to enumerate."
            )
        if n_trajectories > self._max_trajectories:
            raise ValueError(
                f"The number of possible trajectories {n_trajectories} is greater than max_trajectories {self._max_trajectories}. "
            )
