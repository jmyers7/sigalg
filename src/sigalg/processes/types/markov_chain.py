"""Markov chain module."""

from collections.abc import Hashable

import numpy as np
import pandas as pd

from ...core.base.index import Index
from ...core.base.sample_space import SampleSpace
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ..base.stochastic_process import StochasticProcess


class MarkovChain(StochasticProcess):
    """A class representing a Markov chain stochastic process.

    Parameters
    ----------
    transition_matrix : pd.DataFrame
        A DataFrame representing the transition probabilities between states. The index and columns should correspond to the states of the Markov chain, and each row should sum to `1`.
    initial_distribution : ProbabilityMeasure
        A ProbabilityMeasure representing the initial distribution over the states of the Markov chain. Its sample space should match the states defined in the transition matrix.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    index : Index | None, default=None
        The index of the stochastic process. If `None`, it will be generated later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.

    Raises
    ------
    TypeError
        If `transition_matrix` is not a pandas DataFrame or if `initial_distribution` is not a ProbabilityMeasure.
    ValueError
        If the index and columns of `transition_matrix` do not match the sample space of `initial_distribution`, if any row of `transition_matrix` does not sum to `1`, or if any entry in `transition_matrix` is negative.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace
    >>> from sigalg.processes import MarkovChain
    >>> state_space = SampleSpace().from_list(["rain", "sun"])
    >>> P = pd.DataFrame(
    ...     data=[
    ...         [0.9, 0.1],  # P(rain | rain) = 0.9, P(sun | rain) = 0.1
    ...         [0.4, 0.6],  # P(rain | sun) = 0.4, P(sun | sun) = 0.6
    ...     ],
    ...     index=state_space,
    ...     columns=state_space,
    ... )
    >>> pi = ProbabilityMeasure(name="pi").from_dict({"rain": 0.25, "sun": 0.75})
    >>> X = MarkovChain(
    ...     transition_matrix=P,
    ...     initial_distribution=pi,
    ...     name="X",
    ... ).from_simulation(
    ...     max_trajectories=100_000,
    ...     length=3,
    ...     random_state=42,
    ... )
    >>> X # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    time      0     1     2
    0       sun   sun   sun
    1       sun   sun  rain
    2       sun   sun   sun
    3       sun  rain  rain
    4      rain  rain  rain
    ...     ...   ...   ...
    99995   sun  rain  rain
    99996   sun   sun  rain
    99997   sun  rain  rain
    99998  rain  rain  rain
    99999   sun  rain  rain
    <BLANKLINE>
    [100000 rows x 3 columns]
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        transition_matrix: pd.DataFrame,
        initial_distribution: ProbabilityMeasure,
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
    ) -> None:
        if not isinstance(transition_matrix, pd.DataFrame):
            raise TypeError("transition_matrix must be a pandas DataFrame.")
        if not isinstance(initial_distribution, ProbabilityMeasure):
            raise TypeError("initial_distribution must be a ProbabilityMeasure.")
        state_space = initial_distribution.sample_space
        if not transition_matrix.index.equals(
            state_space.data
        ) or not transition_matrix.columns.equals(state_space.data):
            raise ValueError(
                "transition_matrix index and columns must match the sample space of initial_distribution."
            )
        if not np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1.")
        if np.any(transition_matrix.values < 0):
            raise ValueError("All entries in transition_matrix must be non-negative.")

        super().__init__(
            domain=domain,
            index=index,
            name=name,
        )

        self.support = list(state_space)
        self.states = self.support
        self.n_states = len(self.states)
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution
        self._is_discrete_state = True

    # --------------------- data generation methods --------------------- #

    def _simulation_logic(
        self,
        max_trajectories: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Generate simulated data for the Markov chain.

        Parameters
        ----------
        max_trajectories : int
            The maximum number of trajectories to simulate.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        rng = np.random.default_rng(random_state)

        P = self.transition_matrix.values
        n_states = self.n_states
        length = len(self.time)
        initial_distribution = self.initial_distribution

        initial_state_indices = rng.choice(
            n_states, size=max_trajectories, p=initial_distribution.data.values
        )

        trajectory_indices = np.empty((max_trajectories, length), dtype=int)
        trajectory_indices[:, 0] = initial_state_indices

        for t in range(length - 1):
            current_states = trajectory_indices[:, t]
            transition_probs = P[current_states]
            random_vals = rng.random(max_trajectories)
            cumprobs = np.cumsum(transition_probs, axis=1)
            trajectory_indices[:, t + 1] = (cumprobs < random_vals[:, None]).sum(axis=1)

        raw_trajectories = np.array(self.states)[trajectory_indices]
        return pd.DataFrame(data=raw_trajectories)

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        """Generate the exact probability measure for the Markov chain.

        Parameters
        ----------
        name : Hashable | None, default="P"
            The name of the generated probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The generated probability measure.
        """
        data_array = self.data.values
        state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        trajectories_indices = np.vectorize(state_to_idx.get)(data_array)

        initial_probs = self.initial_distribution.data.loc[data_array[:, 0]].values
        transition_probs = self.transition_matrix.values[
            trajectories_indices[:, :-1], trajectories_indices[:, 1:]
        ]
        prob_values = initial_probs * np.prod(transition_probs, axis=1)

        return ProbabilityMeasure(name=name).from_pandas(
            pd.Series(prob_values, index=self.domain.data)
        )

    # --------------------- Markov-specific methods --------------------- #

    @classmethod
    def random_walk(
        cls,
        p: float = 0.5,
        states: list[Hashable] | None = None,
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
    ):
        """Construct a simple random walk Markov chain with specified parameters.

        A random walk is a Markov chain where the process moves to the right with probability `p` and to the left with probability `1-p` at each time step. The states represent the position of the random walk, and the transition probabilities are defined accordingly.

        Parameters
        ----------
        p : float, default=0.5
            The probability of moving to the right (increasing state) at each time step. Must be in the range `[0, 1]`.
        states : list[Hashable] | None, default=None
            A list of three states representing the possible positions of the random walk. If `None`, the default states will be `[-1, 0, 1]`, where `-1` represents a step to the left, `0` represents staying in place, and `1` represents a step to the right.
        domain : SampleSpace | None, default=None
            The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
        index : Index | None, default=None
            The index of the stochastic process. If `None`, it will be generated later through data generation methods.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `states` is not a list of three hashable values or if `p` is not a float in the range `[0, 1]`.
        ValueError
            If `states` does not contain exactly three hashable values or if `p` is not in the range `[0, 1]`.

        Returns
        -------
        random_walk : MarkovChain
            A `MarkovChain` instance representing the random walk.

        Examples
        --------
        >>> from sigalg.processes import MarkovChain
        >>> rw = MarkovChain.random_walk(p=0.3, name="random_walk").from_enumeration(length=2)
        >>> rw # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'random_walk':
        time        0  1
        trajectory
        0          -1 -1
        1          -1  0
        2          -1  1
        3           0 -1
        4           0  0
        5           0  1
        6           1 -1
        7           1  0
        8           1  1
        >>> rw.probability_measure # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
        probability
        sample
        0               0.0
        1               0.0
        2               0.0
        3               0.7
        4               0.0
        5               0.3
        6               0.0
        7               0.0
        8               0.0
        """
        if states is not None and not isinstance(states, list):
            raise TypeError("states must be a list of three hashable values or None.")
        if states is not None and len(states) != 3:
            raise ValueError("states must contain exactly three hashable values.")
        if not (0 <= p <= 1):
            raise ValueError("p must be a float in the range [0, 1].")
        if states is None:
            states = [-1, 0, 1]

        sorted_states = sorted(states)
        transition_matrix = pd.DataFrame(
            [[0, 1 - p, p], [1 - p, 0, p], [1 - p, p, 0]],
            index=sorted_states,
            columns=sorted_states,
        )

        state_space = SampleSpace().from_list(sorted_states)
        probabilities = dict(zip(state_space, [0, 1, 0]))
        initial_distribution = ProbabilityMeasure(sample_space=state_space).from_dict(
            probabilities
        )

        return cls(
            transition_matrix=transition_matrix,
            initial_distribution=initial_distribution,
            domain=domain,
            index=index,
            name=name,
        )

    # @property
    # def stationary_distribution(self) -> pd.Series:
    #     P = self._transition_matrix.values
    #     eigenvalues, eigenvectors = np.linalg.eig(P.T)
    #     stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    #     stationary = np.real(eigenvectors[:, stationary_idx])
    #     stationary = stationary / stationary.sum()
    #     return pd.Series(stationary, index=self._states)

    # @property
    # def is_irreducible(self) -> bool:
    #     P = self._transition_matrix.values
    #     n = len(P)
    #     reachability = np.linalg.matrix_power(P > 0, n)
    #     return np.all(reachability > 0)

    # @property
    # def is_aperiodic(self) -> bool:
    #     P = self._transition_matrix.values
    #     n = len(P)
    #     for i in range(n):
    #         powers_sum = sum(
    #             np.linalg.matrix_power(P, k)[i, i] > 0 for k in range(1, n + 1)
    #         )
    #         if powers_sum > 1:
    #             return True
    #     return False

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated Markov chain" if self.is_enumerated else "Markov chain"
        return f"{prefix} process '{self.name}'"
