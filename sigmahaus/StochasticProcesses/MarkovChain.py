"""
Markov chain implementation.

This module defines the MarkovChain class, a discrete-time stochastic
process characterized by the Markov property. It includes methods for
sample space generation, joint probability computation, and simulation of
multiple chains.
"""

from .DiscreteTimeStochasticProcess import DiscreteTimeStochasticProcess
import pandas as pd
import numpy as np
from itertools import product


class MarkovChain(DiscreteTimeStochasticProcess):
    """
    Discrete-time Markov chain.

    Parameters
    ----------
    transition_matrix : array-like
        One-step transition probability matrix
    init_prob : array-like
        Initial state probabilities
    trajectory_length : int, default=3
        Number of time steps

    Attributes
    ----------
    transition_matrix : array-like
        One-step transition probability matrix
    init_prob : array-like
        Initial state probabilities
    num_states : int
        Number of states

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of setup_sample_space().

    See Also
    --------
    DiscreteTimeStochasticProcess : Parent class for discrete-time processes

    Examples
    --------
    >>> transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> init_prob = np.array([0.6, 0.4])
    >>> chain_length = 3
    >>> mc = MarkovChain(transition_matrix, init_prob, chain_length)
    >>> mc.setup_sample_space()
    >>> print(mc.omega)
         X1  X2  X3     p
    0    0   0   0  0.294
    1    0   0   1  0.168
    2    0   1   0  0.072
    ...
    7    1   1   1  0.144
    >>> trajectories = mc.simulate(num_trajectories=5)
    >>> print(trajectories)
    [[0 0 0]
     [1 0 1]
     [1 1 1]
     [0 0 0]
     [0 1 0]]
    >>> # Define a function for conditional expectation
    >>> def Y(omega):
    ...     return omega["X1"] + omega["X2"] + omega["X3"]
    >>> cond_exp = mc.conditional_expectation(Y, ["X1", "X2"])
    >>> print(cond_exp)
    X1  X2
    0   0     0.363636
        1     1.666667
    1   0     1.363636
        1     2.666667
    """

    def __init__(self, transition_matrix, init_prob, trajectory_length=3):
        """
        Initialize Markov chain.

        Parameters
        ----------
        transition_matrix : array-like
            One-step transition probability matrix
        init_prob : array-like
            Initial state probabilities
        trajectory_length : int, default=3
            Number of time steps

        """
        self._check_initial_prob(init_prob, transition_matrix)
        super().__init__(trajectory_length)
        self.transition_matrix = transition_matrix
        self.num_states = transition_matrix.shape[0]
        self.init_prob = init_prob

    def setup_sample_space(self):
        """
        Generate complete sample space.

        Creates DataFrame with all possible sequences and their joint
        probabilities. Feasible only for small trajectory_length.

        Returns
        -------
        self
            Returns self for method chaining
        """
        omega_cardinality = self.num_states**self.trajectory_length
        if omega_cardinality > 1000:  # Reasonable threshold
            raise ValueError(
                "Sample space size exceeds threshold 1000. Use simulate() instead."
            )
        # Generate all possible sequences
        sequences = list(
            product(list(range(self.num_states)), repeat=self.trajectory_length)
        )
        # Create DataFrame with all possible sequences and their joint probabilities
        column_names = [f"X{i + 1}" for i in range(self.trajectory_length)]
        self.omega = pd.DataFrame(sequences, columns=column_names)
        self.omega["p"] = self.omega[column_names].apply(
            lambda row: self.joint_prob(row.tolist()), axis=1
        )
        return self

    def joint_prob(self, X):
        """
        Compute joint probability P(X_1, ..., X_n).

        Parameters
        ----------
        X : array-like
            State sequence

        Returns
        -------
        float
            Joint probability
        """
        prob = self._init_prob(X[0])
        for i in range(1, len(X)):
            prob *= self._trans_prob(X[i], X[i - 1])
        return prob

    def simulate(self, num_trajectories=1):
        """
        Simulate multiple Markov chains.

        Parameters
        ----------
        num_trajectories : int, default=1
            Number of trajectories to simulate

        Returns
        -------
        array-like
            Simulated trajectories of shape (num_trajectories, trajectory_length)
        """
        chains = np.zeros((num_trajectories, self.trajectory_length), dtype=int)

        chains[:, 0] = np.random.choice(
            self.num_states, size=num_trajectories, p=self.init_prob
        )

        for t in range(1, self.trajectory_length):
            for i in range(num_trajectories):
                current_state = chains[i, t - 1]
                chains[i, t] = np.random.choice(
                    self.num_states,
                    p=self.transition_matrix[current_state],
                )

        return chains

    def _check_initial_prob(self, init_prob, transition_matrix):
        """
        Check if initial probabilities are valid.

        Parameters
        ----------
        init_prob : array-like
            Initial probabilities
        transition_matrix : array-like
            Transition matrix

        Raises
        ------
        ValueError
            If initial probabilities are invalid
        """
        if not np.allclose(np.sum(init_prob), 1):
            raise ValueError("Initial probabilities must sum to 1")
        if len(init_prob) != transition_matrix.shape[0]:
            raise ValueError("Initial probabilities length must match number of states")
        if np.any(init_prob < 0):
            raise ValueError("Initial probabilities must be non-negative")
        if np.any(transition_matrix < 0):
            raise ValueError("Transition probabilities must be non-negative")
        if not np.allclose(np.sum(transition_matrix, axis=1), 1):
            raise ValueError("Each row of transition matrix must sum to 1")
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")

    def _trans_prob(self, current_state, next_state=None):
        """
        Compute one-step transition probability P(next_state | current_state).

        Parameters
        ----------
        current_state : int
            Current state
        next_state : int, optional
            Next state

        Returns
        -------
        float
            Transition probability P(next_state | current_state)
        """
        if next_state is None:
            return self.transition_matrix[current_state, :]
        return self.transition_matrix[current_state, next_state]

    def _init_prob(self, X):
        """
        Compute initial probability P(X).

        Parameters
        ----------
        X : int
            State

        Returns
        -------
        float
            Initial probability P(X)
        """
        return self.init_prob[X]

    def _get_plot_data(self, trajectories, cumulative=False, **kwargs):
        """Get plot data for Markov chain."""
        if cumulative:
            data = np.cumsum(trajectories, axis=1)
            ylabel = "cumulative sum"
            self._plot_type = "cumulative sums"  # Store for title
        else:
            data = trajectories
            ylabel = "state"
            self._plot_type = "states"
        return data, ylabel

    def _get_plot_title(self, **kwargs):
        """Get title for Markov chain plot."""
        plot_type = getattr(self, "_plot_type", "trajectories")
        return f"Markov chain {plot_type} (number of states={self.num_states + 1})"
