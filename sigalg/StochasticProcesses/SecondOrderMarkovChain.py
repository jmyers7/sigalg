"""
Second-order Markov chain module.

This module defines the SecondOrderMarkovChain class, a discrete-time stochastic
process characterized by the second-order Markov property. It includes methods for
sample space generation, joint probability computation, and simulation of
multiple chains.
"""

from .MarkovChain import MarkovChain
import numpy as np
import pandas as pd


class SecondOrderMarkovChain(MarkovChain):
    """
    Second-order Markov chain.

    Parameters
    ----------
    memory_2_transition_tensor : array-like
        Memory-2 transition probability tensor, shape (num_states, num_states, num_states)
    memory_1_transition_tensor : array-like
        Memory-1 transition probability matrix, shape (num_states, num_states)
    init_prob : array-like
        Initial state probabilities, shape (num_states,)
    trajectory_length : int or None, default=None
        Length of each trajectory. Must be set as an int before simulation() or
        setup_sample_space().

    Attributes
    ----------
    memory_2_transition_tensor : array-like
        Memory-2 transition probability tensor
    memory_1_transition_tensor : array-like
        Memory-1 transition probability matrix
    init_prob : array-like
        Initial state probabilities
    num_states : int
        Number of states
    order : int
        Order of the Markov chain, always 2 for this class
    num_trajectories : int
        Number of simulated trajectories, populated after simulate() called
    trajectories : pandas.DataFrame or None
        Simulated trajectories where each row is a trajectory, populated after simulate() called
    omega : pandas.DataFrame or None
        Sample space containing all possible sequences and probabilities.
        Populated by setup_sample_space()
    trajectory_length : int or None, default=None
        Length of each trajectory. Must be set as an int before simulation() or
        setup_sample_space().

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of setup_sample_space().

    See Also
    --------
    MarkovChain : Parent class for discrete-time processes

    Examples
    --------
    >>> memory_2_transition_tensor = np.array(
    ...    [[[0.7, 0.3], [0.4, 0.6]], [[0.2, 0.8], [0.5, 0.5]]]
    ... )
    >>> memory_1_transition_tensor = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> init_prob = np.array([0.6, 0.4])
    >>> mc = SecondOrderMarkovChain(
    ...     memory_2_transition_tensor,
    ...     memory_1_transition_tensor,
    ...     init_prob,
    ...     trajectory_length=5,
    ... )
    >>> mc.setup_sample_space()
    >>> print(mc.omega)
        X1  X2  X3  X4  X5        p
    0    0   0   0   0   0  0.14406
    1    0   0   0   0   1  0.06174
    2    0   0   0   1   0  0.03528
    ...
    30   1   1   1   1   0  0.03000
    31   1   1   1   1   1  0.03000
    >>> mc.simulate(num_trajectories=10)
    >>> print(mc.trajectories)
       X1  X2  X3  X4  X5
    0   0   0   0   0   0
    1   1   1   0   0   0
    2   1   1   0   0   0
    3   0   0   0   1   1
    4   0   0   0   1   0
    5   0   0   1   1   1
    6   0   0   0   0   0
    7   1   1   1   0   1
    8   1   1   1   1   1
    9   1   0   0   0   0
    >>> _, ax = plt.subplots(figsize=(10, 6))
    >>> mc.simulate(num_trajectories=10)
    >>> mc.plot_simulations(ax=ax, simulation_kwargs={"cumulative": True})
    >>> plt.show()
    """

    def __init__(
        self,
        memory_2_transition_tensor,
        memory_1_transition_tensor,
        init_prob,
        trajectory_length=None,
    ):
        """
        Initialize Markov chain.

        Parameters
        ----------
        memory_2_transition_tensor : array-like
            Memory-2 transition probability tensor, shape (num_states, num_states, num_states)
        memory_1_transition_tensor : array-like
            Memory-1 transition probability matrix, shape (num_states, num_states)
        init_prob : array-like
            Initial state probabilities, shape (num_states,)
        trajectory_length : int or None, default=None
            Length of each trajectory. Must be set as an int before simulation() or
            setup_sample_space().
        """
        super().__init__(init_prob, trajectory_length)
        self._check_prob(
            memory_2_transition_tensor, memory_1_transition_tensor, init_prob
        )
        self.memory_2_transition_tensor = memory_2_transition_tensor
        self.memory_1_transition_tensor = memory_1_transition_tensor
        self.order = 2

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
        prob = self.init_prob[X[0]]
        if len(X) > 1:
            prob *= self.memory_1_transition_tensor[X[0], X[1]]

        if len(X) > 2:
            for i in range(2, len(X)):
                prob *= self.memory_2_transition_tensor[X[i - 2], X[i - 1], X[i]]
        return prob

    def simulate(self, trajectory_length=None, num_trajectories=10):
        """
        Simulate multiple Markov chains, stores results in self.trajectories,
        along with num_trajectories in self.num_trajectories.

        Parameters
        ----------
        trajectory_length : int or None, default=None
            Length of each trajectory; if None, uses self.trajectory_length
        num_trajectories : int, default=10
            Number of trajectories to simulate

        Returns
        -------
        self
            Returns self for method chaining
        """
        if (trajectory_length is None) and (self.trajectory_length is None):
            raise ValueError(
                "trajectory_length must be specified either as an argument or "
                "as an attribute before simulation."
            )
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length

        chains = np.zeros((num_trajectories, self.trajectory_length), dtype=int)

        chains[:, 0] = np.random.choice(
            self.num_states, size=num_trajectories, p=self.init_prob
        )

        chains[:, 1] = np.array(
            [
                np.random.choice(
                    self.num_states,
                    p=self.memory_1_transition_tensor[chains[i, 0]],
                )
                for i in range(num_trajectories)
            ]
        )

        for t in range(2, self.trajectory_length):
            chains[:, t] = np.array(
                [
                    np.random.choice(
                        self.num_states,
                        p=self.memory_2_transition_tensor[
                            chains[i, t - 2], chains[i, t - 1]
                        ],
                    )
                    for i in range(num_trajectories)
                ]
            )

        # Convert to DataFrame
        columns = [f"X{i+1}" for i in range(self.trajectory_length)]
        self.trajectories = pd.DataFrame(chains, columns=columns)
        self.num_trajectories = num_trajectories
        return self

    def _check_prob(
        self, memory_2_transition_tensor, memory_1_transition_tensor, init_prob
    ):
        """
        Check if probabilities are valid.

        Parameters
        ----------
        memory_2_transition_tensor : array-like
            Memory-2 transition probability tensor, shape (num_states, num_states, num_states)
        memory_1_transition_tensor : array-like
            Memory-1 transition probability matrix, shape (num_states, num_states)
        init_prob : array-like
            Initial probabilities, shape (num_states,)

        Raises
        ------
        ValueError
            If initial probabilities are invalid
        """
        # Check for negative probabilities
        if np.any(init_prob < 0):
            raise ValueError("Initial probabilities must be non-negative")
        if np.any(memory_1_transition_tensor < 0):
            raise ValueError("Memory-1 transition probabilities must be non-negative")
        if np.any(memory_2_transition_tensor < 0):
            raise ValueError("Memory-2 transition probabilities must be non-negative")

        # Check shapes
        if memory_1_transition_tensor.shape != (self.num_states, self.num_states):
            raise ValueError(
                "Memory-1 transition matrix must have shape (num_states, num_states)"
            )
        if memory_2_transition_tensor.shape != (
            self.num_states,
            self.num_states,
            self.num_states,
        ):
            raise ValueError(
                "Memory-2 transition tensor must have shape (num_states, num_states, num_states)"
            )

        # Check sums
        if not np.allclose(np.sum(init_prob), 1):
            raise ValueError("Initial probabilities must sum to 1")
        if not np.allclose(np.sum(memory_1_transition_tensor, axis=1), 1):
            raise ValueError("Rows of memory-1 transition matrix must sum to 1")
        if not np.allclose(np.sum(memory_2_transition_tensor, axis=2), 1):
            raise ValueError(
                "Slices of memory-2 transition tensor must sum to 1 along last axis"
            )