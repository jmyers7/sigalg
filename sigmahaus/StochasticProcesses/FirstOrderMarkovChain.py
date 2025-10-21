"""
First-order Markov chain module.

This module defines the FirstOrderMarkovChain class, a discrete-time stochastic
process characterized by the first-order Markov property. It includes methods for
sample space generation, joint probability computation, and simulation of
multiple chains.
"""

from .MarkovChain import MarkovChain
import numpy as np


class FirstOrderMarkovChain(MarkovChain):
    """
    First-order Markov chain.

    Parameters
    ----------
    transition_matrix : array-like
        One-step transition probability matrix, shape (num_states, num_states)
    init_prob : array-like
        Initial state probabilities, shape (num_states,)
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
    order : int
        Order of the Markov chain, always 1 for this class
    num_trajectories : int
        Number of simulated trajectories, populated after simulate() called
    trajectories : np.ndarray
        Simulated trajectories, populated after simulate() called

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of setup_sample_space().

    See Also
    --------
    MarkovChain : Parent class for discrete-time processes

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
    >>> mc.simulate(num_trajectories=5)
    >>> print(mc.trajectories)
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
    >>> _, ax = plt.subplots()
    >>> mc.plot_simulations(ax=ax, kwargs={"cumulative": True})
    >>> plt.show()
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
        super().__init__(init_prob, trajectory_length)
        self._check_prob(init_prob, transition_matrix)
        self.transition_matrix = transition_matrix
        self.order = 1

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
            for i in range(1, len(X)):
                prob *= self.transition_matrix[X[i - 1], X[i]]
        return prob

    def simulate(self, num_trajectories=10):
        """
        Simulate multiple Markov chains, stores results in self.trajectories,
        along with num_trajectories in self.num_trajectories.

        Parameters
        ----------
        num_trajectories : int, default=10
            Number of trajectories to simulate
        """
        chains = np.zeros((num_trajectories, self.trajectory_length), dtype=int)

        chains[:, 0] = np.random.choice(
            self.num_states, size=num_trajectories, p=self.init_prob
        )

        for t in range(1, self.trajectory_length):
            chains[:, t] = np.array(
                [
                    np.random.choice(
                        self.num_states,
                        p=self.transition_matrix[chains[i, t - 1]],
                    )
                    for i in range(num_trajectories)
                ]
            )

        self.trajectories = chains
        self.num_trajectories = num_trajectories

    def _check_prob(self, init_prob, transition_matrix):
        """
        Check if probabilities are valid.

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
        # Check for negative probabilities
        if np.any(init_prob < 0):
            raise ValueError("Initial probabilities must be non-negative")
        if np.any(transition_matrix < 0):
            raise ValueError("Transition probabilities must be non-negative")

        # Check for shape
        if transition_matrix.shape != (self.num_states, self.num_states):
            raise ValueError("Transition matrix shape must be (num_states, num_states)")

        # Check sums
        if not np.allclose(np.sum(init_prob), 1):
            raise ValueError("Initial probabilities must sum to 1")
        if not np.allclose(np.sum(transition_matrix, axis=1), 1):
            raise ValueError("Each row of transition matrix must sum to 1")
