"""
Discrete IID (Independent and Identically Distributed) process module.

This module defines the DiscreteIID class, a discrete-time stochastic process
where each time step is independently drawn from the same probability distribution.
It includes methods for sample space generation, joint probability computation,
and simulation of multiple trajectories.
"""

from .DiscreteTimeStochasticProcessWithProb import DiscreteTimeStochasticProcessWithProb
import numpy as np
import pandas as pd


class DiscreteIID(DiscreteTimeStochasticProcessWithProb):
    """
    Discrete IID (Independent and Identically Distributed) process.

    Each time step is independently drawn from the same probability distribution.

    Parameters
    ----------
    prob : array-like
        Probability mass function on states, shape (num_states,)
    trajectory_length : int or None, default=None
        Length of each trajectory. Must be set as an int before simulation() or
        setup_sample_space().

    Attributes
    ----------
    trajectory_length : int or None, default=None
        Length of each trajectory. Must be set as an int before simulation() or
        setup_sample_space().
    trajectories : pandas.DataFrame or None
        Simulated trajectories where each row is a trajectory, populated after simulate() called
    num_trajectories : int or None
        Number of simulated trajectories, populated after simulate() called
    omega : pandas.DataFrame or None
        Sample space containing all possible sequences and probabilities.
        Populated by setup_sample_space()
    prob : array-like
        Probability mass function on states, shape (num_states,)
    num_states : int
        Number of states

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of setup_sample_space().

    See Also
    --------
    DiscreteTimeStochasticProcessWithProb : Parent class for discrete-time processes
    """

    def __init__(self, prob, trajectory_length=None):
        """
        Initialize IID process.

        Parameters
        ----------
        prob : array-like
            Probability mass function on states, shape (num_states,)
        trajectory_length : int or None, default=None
            Length of each trajectory. Must be set as an int before simulation() or
            setup_sample_space().
        """
        super().__init__(trajectory_length)
        self._check_prob(prob)
        self.prob = prob
        self.num_states = prob.shape[0]

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
        prob = 1.0
        for x in X:
            prob *= self.prob[x]
        return prob

    def simulate(self, trajectory_length=None, num_trajectories=10):
        """
        Simulate multiple trajectories, stores results in self.trajectories,
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

        chains = np.random.choice(
            self.num_states,
            size=(num_trajectories, self.trajectory_length),
            p=self.prob,
        )

        columns = [f"X{i+1}" for i in range(self.trajectory_length)]
        self.trajectories = pd.DataFrame(chains, columns=columns)
        self.num_trajectories = num_trajectories
        return self

    def _check_prob(self, prob):
        """
        Check if probabilities are valid.

        Parameters
        ----------
        prob : array-like
            Probability mass function on states, shape (num_states,)

        Raises
        ------
        ValueError
            If probabilities are invalid
        """
        # Convert to numpy array if needed
        prob = np.asarray(prob)

        # Check shape
        if prob.ndim != 1:
            raise ValueError(f"prob must be 1-dimensional, got shape {prob.shape}")

        # Check for negative probabilities
        if np.any(prob < 0):
            raise ValueError("Probabilities must be non-negative")

        # Check sums
        if not np.allclose(np.sum(prob), 1):
            raise ValueError("Probabilities must sum to 1")

    def _get_plot_title(self, **kwargs):
        """Get title for IID plot."""
        default_title = f"discrete IID process (number of states={self.num_states})"
        return kwargs.get("title", default_title)
