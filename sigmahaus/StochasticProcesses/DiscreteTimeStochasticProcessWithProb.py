"""
Base class for discrete-time stochastic processes with probability methods.

Extends DiscreteTimeStochasticProcess with methods for sample space enumeration,
and joint probability computation.
"""

from .DiscreteTimeStochasticProcess import DiscreteTimeStochasticProcess
from abc import abstractmethod
import pandas as pd
from itertools import product

# Set higher precision for probability calculations
pd.set_option("display.precision", 10)


class DiscreteTimeStochasticProcessWithProb(DiscreteTimeStochasticProcess):
    """
    Base class for discrete-time stochastic processes with probability methods.

    Extends DiscreteTimeStochasticProcess with methods for sample space enumeration,
    and joint probability computation.

    Parameters
    ----------
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
        Sample space containing all possible sequences with probabilities.
        Populated by setup_sample_space()

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of setup_sample_space().

    See Also
    --------
    DiscreteTimeStochasticProcess : Parent class for all processes
    """

    def __init__(self, trajectory_length=None):
        """
        Initialize discrete-time process.

        Parameters
        ----------
        trajectory_length : int or None, default=None
            Length of each trajectory. Must be set as an int before simulation() or
            setup_sample_space().
        """
        super().__init__(trajectory_length)

    def setup_sample_space(self, trajectory_length=None):
        """
        Generate complete sample space.

        Creates DataFrame with all possible sequences and their joint
        probabilities. Feasible only for small trajectory_length.

        Parameters
        ----------
        trajectory_length : int or None, default=None
            Length of each trajectory; if None, uses self.trajectory_length

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
            lambda row: self.joint_prob(row.tolist()),
            axis=1,
        )
        return self

    @abstractmethod
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
        pass