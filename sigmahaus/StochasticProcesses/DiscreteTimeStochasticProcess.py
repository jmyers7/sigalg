"""
Base class for discrete-time stochastic processes.

Abstract class extending StochasticProcess with methods specific to discrete-time processes
including sample space enumeration for small trajectory lengths.
"""

from .StochasticProcess import StochasticProcess
from abc import abstractmethod
import pandas as pd

# Set higher precision for probability calculations
pd.set_option("display.precision", 10)


class DiscreteTimeStochasticProcess(StochasticProcess):
    """
    Base class for discrete-time stochastic processes.

    Abstract class extending StochasticProcess with methods specific to discrete-time processes
    including sample space enumeration for small trajectory lengths.

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
        Sample space containing all possible sequences.
        Populated by setup_sample_space()

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of setup_sample_space().

    See Also
    --------
    StochasticProcess : Parent class for all processes
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
        self.omega = None

    @abstractmethod
    def setup_sample_space(self, trajectory_length=None):
        """
        Generate complete sample space.

        Creates DataFrame with all possible sequences. Feasible only for small
        trajectory_length.

        Parameters
        ----------
        trajectory_length : int, optional
            Number of time steps; if None, uses self.trajectory_length

        Returns
        -------
        self
            Returns self for method chaining (subclasses need to implement)
        """
        pass