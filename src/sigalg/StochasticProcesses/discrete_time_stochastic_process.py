"""
Base class for discrete-time stochastic processes.

Abstract class extending StochasticProcess with methods specific to discrete-time processes
including sample space enumeration for small trajectory lengths.
"""

from .stochastic_process import StochasticProcess
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
        generate_sample_space().

    Attributes
    ----------
    trajectory_length : int or None, default=None
        Length of each trajectory. Must be set as an int before simulation() or
        generate_sample_space().
    trajectories : pandas.DataFrame or None
        Simulated trajectories where each row is a trajectory, populated after simulate() called
    num_trajectories : int or None
        Number of simulated trajectories, populated after simulate() called
    omega : pandas.DataFrame or None
        Sample space containing all possible sequences.
        Populated by generate_sample_space()
    initial_time : int
        Starting time index for trajectories (0 or 1), set during simulate()

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of generate_sample_space().

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
            generate_sample_space().
        """
        super().__init__(trajectory_length)
        self.omega = None

    @abstractmethod
    def generate_sample_space(
        self, trajectory_length=None, initial_time=1, column_prefix="X"
    ):
        """
        Generate complete sample space.

        Creates DataFrame with all possible sequences. Feasible only for small
        trajectory_length. If used on an object from the subclass
        DiscreteTimeStochasticProcessWithProb, will also include probabilities
        for each sequence.

        Parameters
        ----------
        trajectory_length : int, optional
            Number of time steps; if None, uses self.trajectory_length
        initial_time : int, default=1
            Starting time index for column names (0 or 1)
        column_prefix : str, default="X"
            Prefix for column names (e.g., "X" gives "X0", "X1", ...)

        Returns
        -------
        self
            Returns self for method chaining (subclasses need to implement)
        """
        pass
