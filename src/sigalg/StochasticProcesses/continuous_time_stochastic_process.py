"""
Base class for continuous-time stochastic processes.

This module provides the base class for processes evolving in continuous time,
with methods for time discretization and numerical simulation. Future
implementations will include Brownian motion and geometric Brownian motion.
"""

from .stochastic_process import StochasticProcess
import numpy as np


class ContinuousTimeStochasticProcess(StochasticProcess):
    """
    Base class for continuous-time stochastic processes.

    Extends StochasticProcess for processes evolving continuously in time.
    Uses numerical discretization for simulation and visualization.

    Parameters
    ----------
    T : float
        Time horizon [0, T]
    dt : float, optional
        Time discretization step. If None, defaults to T/1000.

    Attributes
    ----------
    T : float
        Time horizon
    dt : float
        Discretization step
    time_grid : numpy.ndarray
        Discrete time points for simulation

    Notes
    -----
    Future implementations will include Brownian motion, geometric Brownian
    motion, and other continuous-time processes. Sample space enumeration and
    conditional expectations are not applicable for continuous-time processes.

    See Also
    --------
    StochasticProcess : Parent class for all processes
    """

    def __init__(self, T, dt=None):
        """
        Initialize continuous-time process.

        Parameters
        ----------
        T : float
            Time horizon
        dt : float, optional
            Time step for discretization
        """
        super().__init__()
        self.T = T
        self.dt = dt if dt is not None else T / 1000
        self.time_grid = np.arange(0, T + self.dt, self.dt)

    def _get_x_values(self, series):
        """X-axis is continuous time grid."""
        return self.time_grid
