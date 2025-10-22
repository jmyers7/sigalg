"""
Base class for all stochastic processes.

This module provides the abstract base class that defines the interface for
simulating and visualizing stochastic processes. Concrete implementations for
discrete-time and continuous-time processes extend this class.
"""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from abc import ABC, abstractmethod
import matplotlib


class StochasticProcess(ABC):
    """
    Base class for stochastic processes.

    Parameters
    ----------
    trajectory_length : int or None, default=None
        Length of each trajectory. Must be set as an int before simulation or
        before computing sample spaces (for discrete-time processes).

    Attributes
    ----------
    trajectory_length : int or None, default=None
        Length of each trajectory. Must be set as an int before simulation or
        before computing sample spaces (for discrete-time processes).
    trajectories : pandas.DataFrame or None
        Simulated trajectories, populated after simulate() called
    num_trajectories : int or None
        Number of simulated trajectories, populated after simulate() called
    initial_time : int
        Starting time index for trajectories (0 or 1), set during simulate()
    """

    def __init__(self, trajectory_length=None):
        """Initialize the stochastic process."""
        self.trajectory_length = trajectory_length
        self.trajectories = None
        self.num_trajectories = None
        self.initial_time = None

    @abstractmethod
    def simulate(
        self,
        trajectory_length=None,
        num_trajectories=10,
        initial_time=1,
        column_prefix="X",
    ):
        """
        Generate simulated trajectories of the stochastic process and store
        them in self.trajectories. Also sets self.num_trajectories and self.initial_time.

        Parameters
        ----------
        trajectory_length : int or None, optional
            Length of each trajectory; if None, uses self.trajectory_length
        num_trajectories : int, default=10
            Number of independent trajectories to simulate
        initial_time : int, default=1
            Starting time index for column names (0 or 1)
        column_prefix : str, default="X"
            Prefix for column names (e.g., "X" gives "X0", "X1", ...)

        Returns
        -------
        self
            Returns self for method chaining (subclasses should implement)
        """
        pass

    def plot_simulations(self, ax=None, colors=None, plot_kwargs=None):
        """
        Visualize simulated trajectories.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, required
            Axes object to plot on
        colors : list of str, optional
            Color specification. If single-element list, all trajectories use
            that color. If multi-element list, creates interpolated colormap.
        plot_kwargs : dict, optional
            Additional options for plotting

        Raises
        ------
        ValueError
            If ax is not provided or if simulate() has not been called
        """
        if not isinstance(ax, matplotlib.axes.Axes):
            raise ValueError("ax must be provided and be a matplotlib Axes object")
        if self.trajectories is None:
            raise ValueError("simulate() must be called before plotting")

        # Handle None values for kwargs
        if plot_kwargs is None:
            plot_kwargs = {}

        # Handle color mapping
        if colors is not None:
            if not isinstance(colors, list):
                raise ValueError("colors must be a list")
            if len(colors) == 1:
                colors = [colors[0]] * self.num_trajectories
            else:
                custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                if self.num_trajectories == 1:
                    colors = [custom_cmap(0)]
                else:
                    colors = [
                        custom_cmap(i / (self.num_trajectories - 1))
                        for i in range(self.num_trajectories)
                    ]

        # Plot trajectories (iterate over DataFrame rows)
        for i, (idx, row) in enumerate(self.trajectories.iterrows()):
            if self.initial_time is None:
                raise ValueError(
                    "initial_time must be set by calling simulate() before plotting"
                )
            x_values = range(self.initial_time, self.initial_time + len(row))
            if colors is not None:
                ax.plot(x_values, row.values, color=colors[i], **plot_kwargs)
            else:
                ax.plot(x_values, row.values, **plot_kwargs)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("time")
        ax.set_ylabel("state")
        ax.set_title(self._get_plot_title())

    def _get_plot_title(self):
        """
        Generate plot title.

        Returns
        -------
        str
            Plot title including process type and parameters
        """
        return "stochastic process"
