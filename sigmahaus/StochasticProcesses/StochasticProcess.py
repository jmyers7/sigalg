"""
Base class for all stochastic processes.

This module provides the abstract base class that defines the interface for
simulating and visualizing stochastic processes. Concrete implementations for
discrete-time and continuous-time processes extend this class.
"""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib
from abc import ABC, abstractmethod


class StochasticProcess(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize the stochastic process."""
        pass

    @abstractmethod
    def simulate(self, num_trajectories=10):
        """
        Generate sample paths.

        Parameters
        ----------
        num_trajectories : int, default=10
            Number of independent trajectories to simulate

        Returns
        -------
        array-like
            Simulated trajectories. Format depends on subclass implementation.
        """
        pass

    def plot_simulations(
        self, num_trajectories=10, ax=None, colors=None, kwargs=None, plot_kwargs=None
    ):
        """
        Visualize simulated trajectories.

        Parameters
        ----------
        num_trajectories : int, default=10
            Number of trajectories to simulate and plot
        ax : matplotlib.axes.Axes, required
            Axes object to plot on
        colors : list of str, optional
            Color specification. If single-element list, all trajectories use
            that color. If multi-element list, creates interpolated colormap.
        kwargs : dict, optional
            Additional options for subclasses, e.g., cumulative plotting
        plot_kwargs : dict, optional
            Additional options for plotting

        Raises
        ------
        ValueError
            If ax is not provided
        """
        if not isinstance(ax, matplotlib.axes.Axes):
            raise ValueError("ax must be provided and be a matplotlib Axes object")

        # Handle None values for kwargs
        if kwargs is None:
            kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        trajectories = self.simulate(num_trajectories)
        plot_data, ylabel = self._get_plot_data(trajectories, **kwargs)

        # Handle color mapping
        if colors is not None:
            if not isinstance(colors, list):
                raise ValueError("colors must be a list")
            if len(colors) == 1:
                colors = [colors[0]] * num_trajectories
            else:
                custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                if num_trajectories == 1:
                    colors = [custom_cmap(0)]
                else:
                    colors = [
                        custom_cmap(i / (num_trajectories - 1))
                        for i in range(num_trajectories)
                    ]

        # Plot trajectories
        for i, series in enumerate(plot_data):
            x_values = self._get_x_values(series)
            if colors is not None:
                ax.plot(x_values, series, color=colors[i], **plot_kwargs)
            else:
                ax.plot(x_values, series, **plot_kwargs)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(self._get_x_label(**kwargs))
        ax.set_ylabel(ylabel)
        ax.set_title(self._get_plot_title(**kwargs))

    def _get_plot_data(self, trajectories, **kwargs):
        """
        Transform trajectories for plotting.

        Parameters
        ----------
        trajectories : array-like
            Raw simulation output
        **kwargs : dict
            Process-specific options

        Returns
        -------
        plot_data : array-like
            Transformed data to plot
        ylabel : str
            Y-axis label

        Notes
        -----
        Subclasses override this to compute cumulative sums, extract components,
        or apply other transformations.
        """
        return trajectories, "value"

    def _get_x_values(self, series):
        """
        Define x-axis values for a trajectory.

        Parameters
        ----------
        series : array-like
            One trajectory

        Returns
        -------
        array-like
            X-axis values (typically time steps)

        Notes
        -----
        Default returns discrete indices. Subclasses override for continuous
        time or to include initial time 0.
        """
        return range(len(series))

    def _get_x_label(self, **kwargs):
        """
        Define x-axis label.

        Parameters
        ----------
        **kwargs : dict
            Process-specific options

        Returns
        -------
        str
            X-axis label
        """
        return "time"

    def _get_plot_title(self, **kwargs):
        """
        Generate plot title.

        Parameters
        ----------
        **kwargs : dict
            Process-specific options

        Returns
        -------
        str
            Plot title including process type and parameters
        """
        return "stochastic process"
