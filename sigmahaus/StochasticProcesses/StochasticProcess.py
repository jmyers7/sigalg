"""
Base class for all stochastic processes.

This module provides the abstract base class that defines the interface for
simulating and visualizing stochastic processes. Concrete implementations for
discrete-time and continuous-time processes extend this class.
"""

from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from abc import ABC, abstractmethod


class StochasticProcess(ABC):
    """
    Abstract base class for stochastic processes.

    Defines the interface for simulation and visualization that all stochastic
    process implementations must follow. Subclasses implement specific process
    dynamics (Markov chains, Brownian motion, etc.).

    Notes
    -----
    This class uses the Template Method pattern: plot_simulations defines the
    overall plotting workflow while delegating process-specific details to
    abstract methods that subclasses must implement.

    Future extensions will include continuous-time processes like Brownian
    motion and geometric Brownian motion.

    See Also
    --------
    DiscreteTimeStochasticProcess : Base class for discrete-time processes
    ContinuousTimeStochasticProcess : Base class for continuous-time processes
    TwoStateMarkovChain : First-order Markov chain implementation
    TwoStateSecondOrderMarkovChain : Second-order Markov chain implementation
    ThreeWinStreakSelectionStrategy : Path-dependent betting strategy
    """

    @abstractmethod
    def __init__(self):
        """Initialize the stochastic process."""
        pass

    @abstractmethod
    def simulate(self, num_chains=10):
        """
        Generate sample paths.

        Parameters
        ----------
        num_chains : int, default=10
            Number of independent realizations to generate

        Returns
        -------
        array-like
            Simulated trajectories. Format depends on subclass implementation.

        Raises
        ------
        NotImplementedError
            Subclasses must implement this method
        """
        pass
        

    def plot_simulations(self, num_chains=10, ax=None, colors=None, alpha=1, **kwargs):
        """
        Visualize simulated trajectories.

        Parameters
        ----------
        num_chains : int, default=10
            Number of trajectories to simulate and plot
        ax : matplotlib.axes.Axes, required
            Axes object to plot on
        colors : list of str, optional
            Color specification. If single-element list, all trajectories use
            that color. If multi-element list, creates interpolated colormap.
        alpha : float, default=1
            Line transparency in [0, 1]
        **kwargs : dict
            Process-specific options (e.g., cumulative for Markov chains)

        Raises
        ------
        ValueError
            If ax is not provided

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sigmahaus import TwoStateMarkovChain
        >>> mc = TwoStateMarkovChain(chain_length=100)
        >>> fig, ax = plt.subplots()
        >>> mc.plot_simulations(num_chains=10, ax=ax, colors=['blue', 'red'])
        >>> plt.show()
        """

        if not isinstance(ax, matplotlib.axes.Axes):
            raise ValueError("ax must be provided and be a matplotlib Axes object")
            
        trajectories = self.simulate(num_chains)
        plot_data, ylabel = self._get_plot_data(trajectories, **kwargs)

        # Handle color mapping
        if colors is not None:
            if not isinstance(colors, list):
                raise ValueError("colors must be a list")
            if len(colors) == 1:
                colors = [colors[0]] * num_chains
            else:
                custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                if num_chains == 1:
                    colors = [custom_cmap(0)]
                else:
                    colors = [
                        custom_cmap(i / (num_chains - 1)) for i in range(num_chains)
                    ]

        # Plot trajectories
        for i, series in enumerate(plot_data):
            x_values = self._get_x_values(series)
            if colors is not None:
                ax.plot(x_values, series, color=colors[i], alpha=alpha)
            else:
                ax.plot(x_values, series, alpha=alpha)

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
