"""
Abstract base class for Markov chain implementation.

This module defines the MarkovChain class, a discrete-time stochastic
process characterized by the Markov property. It includes methods for
sample space generation, joint probability computation, and simulation of
multiple chains.
"""

from .discrete_time_stochastic_process_with_prob import DiscreteTimeStochasticProcessWithProb
from abc import abstractmethod


class MarkovChain(DiscreteTimeStochasticProcessWithProb):
    """
    Base class for Markov chains. Extends DiscreteTimeStochasticProcessWithProb.

    Parameters
    ----------
    init_prob : array-like
        Initial state probabilities
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
        Sample space containing all possible sequences and probabilities.
        Populated by generate_sample_space()
    init_prob : array-like
        Initial state probabilities
    num_states : int
        Number of states
    order : int
        Order of the Markov chain
    initial_time : int
        Starting time index for trajectories (0 or 1), set during simulate() or
        generate_sample_space()

    Notes
    -----
    Sample space enumeration has exponential complexity in trajectory_length.
    For long chains, use simulate() instead of generate_sample_space().

    See Also
    --------
    DiscreteTimeStochasticProcessWithProb : Parent class for discrete-time processes
    """

    def __init__(self, init_prob, trajectory_length=None):
        """
        Initialize Markov chain.

        Parameters
        ----------
        init_prob : array-like
            Initial state probabilities
        trajectory_length : int or None, default=None
            Length of each trajectory. Must be set as an int before simulation() or
            generate_sample_space().

        Notes
        -----
        Subclasses should set self.order.
        """
        super().__init__(trajectory_length)
        self.num_states = init_prob.shape[0]
        self.init_prob = init_prob
        self.order = None

    @abstractmethod
    def _check_prob(self, init_prob):
        """
        Check if initial probabilities are valid.

        Parameters
        ----------
        init_prob : array-like
            Initial probabilities

        Raises
        ------
        ValueError
            If initial probabilities are invalid

        Notes
        -----
        For subclasses, this method will accept higher-order transition structures.
        """
        pass

    def _get_plot_title(self):
        """Get title for Markov chain plot."""
        default_title = (
            f"order-{self.order} Markov chain (number of states={self.num_states})"
        )
        return default_title
