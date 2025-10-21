"""
Transformed discrete-time stochastic process module.

This module defines the TransformedDiscreteTimeStochasticProcess class, which
applies a transformation function to trajectories from an existing discrete-time
stochastic process. This enables creating derived processes like cumulative sums,
differences, or other functional transformations.
"""

from .DiscreteTimeStochasticProcess import DiscreteTimeStochasticProcess


class TransformedDiscreteTimeStochasticProcess(DiscreteTimeStochasticProcess):
    """
    Transformed discrete-time stochastic process.

    Applies a transformation function to trajectories from an underlying process.

    Parameters
    ----------
    process : DiscreteTimeStochasticProcess
        The original stochastic process to transform
    transform_func : Callable
        Function that takes a DataFrame and returns a transformed DataFrame.
        Applied element-wise or column-wise to trajectories.

    Attributes
    ----------
    trajectory_length : int or None
        Length of each trajectory, inherited from original process
    trajectories : pandas.DataFrame or None
        Transformed simulated trajectories where each row is a trajectory,
        populated after simulate() called
    num_trajectories : int or None
        Number of simulated trajectories, populated after simulate() called
    omega : pandas.DataFrame or None
        Transformed sample space containing all possible sequences.
        Populated by setup_sample_space()
    original_process : DiscreteTimeStochasticProcess
        The underlying stochastic process
    transform_func : Callable
        The transformation function applied to trajectories

    Notes
    -----
    The transformation function should preserve the DataFrame structure and
    return a DataFrame with the same shape as the input.

    See Also
    --------
    DiscreteTimeStochasticProcess : Parent class for discrete-time processes
    transform_discrete_process : Factory function for creating transformed processes
    """

    def __init__(self, process, transform_func):
        """
        Initialize transformed process.

        Parameters
        ----------
        process : DiscreteTimeStochasticProcess
            The original stochastic process to transform
        transform_func : callable
            Function that takes a DataFrame and returns a transformed DataFrame
        """
        super().__init__(trajectory_length=process.trajectory_length)
        self.original_process = process
        self.transform_func = transform_func

    def setup_sample_space(self, trajectory_length=None):
        """
        Generate complete sample space with transformation applied.

        Creates the sample space from the original process and applies the
        transformation function. Note that probabilities are not preserved
        in the transformed space.

        Parameters
        ----------
        trajectory_length : int or None, default=None
            Number of time steps; if None, uses self.trajectory_length

        Returns
        -------
        self
            Returns self for method chaining
        """
        # Get original sample space
        if self.original_process.omega is None:
            self.original_process.setup_sample_space(trajectory_length)

        # Transform the values
        self.omega = self.original_process.omega.copy()
        self.omega.drop("p", axis=1, inplace=True)
        self.omega = self.transform_func(self.omega)
        return self

    def simulate(self, trajectory_length=None, num_trajectories=10):
        """
        Simulate multiple transformed trajectories.

        Simulates from the original process and applies the transformation
        function to the results. Stores transformed trajectories in
        self.trajectories.

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
        # Simulate from original and transform
        self.original_process.simulate(trajectory_length, num_trajectories)
        self.trajectories = self.transform_func(self.original_process.trajectories)
        self.num_trajectories = num_trajectories
        return self
