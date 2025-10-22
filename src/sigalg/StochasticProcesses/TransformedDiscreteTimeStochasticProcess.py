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
    src_process : DiscreteTimeStochasticProcess
        The source stochastic process to transform
    transform_func : Callable
        Function that takes a DataFrame and returns a transformed DataFrame.
        Applied element-wise or column-wise to trajectories.

    Attributes
    ----------
    trajectory_length : int or None
        The trajectory_length is not set until generate_sample_space() or
        simulate() is called, as it may differ from the source process
        depending on the transformation.
    trajectories : pandas.DataFrame or None
        Transformed simulated trajectories where each row is a trajectory,
        populated after simulate() called
    num_trajectories : int or None
        Number of simulated trajectories, populated after simulate() called
    omega : pandas.DataFrame or None
        Transformed sample space containing all possible sequences.
        Populated by generate_sample_space()
    src_process : DiscreteTimeStochasticProcess
        The underlying stochastic process
    transform_func : Callable
        The transformation function applied to trajectories
    initial_time : int or None
        Starting time index for trajectories (0 or 1), set during simulate() or
        generate_sample_space()

    See Also
    --------
    DiscreteTimeStochasticProcess : Parent class for discrete-time processes
    transform_discrete_process : Factory function for creating transformed processes
    """

    def __init__(self, src_process, transform_func):
        """
        Initialize transformed process.

        Parameters
        ----------
        src_process : DiscreteTimeStochasticProcess
            The source stochastic process to transform
        transform_func : callable
            Function that takes a DataFrame and returns a transformed DataFrame
            
        Notes
        -----
        The trajectory_length is not set until generate_sample_space() or
        simulate() is called, as it may differ from the source process
        depending on the transformation.
        """
        super().__init__(trajectory_length=None)
        self.src_process = src_process
        self.transform_func = transform_func

    def generate_sample_space(self):
        """
        Generate complete sample space with transformation applied.

        Returns
        -------
        self
            Returns self for method chaining
            
        Raises
        ------
        ValueError
            If generate_sample_space() has not been called on the source process
            
        Notes
        -----
        The new trajectory length and initial time may differ from the source 
        process depending on the transformation function. The initial time is 
        extracted from the first column name.
        """
        if self.src_process.omega is None:
            raise ValueError(
                "generate_sample_space() must be called on the source process "
                "before calling it on the transformed process"
            )

        self.omega = self.src_process.omega.copy()
        self.omega.drop("p", axis=1, inplace=True)
        self.omega = self.transform_func(self.omega)
        self.trajectory_length = len(self.omega.columns)
        
        # Extract initial_time from first column name
        first_col = self.omega.columns[0]
        self.initial_time = int(first_col[1:])
        
        return self

    def simulate(self):
        """
        Simulate multiple transformed trajectories.

        Returns
        -------
        self
            Returns self for method chaining
            
        Raises
        ------
        ValueError
            If simulate() has not been called on the source process
            
        Notes
        -----
        The trajectory length and initial time are determined by the 
        transformation function applied to the source process's trajectories.
        The initial time is extracted from the first column name.
        """
        if self.src_process.trajectories is None:
            raise ValueError(
                "simulate() must be called on the source process "
                "before calling it on the transformed process"
            )
        
        self.trajectories = self.transform_func(self.src_process.trajectories)
        self.trajectory_length = len(self.trajectories.columns)
        self.num_trajectories = len(self.trajectories)
        
        # Extract initial_time from first column name
        first_col = self.trajectories.columns[0]
        self.initial_time = int(first_col[1:])
        
        return self