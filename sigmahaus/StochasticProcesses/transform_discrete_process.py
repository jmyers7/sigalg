"""
Factory function for creating transformed discrete-time stochastic processes.

This module provides a convenience function for applying transformations to
existing discrete-time stochastic processes, creating derived processes such as
cumulative sums, differences, or other functional transformations.
"""

from .DiscreteTimeStochasticProcessWithProb import DiscreteTimeStochasticProcessWithProb
from .TransformedDiscreteTimeStochasticProcess import (
    TransformedDiscreteTimeStochasticProcess,
)
from collections.abc import Callable


def transform_discrete_process(process, transform_func):
    """
    Create a transformed discrete stochastic process.

    Applies a transformation function to trajectories from an existing discrete-time
    stochastic process with probability measures. The transformation is applied to
    both simulated trajectories and the sample space (if generated).

    Parameters
    ----------
    process : DiscreteTimeStochasticProcessWithProb
        The original discrete stochastic process to be transformed. Must have
        probability measures defined.
    transform_func : callable
        Function that takes a DataFrame and returns a transformed DataFrame.
        Should preserve the DataFrame structure (same shape). Common examples:
        - Cumulative sum: lambda df: df.cumsum(axis=1)
        - Differences: lambda df: df.diff(axis=1)
        - Scaling: lambda df: df * 2
        - Custom transformations: any function operating on DataFrames

    Returns
    -------
    TransformedDiscreteTimeStochasticProcess
        A new discrete stochastic process with the transformation applied to
        trajectories and sample space.

    Raises
    ------
    ValueError
        If transform_func is not callable
    ValueError
        If process is not an instance of DiscreteTimeStochasticProcessWithProb

    See Also
    --------
    TransformedDiscreteTimeStochasticProcess : The class for transformed processes
    DiscreteTimeStochasticProcessWithProb : Base class for processes with probabilities

    Notes
    -----
    The transformation function should:
    - Accept a pandas DataFrame as input
    - Return a pandas DataFrame with the same shape
    - Be deterministic (same input always produces same output)

    Note that probability measures are not automatically adjusted by the
    transformation. The transformed process maintains the trajectory structure
    but probabilities from the original sample space are dropped.
    """
    if not isinstance(transform_func, Callable):
        raise ValueError("transform_func must be a callable function.")
    if not isinstance(process, DiscreteTimeStochasticProcessWithProb):
        raise ValueError(
            "process must be an instance of DiscreteTimeStochasticProcessWithProb."
        )

    return TransformedDiscreteTimeStochasticProcess(process, transform_func)
