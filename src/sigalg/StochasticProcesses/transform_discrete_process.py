"""
Factory function for creating transformed discrete-time stochastic processes.

This module provides a convenience function for applying transformations to
existing discrete-time stochastic processes, creating derived processes such as
cumulative sums, differences, or other functional transformations.
"""

from .discrete_time_stochastic_process_with_prob import DiscreteTimeStochasticProcessWithProb
from .transformed_discrete_time_stochastic_process import (
    TransformedDiscreteTimeStochasticProcess,
)
from collections.abc import Callable


def transform_discrete_process(src_process, transform_func):
    """
    Create a transformed discrete stochastic process.

    Applies a transformation function to trajectories from an existing discrete-time
    stochastic process with probability measures. The transformation is applied to
    both simulated trajectories and the sample space (if generated).

    Parameters
    ----------
    src_process : DiscreteTimeStochasticProcessWithProb
        The source discrete stochastic process to be transformed. Must have
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
    - Be deterministic (same input always produces same output)

    Note that probability measures are not automatically adjusted by the
    transformation. The transformed process maintains the trajectory structure
    but probabilities from the source sample space are dropped.

    Examples
    --------
    >>> # P(X_1 = 0) = 0.5, P(X_1 = 1) = 0.5
    >>> init_prob = np.array([0.5, 0.5])
    >>> # P(X_{n+1} = 0 | X_n = 0) = 0.7, P(X_{n+1} = 1 | X_n = 0) = 0.3
    >>> # P(X_{n+1} = 0 | X_n = 1) = 0.2, P(X_{n+1} = 1 | X_n = 1) = 0.8
    >>> transition_matrix = np.array([[0.7, 0.3], [0.2, 0.8]])
    >>> # Instantiate Markov chain
    >>> mc = FirstOrderMarkovChain(
    ...     transition_matrix,
    ...     init_prob,
    ... )
    >>> mc.simulate(trajectory_length=4)
    >>> print(mc.trajectories)
       X1  X2  X3  X4
    0   0   0   0   0
    1   1   1   0   0
    2   1   1   1   0
    3   1   1   1   1
    4   0   0   0   1
    5   0   0   1   1
    6   0   0   0   0
    7   1   1   1   0
    8   1   1   1   1
    9   1   1   0   0
    >>> transformed_process = transform_discrete_process(mc, lambda omega: omega.cumsum(axis=1))
    >>> transformed_process.simulate()
    >>> print(transformed_process.trajectories)
       X1  X2  X3  X4
    0   0   0   0   0
    1   1   2   2   2
    2   1   2   3   3
    3   1   2   3   4
    4   0   0   0   1
    5   0   0   1   2
    6   0   0   0   0
    7   1   2   3   3
    8   1   2   3   4
    9   1   2   2   2
    """
    if not isinstance(transform_func, Callable):
        raise ValueError("transform_func must be a callable function.")
    if not isinstance(src_process, DiscreteTimeStochasticProcessWithProb):
        raise ValueError(
            "process must be an instance of DiscreteTimeStochasticProcessWithProb."
        )

    return TransformedDiscreteTimeStochasticProcess(src_process, transform_func)
