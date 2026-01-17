"""Markov chain module."""

from collections.abc import Hashable
from itertools import product

import numpy as np
import pandas as pd

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ..base.stochastic_process import StochasticProcess


class MarkovChain(StochasticProcess):
    """A class representing a Markov chain stochastic process.

    Parameters
    ----------
    transition_matrix : pd.DataFrame
        A DataFrame representing the transition probabilities between states. The index and columns should correspond to the states of the Markov chain, and each row should sum to `1`.
    initial_distribution : ProbabilityMeasure
        A ProbabilityMeasure representing the initial distribution over the states of the Markov chain. Its sample space should match the states defined in the transition matrix.
    time : Time | None, default=None
        The time index of the stochastic process. If `None`, then the `is_discrete_time` property must be provided.
    is_discrete_time : bool | None, default=None
        Whether the stochastic process is a discrete-time process. If `None`, then `time` parameter must be provided.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.

    Raises
    ------
    TypeError
        If `transition_matrix` is not a pandas DataFrame or if `initial_distribution` is not a ProbabilityMeasure.
    ValueError
        If the index and columns of `transition_matrix` do not match the sample space of `initial_distribution`, if any row of `transition_matrix` does not sum to `1`, or if any entry in `transition_matrix` is negative.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace
    >>> from sigalg.processes import MarkovChain
    >>> state_space = SampleSpace().from_list(["rain", "sun"])
    >>> P = pd.DataFrame(
    ...     data=[
    ...         [0.9, 0.1],  # P(rain | rain) = 0.9, P(sun | rain) = 0.1
    ...         [0.4, 0.6],  # P(rain | sun) = 0.4, P(sun | sun) = 0.6
    ...     ],
    ...     index=state_space,
    ...     columns=state_space,
    ... )
    >>> pi = ProbabilityMeasure(name="pi").from_dict({"rain": 0.25, "sun": 0.75})
    >>> X = MarkovChain(
    ...     transition_matrix=P,
    ...     initial_distribution=pi,
    ...     is_discrete_time=True,
    ...     name="X",
    ... ).from_simulation(
    ...     n_trajectories=100_000,
    ...     length=3,
    ...     random_state=42,
    ... )
    >>> X # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    time      0     1     2
    trajectory
    0       sun   sun   sun
    1       sun   sun  rain
    2       sun   sun   sun
    3       sun  rain  rain
    4      rain  rain  rain
    ...     ...   ...   ...
    99995   sun  rain  rain
    99996   sun   sun  rain
    99997   sun  rain  rain
    99998  rain  rain  rain
    99999   sun  rain  rain
    <BLANKLINE>
    [100000 rows x 3 columns]
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        transition_matrix: pd.DataFrame,
        initial_distribution: ProbabilityMeasure,
        time: Time | None = None,
        is_discrete_time: bool | None = None,
        domain: SampleSpace | None = None,
        name: Hashable | None = "X",
    ) -> None:
        if not isinstance(transition_matrix, pd.DataFrame):
            raise TypeError("transition_matrix must be a pandas DataFrame.")
        if not isinstance(initial_distribution, ProbabilityMeasure):
            raise TypeError("initial_distribution must be a ProbabilityMeasure.")
        state_space = initial_distribution.sample_space
        if not transition_matrix.index.equals(
            state_space.data
        ) or not transition_matrix.columns.equals(state_space.data):
            raise ValueError(
                "transition_matrix index and columns must match the sample space of initial_distribution."
            )
        if not np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1.")
        if np.any(transition_matrix.values < 0):
            raise ValueError("All entries in transition_matrix must be non-negative.")

        super().__init__(
            domain=domain,
            time=time,
            is_discrete_time=is_discrete_time,
            is_discrete_state=True,
            name=name,
        )

        self.support = list(state_space)
        self.states = self.support
        self.n_states = len(self.states)
        self.transition_matrix = transition_matrix
        self.initial_distribution = initial_distribution

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs):
        """Generate the enumerated trajectories for the Markov chain.

        Parameters
        ----------
        **kwargs
            Not needed for Markov chain enumeration, but included for consistency with the base class.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the enumerated trajectories as rows and time points as columns.
        """
        trajectories = list(product(self.states, repeat=len(self.time)))
        return pd.DataFrame(data=trajectories, columns=self.time.data)

    def _simulation_logic(
        self,
        n_trajectories: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Generate simulated data for the Markov chain.

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories to simulate.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        rng = np.random.default_rng(random_state)

        P = self.transition_matrix.values
        n_states = self.n_states
        length = len(self.time)
        initial_distribution = self.initial_distribution

        initial_state_indices = rng.choice(
            n_states, size=n_trajectories, p=initial_distribution.data.values
        )

        trajectory_indices = np.empty((n_trajectories, length), dtype=int)
        trajectory_indices[:, 0] = initial_state_indices

        for t in range(length - 1):
            current_states = trajectory_indices[:, t]
            transition_probs = P[current_states]
            random_vals = rng.random(n_trajectories)
            cumprobs = np.cumsum(transition_probs, axis=1)
            trajectory_indices[:, t + 1] = (cumprobs < random_vals[:, None]).sum(axis=1)

        raw_trajectories = np.array(self.states)[trajectory_indices]
        return pd.DataFrame(data=raw_trajectories)

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        """Generate the exact probability measure for the Markov chain.

        Parameters
        ----------
        name : Hashable | None, default="P"
            The name of the generated probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The generated probability measure.
        """
        data_array = self.data.values
        state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        trajectories_indices = np.vectorize(state_to_idx.get)(data_array)

        initial_probs = self.initial_distribution.data.loc[data_array[:, 0]].values
        transition_probs = self.transition_matrix.values[
            trajectories_indices[:, :-1], trajectories_indices[:, 1:]
        ]
        prob_values = initial_probs * np.prod(transition_probs, axis=1)

        return ProbabilityMeasure(sample_space=self.domain, name=name).from_pandas(
            pd.Series(prob_values, index=self.domain.data)
        )

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated Markov chain" if self.is_enumerated else "Markov chain"
        return f"{prefix} process '{self.name}'"
