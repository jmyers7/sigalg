"""A class representing a Markov chain stochastic process."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess, generator

if TYPE_CHECKING:
    from ...core.indices.index import Time
    from ...core.measures.probability_measure import ProbabilityMeasure
    from ...core.spaces.domain import Domain
    from ...validation.index_validator import IndexLike


# TODO: add a `ProbabilitiesLike` so that the user can pass in e.g. numpy arrays
class MarkovChain(StochasticProcess):
    """A class representing a Markov chain stochastic process.

    The constructor is not intended for direct usage. Instead, user's should call the `generate` class method. See the Examples section below.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace, Time
    >>> from sigalg.processes import MarkovChain

    Define the transition matrix of a 2-state Markov chain modeling the probability of a day's wheather, given the probability of the previous day's.

    >>> Omega = SampleSpace(["rain", "sun"])
    >>> P = pd.DataFrame(
    ...     data=[
    ...         [0.9, 0.1],  # P(rain | rain) = 0.9, P(sun | rain) = 0.1
    ...         [0.4, 0.6],  # P(rain | sun) = 0.4, P(sun | sun) = 0.6
    ...     ],
    ...     index=Omega,
    ...     columns=Omega,
    ... )

    Define the initial distribution of the first day's weather.

    >>> pi = ProbabilityMeasure(
    ...     sample_space=Omega,
    ...     mapping={"rain": 0.25, "sun": 0.75},
    ...     name="pi",
    ... )

    Enumerate all possible length-3 trajectories of the Markov chain.

    >>> T = Time.discrete(start=1, stop=3)
    >>> X = MarkovChain.generate(
    ...     mode="enum",
    ...     transition_matrix=P,
    ...     initial_distribution=pi,
    ...     index=T,
    ... )
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Markov chain 'X':
    time       1     2     3
    sample
    0       rain  rain  rain
    1       rain  rain   sun
    2       rain   sun  rain
    3       rain   sun   sun
    4        sun  rain  rain
    5        sun  rain   sun
    6        sun   sun  rain
    7        sun   sun   sun

    Print the probability of each trajectory.

    >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
            probability
    sample
    0            0.2025
    1            0.0225
    2            0.0100
    3            0.0150
    4            0.2700
    5            0.0300
    6            0.1800
    7            0.2700

    Instead of exhaustively enumerating all trajectories, we simulate 100,000 of them.

    >>> Y = MarkovChain.generate(
    ...     mode="sim",
    ...     transition_matrix=P,
    ...     initial_distribution=pi,
    ...     n_trajectories=100_000,
    ...     random_state=42,
    ...     index=T,
    ...     name="Y",
    ... )
    >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
    Markov chain 'Y':
    time       1     2     3
    sample
    0        sun   sun   sun
    1        sun   sun  rain
    2        sun   sun   sun
    3        sun  rain  rain
    4       rain  rain  rain
    ...      ...   ...   ...
    99995    sun  rain  rain
    99996    sun   sun  rain
    99997    sun  rain  rain
    99998   rain  rain  rain
    99999    sun  rain  rain
    <BLANKLINE>
    [100000 rows x 3 columns]

    We then print the empirical probability distribution. Note how similar its values are to the exact probability distribution printed above.

    >>> print(Y.pushforward())  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U_Y':
                    probability
    Y_1  Y_2  Y_3
    sun  sun  sun       0.27170
              rain      0.17874
         rain rain      0.27134
    rain rain rain      0.20124
         sun  sun       0.01557
              rain      0.00978
    sun  rain sun       0.02874
    rain rain sun       0.02289
    """

    _repr_name = "MarkovChain"
    _str_name = "Markov chain"

    # --------------------- constructors --------------------- #

    @generator
    def generate(
        cls,
        transition_matrix: pd.DataFrame,
        initial_distribution: ProbabilityMeasure,
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        index: Time | IndexLike | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> MarkovChain:
        """Generate trajectories of the Markov chain by either exhaustive enumeration or Monte Carlo simulation.

        Parameters
        ----------
        transition_matrix : pd.DataFrame
            A `pd.DataFrame` representing the transition probabilities between states. The index and columns should correspond to the states of the Markov chain, and each row should sum to `1`.
        initial_distribution : ProbabilityMeasure
            A ProbabilityMeasure representing the initial distribution over the states of the Markov chain. Its sample space should match the states defined in the transition matrix.
        mode : Literal["enum", "sim"], default="sim"
            Whether to generate trajectories by exhaustive enumeration or Monte Carlo simulation.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. If the generation mode is set to `enum`, this parameter is ignored.
        index : Time | IndexLike | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        random_state : int | np.random.Generator | None, default=None
            An optional random state for reproducibility.
        name : Hashable, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `transition_matrix` is not a `pd.DataFrame`, or if `initial_distribution` is not a `ProbabilityMeasure`.
        ValueError
            If the index and columns of `transition_matrix` do not match the sample space of `initial_distribution`, or if the rows of `transition_matrix` are not valid probability distributions.

        Returns
        -------
        self : MarkovChain
            The current instance with generated trajectories.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, Time
        >>> from sigalg.processes import MarkovChain

        Define the transition matrix of a 2-state Markov chain modeling the probability of a day's wheather, given the probability of the previous day's.

        >>> Omega = SampleSpace(["rain", "sun"])
        >>> P = pd.DataFrame(
        ...     data=[
        ...         [0.9, 0.1],  # P(rain | rain) = 0.9, P(sun | rain) = 0.1
        ...         [0.4, 0.6],  # P(rain | sun) = 0.4, P(sun | sun) = 0.6
        ...     ],
        ...     index=Omega,
        ...     columns=Omega,
        ... )

        Define the initial distribution of the first day's weather.

        >>> pi = ProbabilityMeasure(
        ...     sample_space=Omega,
        ...     mapping={"rain": 0.25, "sun": 0.75},
        ...     name="pi",
        ... )

        Enumerate all possible length-3 trajectories of the Markov chain.

        >>> T = Time.discrete(start=1, stop=3)
        >>> X = MarkovChain.generate(
        ...     mode="enum",
        ...     transition_matrix=P,
        ...     initial_distribution=pi,
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Markov chain 'X':
        time       1     2     3
        sample
        0       rain  rain  rain
        1       rain  rain   sun
        2       rain   sun  rain
        3       rain   sun   sun
        4        sun  rain  rain
        5        sun  rain   sun
        6        sun   sun  rain
        7        sun   sun   sun

        Print the probability of each trajectory.

        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0            0.2025
        1            0.0225
        2            0.0100
        3            0.0150
        4            0.2700
        5            0.0300
        6            0.1800
        7            0.2700

        Instead of exhaustively enumerating all trajectories, we simulate 100,000 of them.

        >>> Y = MarkovChain.generate(
        ...     mode="sim",
        ...     transition_matrix=P,
        ...     initial_distribution=pi,
        ...     n_trajectories=100_000,
        ...     random_state=42,
        ...     index=T,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Markov chain 'Y':
        time       1     2     3
        sample
        0        sun   sun   sun
        1        sun   sun  rain
        2        sun   sun   sun
        3        sun  rain  rain
        4       rain  rain  rain
        ...      ...   ...   ...
        99995    sun  rain  rain
        99996    sun   sun  rain
        99997    sun  rain  rain
        99998   rain  rain  rain
        99999    sun  rain  rain
        <BLANKLINE>
        [100000 rows x 3 columns]

        We then print the empirical probability distribution. Note how similar its values are to the exact probability distribution printed above.

        >>> print(Y.pushforward())  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'U_Y':
                        probability
        Y_1  Y_2  Y_3
        sun  sun  sun       0.27170
                  rain      0.17874
             rain rain      0.27134
        rain rain rain      0.20124
             sun  sun       0.01557
                  rain      0.00978
        sun  rain sun       0.02874
        rain rain sun       0.02289
        """
        from ...core.measures.probability_measure import ProbabilityMeasure

        if not isinstance(transition_matrix, pd.DataFrame):
            raise TypeError("transition_matrix must be a pandas DataFrame.")
        if not isinstance(initial_distribution, ProbabilityMeasure):
            raise TypeError("initial_distribution must be a ProbabilityMeasure.")
        state_space = initial_distribution.domain
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

        return {
            "transition_matrix": transition_matrix,
            "initial_distribution": initial_distribution,
        }

    # --------------------- properties --------------------- #

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """The transition matrix of the Markov chain.

        Returns
        -------
        transition_matrix : pd.DataFrame
            A `pd.DataFrame` representing the transition probabilities between states.
        """
        return self._transition_matrix

    @property
    def initial_distribution(self) -> ProbabilityMeasure:
        """The initial distribution of the Markov chain.

        Returns
        -------
        initial_distribution : ProbabilityMeasure
            A ProbabilityMeasure representing the initial distribution over the states of the Markov chain.
        """
        return self._initial_distribution

    @property
    def state_space(self) -> Domain:
        """The state space of the Markov chain.

        This is a derived property, not explicitly set at generation by the user.

        Returns
        -------
        state_space : Domain
            A `Domain` representing the states of the Markov chain.
        """
        return self.initial_distribution.domain

    @property
    def states(self) -> list[Hashable]:
        """The support of the Markov chain.

        This is a derived property, not explicitly set at generation by the user.

        Returns
        -------
        states : list[Hashable]
            A list of the states in the sample space of the Markov chain.
        """
        return list(self.state_space)

    @property
    def n_states(self) -> int:
        """The number of states in the Markov chain.

        This is a derived property, not explicitly set at generation by the user.

        Returns
        -------
        n_states : int
            The number of states in the sample space of the Markov chain.
        """
        return len(self.states)

    # --------------------- enumeration methods --------------------- #

    def _enumeration_subclass_hook(self):
        """Hook for enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """  # noqa: D401
        trajectories = list(product(self.states, repeat=len(self.time)))
        return pd.DataFrame(data=trajectories, columns=self.time.data)

    def _generate_exact_prob_measure(self, domain: Domain) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated Markov chain.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, Time
        >>> from sigalg.processes import MarkovChain

        Define the transition matrix of a 2-state Markov chain modeling the probability of a day's wheather, given the probability of the previous day's.

        >>> Omega = SampleSpace(["rain", "sun"])
        >>> P = pd.DataFrame(
        ...     data=[
        ...         [0.9, 0.1],  # P(rain | rain) = 0.9, P(sun | rain) = 0.1
        ...         [0.4, 0.6],  # P(rain | sun) = 0.4, P(sun | sun) = 0.6
        ...     ],
        ...     index=Omega,
        ...     columns=Omega,
        ... )

        Define the initial distribution of the first day's weather.

        >>> pi = ProbabilityMeasure(
        ...     sample_space=Omega,
        ...     mapping={"rain": 0.25, "sun": 0.75},
        ...     name="pi",
        ... )

        Enumerate all possible length-3 trajectories of the Markov chain.

        >>> T = Time.discrete(start=1, stop=3)
        >>> X = MarkovChain.generate(
        ...     mode="enum",
        ...     transition_matrix=P,
        ...     initial_distribution=pi,
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Markov chain 'X':
        time       1     2     3
        sample
        0       rain  rain  rain
        1       rain  rain   sun
        2       rain   sun  rain
        3       rain   sun   sun
        4        sun  rain  rain
        5        sun  rain   sun
        6        sun   sun  rain
        7        sun   sun   sun

        Print the probability of each trajectory.

        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0            0.2025
        1            0.0225
        2            0.0100
        3            0.0150
        4            0.2700
        5            0.0300
        6            0.1800
        7            0.2700
        """
        from ...core.measures.probability_measure import ProbabilityMeasure

        data_array = self.data.values
        state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        trajectories_indices = np.vectorize(state_to_idx.get)(data_array)

        initial_probs = self.initial_distribution.data.loc[data_array[:, 0]].values
        transition_probs = self.transition_matrix.values[
            trajectories_indices[:, :-1], trajectories_indices[:, 1:]
        ]
        prob_values = initial_probs * np.prod(transition_probs, axis=1)

        return ProbabilityMeasure(
            domain=domain,
            mapping=pd.Series(prob_values, index=domain.data),
        )

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        """Generate simulated data for the Markov chain.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        P = self.transition_matrix.values
        n_states = self.n_states
        length = len(self.time)
        initial_distribution = self.initial_distribution

        initial_state_indices = self.random_state.choice(
            n_states, size=self.n_trajectories, p=initial_distribution.data.values
        )

        trajectory_indices = np.empty((self.n_trajectories, length), dtype=int)
        trajectory_indices[:, 0] = initial_state_indices

        for t in range(length - 1):
            current_states = trajectory_indices[:, t]
            transition_probs = P[current_states]
            random_vals = self.random_state.random(self.n_trajectories)
            cumprobs = np.cumsum(transition_probs, axis=1)
            trajectory_indices[:, t + 1] = (cumprobs < random_vals[:, None]).sum(axis=1)

        raw_trajectories = np.array(self.states)[trajectory_indices]
        return pd.DataFrame(data=raw_trajectories, columns=self.time.data)
