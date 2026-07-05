"""A class representing a Markov chain stochastic process."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ...core.base.index import Index
    from ...core.probability_measures.probability_measure import ProbabilityMeasure


class MarkovChain(StochasticProcess):
    """A class representing a Markov chain stochastic process.

    The constructor is not intended for direct usage. Instead, user's should call one of either class methods `from_enumeration` or `from_simulation`. See the Examples section below.

    See also the Notes section below for the mathematical details.

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
    >>> X = MarkovChain.from_enumeration(
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

    >>> Y = MarkovChain.from_simulation(
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

    >>> print(Y.range.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P_Y':
                    probability
    Y_1  Y_2  Y_3
    rain rain rain      0.20124
              sun       0.02289
         sun  rain      0.00978
              sun       0.01557
    sun  rain rain      0.27134
              sun       0.02874
         sun  rain      0.17874
              sun       0.27170
    """

    _repr_name = "Markov chain"

    # --------------------- enumeration methods --------------------- #

    @classmethod
    def from_enumeration(
        cls,
        transition_matrix: pd.DataFrame,
        initial_distribution: ProbabilityMeasure,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Generate all trajectories of the Markov chain by exhaustive enumeration.

        Parameters
        ----------
        transition_matrix : pd.DataFrame
            A DataFrame representing the transition probabilities between states. The index and columns should correspond to the states of the Markov chain, and each row should sum to `1`.
        initial_distribution : ProbabilityMeasure
            A ProbabilityMeasure representing the initial distribution over the states of the Markov chain. Its sample space should match the states defined in the transition matrix.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `transition_matrix` is not a `pd.DataFrame`, or if `initial_distribution` is not a `ProbabilityMeasure`.
        ValueError
            If the index and columns of `transition_matrix` do not match the sample space of `initial_distribution`, or if the rows of `transition_matrix` are not valid probability distributions.

        Returns
        -------
        self : StochasticProcess
            The current instance with enumerated trajectories.

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
        >>> X = MarkovChain.from_enumeration(
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
        """
        from ...core.probability_measures.probability_measure import ProbabilityMeasure

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

        index = cls._validate_and_return_index(index=index, length=length)
        process = cls(index=index, name=name)

        process.support = list(state_space)
        process.states = process.support
        process.n_states = len(process.states)
        process.transition_matrix = transition_matrix
        process.initial_distribution = initial_distribution

        return process._enumeration_logic()

    def _enumeration_hook(self):
        """Hook for enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """  # noqa: D401
        trajectories = list(product(self.states, repeat=len(self.time)))
        return pd.DataFrame(data=trajectories, columns=self.time.data)

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
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
        >>> X = MarkovChain.from_enumeration(
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
        from ...core.probability_measures.probability_measure import ProbabilityMeasure
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

        data_array = self.data.values
        state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        trajectories_indices = np.vectorize(state_to_idx.get)(data_array)

        initial_probs = self.initial_distribution.data.loc[data_array[:, 0]].values
        transition_probs = self.transition_matrix.values[
            trajectories_indices[:, :-1], trajectories_indices[:, 1:]
        ]
        prob_values = initial_probs * np.prod(transition_probs, axis=1)

        sig_alg = SigmaAlgebra.power_set(self.sample_space)

        return ProbabilityMeasure(
            sig_alg=sig_alg,
            mapping=pd.Series(prob_values, index=self.sample_space.data),
        )

    # --------------------- simulation methods --------------------- #

    @classmethod
    def from_simulation(
        cls,
        transition_matrix: pd.DataFrame,
        initial_distribution: ProbabilityMeasure,
        n_trajectories: int,
        index: Index | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Simulate trajectories of the Markov chain.

        Parameters
        ----------
        transition_matrix : pd.DataFrame
            A DataFrame representing the transition probabilities between states. The index and columns should correspond to the states of the Markov chain, and each row should sum to `1`.
        initial_distribution : ProbabilityMeasure
            A ProbabilityMeasure representing the initial distribution over the states of the Markov chain. Its sample space should match the states defined in the transition matrix.
        n_trajectories : int
            The number of trajectories to simulate.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `transition_matrix` is not a `pd.DataFrame`, or if `initial_distribution` is not a `ProbabilityMeasure`.
        ValueError
            If the index and columns of `transition_matrix` do not match the sample space of `initial_distribution`, or if the rows of `transition_matrix` are not valid probability distributions.

        Returns
        -------
        self : StochasticProcess
            The current instance with simulated trajectories.

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

        Simulate 100,000 length-2 trajectories of the Markov chain.

        >>> T = Time.discrete(start=1, stop=3)
        >>> X = MarkovChain.from_simulation(
        ...     transition_matrix=P,
        ...     initial_distribution=pi,
        ...     n_trajectories=100_000,
        ...     random_state=42,
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Markov chain 'X':
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
        """
        from ...core.probability_measures.probability_measure import ProbabilityMeasure

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

        index = cls._validate_and_return_index(index=index, length=length)
        random_state = cls._validate_simulation_parameters_and_return_rng(
            n_trajectories=n_trajectories, random_state=random_state
        )
        process = cls(index=index, name=name)

        process.n_trajectories = n_trajectories
        process.random_state = random_state
        process.support = list(state_space)
        process.states = process.support
        process.n_states = len(process.states)
        process.transition_matrix = transition_matrix
        process.initial_distribution = initial_distribution

        return process._simulation_logic()

    def _simulation_hook(self) -> pd.DataFrame:
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
