"""
First-order two-state Markov chain implementation.

This module implements a discrete-time Markov chain on state space {0, 1}
with configurable transition probabilities.
"""

from .DiscreteTimeStochasticProcess import DiscreteTimeStochasticProcess
import pandas as pd
import numpy as np
import math
from itertools import product


class TwoStateMarkovChain(DiscreteTimeStochasticProcess):
    """
    Two-state first-order Markov chain.

    Models a Markov chain on {0, 1} with transition probabilities:
        P(X_n+1 = 1 | X_n = 1) = alpha
        P(X_n+1 = 1 | X_n = 0) = beta

    Parameters
    ----------
    alpha : float, default=0.8
        Probability P(1|1), must be in [0, 1]
    beta : float, default=0.3
        Probability P(1|0), must be in [0, 1]
    theta : float, default=0.5
        Initial probability P(X_1 = 1), must be in [0, 1]
    chain_length : int, default=3
        Number of time steps

    Attributes
    ----------
    alpha : float
        Transition probability P(1|1)
    beta : float
        Transition probability P(1|0)
    theta : float
        Initial probability
    omega : pandas.DataFrame or None
        Sample space with columns X1, ..., Xn (states), S1, ..., Sn
        (cumulative sums), and p (probabilities)

    Examples
    --------
    >>> mc = TwoStateMarkovChain(alpha=0.8, beta=0.3, chain_length=5)
    >>> mc.setup_sample_space()
    >>> E_S3_X1 = mc.conditional_expectation("S3", ["X1"])

    >>> mc_long = TwoStateMarkovChain(chain_length=1000)
    >>> chains = mc_long.simulate(num_chains=100)
    >>> fig, ax = plt.subplots()
    >>> mc_long.plot_simulations(num_chains=10, ax=ax, cumulative=True)
    """

    def __init__(self, alpha=0.8, beta=0.3, theta=0.5, chain_length=3):
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        super().__init__(chain_length)

    def setup_sample_space(self):
        """
        Generate complete sample space of all possible state sequences.

        Creates a DataFrame with all 2^n possible sequences of length n,
        along with cumulative sums and joint probabilities.

        Returns
        -------
        self
            Returns self for method chaining

        Warnings
        --------
        Computational complexity is O(2^n) in chain_length. For long chain lengths,
        consider using simulate() instead.
        """
        # Generate all binary sequences
        sequences = list(product([0, 1], repeat=self.chain_length))
        column_names = [f"X{i+1}" for i in range(self.chain_length)]
        self.omega = pd.DataFrame(sequences, columns=column_names)

        # Add cumulative sums S_n = X_1 + ... + X_n
        S = self.omega.apply(np.cumsum, axis=1)
        S.columns = [f"S{i+1}" for i in range(self.chain_length)]
        self.omega = pd.concat([self.omega, S], axis=1)

        # Compute joint probabilities using chain rule
        self.omega["p"] = self.omega[column_names].apply(
            lambda row: self.joint_prob(row.tolist()), axis=1
        )

        return self

    def _trans_prob(self, Y, X):
        """
        Compute one-step transition probability P(Y | X).

        Parameters
        ----------
        y : int
            Next state (0 or 1)
        x : int
            Current state (0 or 1)

        Returns
        -------
        float
            Transition probability P(Y | X)
        """
        match (Y, X):
            case (0, 0):
                return 1 - self.beta
            case (0, 1):
                return 1 - self.alpha
            case (1, 0):
                return self.beta
            case (1, 1):
                return self.alpha

    def _init_prob(self, X):
        """
        Compute initial probability P(X_1 = x).

        Parameters
        ----------
        x : int
            Initial state (0 or 1)

        Returns
        -------
        float
            Initial probability
        """
        return self.theta**X * (1 - self.theta) ** (1 - X)

    def joint_prob(self, X):
        """
        Compute joint probability P(X_1, ..., X_n) using chain rule.

        Applies the Markov property:
            P(X_1, ..., X_n) = P(X_1) * P(X_2|X_1) * ... * P(X_n|X_{n-1})

        Parameters
        ----------
        x : list or array-like
            State sequence of length n

        Returns
        -------
        float
            Joint probability of the sequence
        """
        return self._init_prob(X[0]) * math.prod(
            [self._trans_prob(X[i], X[i - 1]) for i in range(1, len(X))]
        )

    def simulate(self, num_chains=1):
        """
        Generate sample paths from the Markov chain.

        Parameters
        ----------
        num_chains : int, default=1
            Number of independent chains to simulate

        Returns
        -------
        numpy.ndarray
            Array of shape (num_chains, chain_length) containing simulated
            state sequences

        Examples
        --------
        >>> mc = TwoStateMarkovChain(chain_length=100)
        >>> chains = mc.simulate(num_chains=1000)
        >>> print(chains.shape)  # (1000, 100)
        >>> print(chains.mean())  # Should be close to stationary probability
        """
        chains = np.zeros((num_chains, self.chain_length), dtype=int)

        # Sample initial states from Bernoulli(theta)
        chains[:, 0] = np.random.binomial(1, self.theta, size=num_chains)

        # Generate subsequent states using transition probabilities
        for i in range(1, self.chain_length):
            # Transition probability depends on previous state
            probs = np.where(chains[:, i - 1] == 1, self.alpha, self.beta)
            chains[:, i] = np.random.binomial(1, probs)

        return chains

    def _get_plot_data(self, trajectories, cumulative=True, **kwargs):
        """Get plot data for Markov chain."""
        if cumulative:
            data = np.cumsum(trajectories, axis=1)
            ylabel = "cumulative sum"
            self._plot_type = "cumulative sums"  # Store for title
        else:
            data = trajectories
            ylabel = "state"
            self._plot_type = "states"
        return data, ylabel

    def _get_x_values(self, series):
        """X-axis is discrete time steps."""
        return range(self.chain_length)

    def _get_plot_title(self, **kwargs):
        """Get title for Markov chain plot."""
        plot_type = getattr(self, "_plot_type", "trajectories")
        return f"2-state markov chain {plot_type} (α={self.alpha}, β={self.beta})"
