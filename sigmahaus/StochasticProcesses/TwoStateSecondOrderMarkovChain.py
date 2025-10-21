"""
Second-order two-state Markov chain implementation.

This module implements a discrete-time Markov chain where transition
probabilities depend on the two most recent states.
"""

from .DiscreteTimeStochasticProcess import DiscreteTimeStochasticProcess
import pandas as pd
import numpy as np
import math
from itertools import product


class TwoStateSecondOrderMarkovChain(DiscreteTimeStochasticProcess):
    """
    Two-state second-order Markov chain.

    Models a chain on {0, 1} where transitions depend on two previous states:
        P(X_1 = 1) = theta_1
        P(X_2 = 1 | X_1) = theta_2[X_1]
        P(X_n = 1 | X_n-1, X_n-2) = theta_n[X_n-1, X_n-2] for n >= 3

    Parameters
    ----------
    theta_1 : float
        Initial probability P(X_1 = 1), must be in [0, 1]
    theta_2 : array-like of length 2
        Second-step probabilities: theta_2[i] = P(X_2 = 1 | X_1 = i)
    theta_n : 2x2 array-like
        Subsequent probabilities: theta_n[i, j] = P(X_n = 1 | X_n-1 = i, X_n-2 = j)
    chain_length : int, default=3
        Number of time steps, must be >= 3

    Examples
    --------
    >>> theta_2 = [0.3, 0.7]
    >>> theta_n = [[0.2, 0.4], [0.6, 0.9]]
    >>> mc = TwoStateSecondOrderMarkovChain(
    ...     theta_1=0.5, theta_2=theta_2, theta_n=theta_n, chain_length=100
    ... )
    >>> fig, ax = plt.subplots()
    >>> mc.plot_simulations(num_chains=10, ax=ax, cumulative=True)
    """

    def __init__(self, theta_1, theta_2, theta_n, chain_length=3):
        if chain_length < 3:
            raise ValueError("chain_length must be at least 3 for second-order chain")

        self.theta_1 = theta_1
        self.theta_2 = np.array(theta_2)  # [P(1|0), P(1|1)]
        self.theta_n = np.array(theta_n)  # 2x2 matrix
        super().__init__(chain_length)

    def setup_sample_space(self):
        """
        Generate complete sample space of all possible state sequences.

        Returns
        -------
        self
            Returns self for method chaining
        """
        # Generate all binary sequences
        sequences = list(product([0, 1], repeat=self.trajectory_length))
        column_names = [f"X{i+1}" for i in range(self.trajectory_length)]
        self.omega = pd.DataFrame(sequences, columns=column_names)

        # Add cumulative sums S_n = X_1 + ... + X_n
        S = self.omega.apply(np.cumsum, axis=1)
        S.columns = [f"S{i+1}" for i in range(self.trajectory_length)]
        self.omega = pd.concat([self.omega, S], axis=1)

        # Compute joint probabilities using chain rule
        self.omega["p"] = self.omega[column_names].apply(
            lambda row: self.joint_prob(row.tolist()), axis=1
        )

        return self

    def _prob_1(self, X1):
        """P(X_1 = X1)"""
        return self.theta_1**X1 * (1 - self.theta_1) ** (1 - X1)

    def _prob_2(self, X2, X1):
        """P(X_2 = X2 | X_1 = X1)"""
        prob_1 = self.theta_2[X1]  # P(X_2 = 1 | X_1 = X1)
        return prob_1**X2 * (1 - prob_1) ** (1 - X2)

    def _prob_n(self, Xn, Xn1, Xn2):
        """P(X_n = Xn | X_{n-1} = Xn1, X_{n-2} = Xn2)"""
        prob_1 = self.theta_n[Xn1, Xn2]  # P(X_n = 1 | X_{n-1}, X_{n-2})
        return prob_1**Xn * (1 - prob_1) ** (1 - Xn)

    def joint_prob(self, X):
        """
        Compute joint probability P(X_1, ..., X_n) using chain rule.

        P(X_1, ..., X_n) = P(X_1) * P(X_2|X_1) * P(X_3|X_2,X_1) * ... * P(X_n|X_{n-1},X_{n-2})

        Parameters
        ----------
        X : list or array-like
            State sequence of length n

        Returns
        -------
        float
            Joint probability of the sequence
        """
        if len(X) < 3:
            raise ValueError("Sequence must have at least 3 elements")

        return (
            self._prob_1(X[0])
            * self._prob_2(X[1], X[0])
            * math.prod(
                [self._prob_n(X[i], X[i - 1], X[i - 2]) for i in range(2, len(X))]
            )
        )

    def simulate(self, num_chains=1):
        """
        Generate sample paths from the second-order Markov chain.

        Parameters
        ----------
        num_chains : int, default=1
            Number of independent chains to simulate

        Returns
        -------
        numpy.ndarray
            Array of shape (num_chains, chain_length) containing simulated sequences
        """
        chains = np.zeros((num_chains, self.trajectory_length), dtype=int)

        # First step: X_1 ~ Bernoulli(theta_1)
        chains[:, 0] = np.random.binomial(1, self.theta_1, size=num_chains)

        # Second step: X_2 | X_1
        probs = self.theta_2[chains[:, 0]]
        chains[:, 1] = np.random.binomial(1, probs)

        # Subsequent steps: X_n | X_{n-1}, X_{n-2}
        for i in range(2, self.trajectory_length):
            Xn1 = chains[:, i - 1]
            Xn2 = chains[:, i - 2]
            # Get transition probabilities based on previous two states
            probs = self.theta_n[Xn1, Xn2]
            chains[:, i] = np.random.binomial(1, probs)

        return chains

    def _get_plot_data(self, trajectories, cumulative=True, **kwargs):
        """Get plot data for second-order Markov chain."""
        if cumulative:
            data = np.cumsum(trajectories, axis=1)
            ylabel = "cumulative sum"
            self._plot_type = "cumulative sums"
        else:
            data = trajectories
            ylabel = "state"
            self._plot_type = "states"
        return data, ylabel

    def _get_x_values(self, series):
        """X-axis is discrete time steps."""
        return range(self.trajectory_length)

    def _get_plot_title(self, **kwargs):
        """Get title for second-order Markov chain plot."""
        plot_type = getattr(self, "_plot_type", "trajectories")
        return f"2-state second-order markov chain {plot_type}"
