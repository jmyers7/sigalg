"""
Three-win streak betting strategy implementation.

This module implements a path-dependent betting strategy where bets are
placed only after observing three consecutive wins.
"""

from .DiscreteTimeStochasticProcess import DiscreteTimeStochasticProcess
import pandas as pd
import numpy as np
from itertools import product


class ThreeWinStreakSelectionStrategy(DiscreteTimeStochasticProcess):
    """
    Betting strategy with three-win streak condition.

    Models a strategy where a bettor observes win/loss trials but only places
    bets after three consecutive wins. Capital evolves based on selective bets.

    Parameters
    ----------
    theta : float
        Win probability for each trial, must be in [0, 1]
    trajectory_length : int
        Number of trials to observe
    a : float, default=0
        Initial capital, must be non-negative

    Attributes
    ----------
    theta : float
        Win probability
    a : float
        Initial capital
    omega : pandas.DataFrame or None
        Sample space with columns X1, ..., Xn (trial outcomes +/- 1),
        B1, ..., Bn (bet outcomes), S0, ..., Sn (capital), and p (probabilities)

    Raises
    ------
    ValueError
        If theta not in [0, 1] or a < 0

    Notes
    -----
    Capital remains unchanged (S_n = S_n-1) when no bet is placed. The process
    is path-dependent: identical win counts can yield different final capital
    depending on when three-win streaks occur.

    Examples
    --------
    >>> bet = ThreeWinStreakSelectionStrategy(theta=0.6, trajectory_length=4, a=10)
    >>> bet.setup_sample_space()
    >>> print(bet.omega)
        X1  X2  X3  X4  B1  B2  B3  B4  S0  S1  S2  S3  S4       p
    0   -1  -1  -1  -1   0   0   0   0  10  10  10  10  10  0.0256
    1   -1  -1  -1   1   0   0   0   0  10  10  10  10  10  0.0384
    2   -1  -1   1  -1   0   0   0   0  10  10  10  10  10  0.0384
    ...
    14   1   1   1  -1   0   0   0  -1  10  10  10  10   9  0.0864
    15   1   1   1   1   0   0   0   1  10  10  10  10  11  0.1296
    >>> cond_exp = bet.conditional_expectation("S4", ["X1", "X2", "X3"])
    >>> print(cond_exp)
    X1  X2  X3
    -1  -1  -1   10.0
            1    10.0
        1  -1    10.0
            1    10.0
    1  -1  -1    10.0
            1    10.0
        1  -1    10.0
            1    10.2
    """

    def __init__(self, theta, trajectory_length, a=0):
        # Validate parameters
        if not 0 <= theta <= 1:
            raise ValueError("theta must be a probability between 0 and 1")
        if a < 0:
            raise ValueError("a must be nonnegative")

        super().__init__(trajectory_length)
        self.theta = theta
        self.a = a

    def setup_sample_space(self):
        """
        Generate complete sample space of all win/loss sequences.

        Creates a DataFrame with all 2^n possible trial sequences, bet outcomes,
        capital trajectories, and probabilities.

        Returns
        -------
        self
            Returns self for method chaining

        Warnings
        --------
        Complexity is O(2^n) in chain_length. Infeasible for long chain lengths.
        Use simulate() instead for longer chains.

        Notes
        -----
        The resulting DataFrame contains:
        - X1, ..., Xn: Trial outcomes (±1)
        - B1, ..., Bn: Bet outcomes (0 if no bet, ±1 if bet placed)
        - S0, ..., Sn: Capital at each time step (S0 is initial capital)
        - p: Joint probability of each sequence
        """
        omega_cardinality = 2**self.trajectory_length
        if omega_cardinality > 1000:  # Reasonable threshold
            raise ValueError(
                "Sample space size exceeds threshold 1000. Use simulate() instead."
            )
        # Generate all possible win (+1) / loss (-1) sequences
        sequences = list(product([-1, 1], repeat=self.trajectory_length))
        X_names = [f"X{i + 1}" for i in range(self.trajectory_length)]
        self.omega = pd.DataFrame(sequences, columns=X_names)

        # Generate bet outcomes (0 when no bet placed, ±1 when bet placed)
        bet_outcomes = self.omega.apply(self._generate_bet_outcomes, axis=1)
        bet_df = pd.DataFrame(bet_outcomes.tolist())
        bet_df.columns = [f"B{i + 1}" for i in range(self.trajectory_length)]

        # Generate capital trajectories (includes S0 = initial capital)
        capitals = bet_df.apply(self._compute_capital, axis=1)
        capitals_df = pd.DataFrame(capitals.tolist())
        capitals_df.columns = [f"S{i}" for i in range(self.trajectory_length + 1)]

        # Concatenate all columns
        self.omega = pd.concat([self.omega, bet_df, capitals_df], axis=1)

        # Compute probabilities based on number of wins
        num_wins = (self.omega[X_names] == 1).sum(axis=1)
        self.omega["p"] = self.theta**num_wins * (1 - self.theta) ** (
            self.trajectory_length - num_wins
        )

        return self

    def joint_prob(self, X):
        """
        Compute probability of a win/loss sequence.

        Assumes independent trials with success probability theta.

        Parameters
        ----------
        X : array-like
            Sequence of wins (1) and losses (-1)

        Returns
        -------
        float
            Probability of observing this sequence
        """
        num_wins = sum(1 for win in X if win == 1)
        num_losses = len(X) - num_wins
        return self.theta**num_wins * (1 - self.theta) ** num_losses

    def _generate_wins_losses(self):
        """
        Generate one random sequence of wins and losses.

        Returns
        -------
        numpy.ndarray
            Array of length chain_length with values in {-1, 1} where
            1 represents a win and -1 represents a loss
        """
        X = np.random.binomial(1, self.theta, size=self.trajectory_length)
        X = np.where(X == 1, 1, -1)
        return X

    def _generate_bet_outcomes(self, X):
        """
        Generate bet outcomes from trial sequence.

        A bet is placed at time t if trials X[t-3], X[t-2], X[t-1] are all
        wins. The bet outcome B[t] equals X[t] (the next trial's outcome).
        When no bet is placed, B[t] = 0.

        Parameters
        ----------
        X : array-like
            Complete sequence of trial outcomes (±1)

        Returns
        -------
        numpy.ndarray
            Bet outcomes at each time step. Length equals len(X).
            Values are 0 (no bet), 1 (bet placed and won), or -1 (bet placed and lost).

        Notes
        -----
        First three positions are always 0 since at least three previous trials
        are needed to trigger a bet.
        """
        X = np.array(X)
        B = [0, 0, 0]  # No bets can be placed in first three trials

        for t in range(3, len(X)):
            # Check if previous three trials were all wins
            if np.all(X[t - 3 : t] == 1):
                B.append(X[t])  # Place bet with outcome = current trial
            else:
                B.append(0)  # No bet placed

        return np.array(B)

    def _compute_capital(self, B):
        """
        Compute capital trajectory from bet outcomes.

        Capital starts at initial value a and changes by +1 or -1 when bets
        are placed. When B[t] = 0 (no bet), capital remains unchanged.

        Parameters
        ----------
        B : array-like
            Bet outcomes at each time step (0, +1, or -1)

        Returns
        -------
        numpy.ndarray
            Capital values S0, S1, ..., Sn where S0 = a (initial capital)
            and S_t = S_{t-1} + B_t for t >= 1.
            Length is len(B) + 1.
        """
        # S_t = a + sum of all bet outcomes up to time t
        S = np.concatenate([[self.a], self.a + np.cumsum(B)])
        return S

    def simulate(self, num_chains=1):
        """
        Generate sample capital trajectories.

        Parameters
        ----------
        num_chains : int, default=1
            Number of independent trajectories to simulate

        Returns
        -------
        list of numpy.ndarray
            Each array contains capital values S0, S1, ..., Sn for one trajectory.
            All arrays have length chain_length + 1.
        """
        trajectories = []

        for _ in range(num_chains):
            X = self._generate_wins_losses()  # Trial outcomes (±1)
            B = self._generate_bet_outcomes(X)  # Bet outcomes (0 or ±1)
            S = self._compute_capital(B)  # Capital trajectory
            trajectories.append(S)

        return trajectories

    def _get_plot_data(self, trajectories, **kwargs):
        """Get plot data for betting strategy (capital over time)."""
        # trajectories is a list of arrays, not a 2D numpy array
        return trajectories, "capital"

    def _get_x_values(self, series):
        """X-axis includes initial capital (time 0)."""
        return range(self.trajectory_length + 1)

    def _get_plot_title(self, **kwargs):
        """Get title for betting strategy plot."""
        return f"3-win streak selection strategy (θ={self.theta}, initial capital={self.a})"
