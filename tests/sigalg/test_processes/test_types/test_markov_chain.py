from math import prod

import numpy as np

from sigalg.processes import MarkovChain

# --------------------- test properties --------------------- #


class TestProbMeasure:
    def test_order_2_prob_measure_factors_according_to_markov_property(self):
        """Test that the probability measure of an order-2 Markov chain factors according to the Markov property."""
        K = np.array([[0.3, 0.7], [0.6, 0.4]])
        pi = np.array([0.15, 0.85])
        X = MarkovChain.generate(
            mode="enum",
            kernel=K,
            initial_distribution=pi,
            length=2,
        )
        P = X.prob_measure

        assert P >> X == (P >> X[0]) * (P.given(X[0]) >> X[1]) * (P.given(X[1]) >> X[2])

    def test_order_3_prob_measure_factors_according_to_markov_property(self):
        """Test that the probability measure of an order-3 Markov chain factors according to the Markov property."""
        rng = np.random.default_rng(42)
        n = 2
        k = 3
        length = 5
        K = rng.integers(low=0, high=101, size=(n,) * (k + 1))
        pi = rng.integers(low=0, high=101, size=(n,) * k)
        K = K / np.expand_dims(K.sum(axis=-1), axis=-1)
        pi = pi / pi.sum()

        X = MarkovChain.generate(
            mode="enum",
            kernel=K,
            initial_distribution=pi,
            length=length,
        )
        P = X.prob_measure

        assert P >> X == (P >> X[0, 1, 2]) * prod(
            (P.given(X[t - 3, t - 2, t - 1]) >> X[t]) for t in range(3, length + 1)
        )

    def test_order_4_prob_measure_factors_according_to_markov_property(self):
        """Test that the probability measure of an order-4 Markov chain factors according to the Markov property."""
        rng = np.random.default_rng(42)
        n = 2
        k = 4
        length = 6
        K = rng.integers(low=0, high=101, size=(n,) * (k + 1))
        pi = rng.integers(low=0, high=101, size=(n,) * k)
        K = K / np.expand_dims(K.sum(axis=-1), axis=-1)
        pi = pi / pi.sum()

        X = MarkovChain.generate(
            mode="enum",
            kernel=K,
            initial_distribution=pi,
            length=length,
        )
        P = X.prob_measure

        assert P >> X == (P >> X[0, 1, 2, 3]) * prod(
            (P.given(X[t - 4, t - 3, t - 2, t - 1]) >> X[t])
            for t in range(4, length + 1)
        )

    def test_empirical_measure_of_order_2_markov_chain(self):
        """Test that the empirical distribution of a simulated order-2 Markov chain is approximately correct."""
        rng = np.random.default_rng(42)
        K = [[0.3, 0.7], [0.1, 0.9]]
        pi = [0.4, 0.6]
        T = 3
        X = MarkovChain.generate(
            kernel=K,
            initial_distribution=pi,
            length=T,
            n_trajectories=100_000,
            random_state=rng,
        )
        Y = MarkovChain.generate(
            kernel=K,
            initial_distribution=pi,
            mode="enum",
            length=T,
            name="Y",
        )
        s = X.pushforward().data.copy()

        assert np.allclose(s.sort_index(), Y.measure, rtol=0.08)
