from math import prod

import numpy as np
from sigalg.core import (
    Domain,
    ParametrizedProbabilityMeasure,
    ProbabilityMeasure,
    ProbabilitySpace,
)
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

        assert np.allclose(
            P >> X,
            (P >> X[0]) * (P.conditional(X[0]) >> X[1]) * (P.conditional(X[1]) >> X[2]),
        )

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

        assert np.allclose(
            P >> X,
            (P >> X[0, 1, 2])
            * prod(
                (P.conditional(X[t - 3, t - 2, t - 1]) >> X[t])
                for t in range(3, length + 1)
            ),
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

        assert np.allclose(
            P >> X,
            (P >> X[0, 1, 2, 3])
            * prod(
                (P.conditional(X[t - 4, t - 3, t - 2, t - 1]) >> X[t])
                for t in range(4, length + 1)
            ),
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

    def test_creating_order_2_markov_chain_from_scratch(self):
        """Test creating an order-2 Markov chain from scratch and verifying it matches a MarkovChain instance."""
        X0 = Domain.from_sequence(size=2, variable_name="x_0")
        X1 = Domain.from_sequence(size=2, variable_name="x_1")
        X2 = Domain.from_sequence(size=2, variable_name="x_2")
        pi = ProbabilityMeasure.from_rand(domain=X0, name="pi", random_state=42)
        K_01 = ParametrizedProbabilityMeasure.from_rand(
            measure_domain=X1,
            parameter_domain=X0,
            name="K_01",
            random_state=42,
        )
        K_12 = ParametrizedProbabilityMeasure.from_rand(
            measure_domain=X2,
            parameter_domain=X1,
            name="K_12",
            random_state=42,
        )
        P = pi * K_01 * K_12
        P = P.to_measure(kind="probability")
        markov_chain_from_scratch = ProbabilitySpace(domain=X0 @ X1 @ X2, measure=P)
        generated_markov_chain = MarkovChain.generate(
            kernel=K_01,
            initial_distribution=pi,
            mode="enum",
            length=2,
            name="generated_markov_chain",
        )

        assert np.allclose(
            generated_markov_chain.measure, markov_chain_from_scratch.measure
        )

    def test_creating_order_3_markov_chain_from_scratch(self):
        """Test creating an order-3 Markov chain from scratch and verifying it matches a MarkovChain instance."""
        X0 = Domain.from_sequence(size=2, variable_name="x_0")
        X1 = Domain.from_sequence(size=2, variable_name="x_1")
        X2 = Domain.from_sequence(size=2, variable_name="x_2")
        X3 = Domain.from_sequence(size=2, variable_name="x_3")
        pi = ProbabilityMeasure.from_rand(
            domain=X0 @ X1,
            name="pi",
            random_state=42,
        )
        K_012 = ParametrizedProbabilityMeasure.from_rand(
            measure_domain=X2,
            parameter_domain=X0 @ X1,
            name="K_012",
            random_state=42,
        )
        K_123 = ParametrizedProbabilityMeasure.from_rand(
            measure_domain=X3,
            parameter_domain=X1 @ X2,
            name="K_123",
            random_state=42,
        )
        P = pi * K_012 * K_123
        P = P.to_measure(kind="probability")
        markov_chain_from_scratch = ProbabilitySpace(
            domain=X0 @ X1 @ X2 @ X3, measure=P
        )
        generated_markov_chain = MarkovChain.generate(
            kernel=K_012,
            initial_distribution=pi,
            mode="enum",
            length=3,
            name="generated_markov_chain",
        )

        assert np.allclose(
            generated_markov_chain.measure, markov_chain_from_scratch.measure
        )
