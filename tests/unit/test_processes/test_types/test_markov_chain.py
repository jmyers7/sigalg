import pandas as pd

from sigalg.core import ProbabilityMeasure, SampleSpace
from sigalg.processes import MarkovChain

# --------------------- test properties --------------------- #


class TestProbMeasure:
    def test_prob_measure_factors_according_to_markov_property(self):
        """Test that the probability measure of a Markov chain factors according to the Markov property."""
        Omega = SampleSpace.from_sequence(size=2)
        K = pd.DataFrame([[0.3, 0.7], [0.6, 0.4]], index=Omega, columns=Omega)
        pi = ProbabilityMeasure(
            sample_space=Omega,
            mapping={
                0: 0.15,
                1: 0.85,
            },
            name="pi",
        )
        X = MarkovChain.generate(
            mode="enum",
            transition_matrix=K,
            initial_distribution=pi,
            length=2,
        )
        P = X.prob_measure

        assert P >> X == (P >> X[0]) * (P.given(X[0]) >> X[1]) * (P.given(X[1]) >> X[2])
