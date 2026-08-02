from scipy.stats import bernoulli

from sigalg.processes import IIDProcess

# --------------------- test properties --------------------- #


class TestProbMeasure:
    def test_prob_measure_factors_according_to_markov_property(self):
        """Test that the probability measure of an IID process factors according to the Markov property."""
        X = IIDProcess.generate(
            mode="enum",
            distribution=bernoulli(p=0.6),
            support=[0, 1],
            length=2,
        )
        P = X.prob_measure

        assert P >> X == (P >> X[0]) * (P >> X[1]) * (P >> X[2])
