from sigalg.processes import RandomWalk

# --------------------- test properties --------------------- #


class TestProbMeasure:
    def test_prob_measure_factors_according_to_markov_property(self):
        """Test that the probability measure of a random walk factors according to the Markov property."""
        X = RandomWalk.generate(
            mode="enum",
            p=0.7,
            length=2,
        )
        P = X.prob_measure

        assert P >> X == (P >> X[0]) * (P.given(X[0]) >> X[1]) * (
            P.given(X[0, 1]) >> X[2]
        )
