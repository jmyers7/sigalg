import sigalg as sa
import pytest


class TestInitProbabilityMeasure:

    @pytest.fixture
    def sample_space(self):
        state_space = [0, 1]
        time = sa.Time.from_discrete_params(initial_time=0, trajectory_length=3)
        sample_space = sa.SampleSpace.create_from_sequences(state_space, time)
        return sample_space

    def test_init_probability_measure_valid(self, sample_space):

        def measure_function(event):
            num_heads = event.sum()
            num_tails = event.num_features - num_heads
            prob = (0.75**num_heads) * (0.25**num_tails)
            return prob.sum()

        _ = sa.ProbabilityMeasure(sample_space, measure_function)

    def test_init_probability_measure_invalid(self, sample_space):

        def measure_function(event):
            num_heads = event.sum()
            num_tails = event.num_features - num_heads
            prob = (0.8**num_heads) * (0.4**num_tails)
            return prob.sum()

        with pytest.raises(ValueError):
            _ = sa.ProbabilityMeasure(sample_space, measure_function)
    
    def test_init_probability_measure_invalid_sample_space(self):

        def measure_function(event):
            return 1.0

        with pytest.raises(TypeError):
            _ = sa.ProbabilityMeasure("not_a_sample_space", measure_function)


class TestProbabilityMeasureMethods:

    @pytest.fixture
    def probability_measure(self):
        state_space = [0, 1]
        time = sa.Time.from_discrete_params(initial_time=0, trajectory_length=3)
        sample_space = sa.SampleSpace.create_from_sequences(state_space, time)

        def measure_function(event):
            num_heads = event.sum()
            num_tails = event.num_features - num_heads
            prob = (0.75**num_heads) * (0.25**num_tails)
            return prob.sum()

        probability_measure = sa.ProbabilityMeasure(sample_space, measure_function)
        return probability_measure

    def test_call_probability_measure(self, probability_measure):
        sample_space = probability_measure.sample_space
        event_A = sample_space[["omega1", "omega3", "omega4"]]
        p = probability_measure(event_A)
        expected_p = 0.25**3 + 0.75 * 0.25**2 + 0.75**2 * 0.25
        assert abs(p - expected_p) < 1e-8
