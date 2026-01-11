import numpy as np
import pandas as pd
import pytest

from sigalg.core import ProbabilityMeasure, SampleSpace, Time
from sigalg.processes import MarkovChain


class TestConstructor:

    def test_constructor_with_valid_parameters(self):
        """Test basic construction with valid transition matrix and initial distribution."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
            name="X",
        )

        assert mc.name == "X"
        assert mc.n_states == 2
        assert mc.states == ["A", "B"]
        assert mc.transition_matrix.equals(P)
        assert mc.initial_distribution == pi
        assert mc.is_discrete_state is True

    def test_constructor_with_time_index(self):
        """Test construction with a specified time index."""
        state_space = SampleSpace().from_list(["rain", "sun"])
        P = pd.DataFrame(
            [[0.9, 0.1], [0.4, 0.6]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"rain": 0.25, "sun": 0.75})
        time = Time.discrete(length=5)

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
            index=time,
            name="Weather",
        )

        assert mc.time == time
        assert mc.name == "Weather"

    def test_constructor_invalid_transition_matrix_type(self):
        """Test that constructor raises TypeError for non-DataFrame transition matrix."""
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        with pytest.raises(
            TypeError, match="transition_matrix must be a pandas DataFrame"
        ):
            MarkovChain(
                transition_matrix=[[0.7, 0.3], [0.4, 0.6]],
                initial_distribution=pi,
            )

    def test_constructor_invalid_initial_distribution_type(self):
        """Test that constructor raises TypeError for non-ProbabilityMeasure initial distribution."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]],
            index=state_space.data,
            columns=state_space.data,
        )

        with pytest.raises(
            TypeError, match="initial_distribution must be a ProbabilityMeasure"
        ):
            MarkovChain(
                transition_matrix=P,
                initial_distribution={"A": 0.5, "B": 0.5},
            )

    def test_constructor_mismatched_states(self):
        """Test that constructor raises ValueError when states don't match."""
        state_space_1 = SampleSpace().from_list(["A", "B"])
        state_space_2 = SampleSpace().from_list(["X", "Y"])
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]],
            index=state_space_1.data,
            columns=state_space_1.data,
        )
        pi = ProbabilityMeasure(sample_space=state_space_2).from_dict(
            {"X": 0.5, "Y": 0.5}
        )

        with pytest.raises(
            ValueError, match="transition_matrix index and columns must match"
        ):
            MarkovChain(transition_matrix=P, initial_distribution=pi)

    def test_constructor_transition_matrix_rows_not_sum_to_one(self):
        """Test that constructor raises ValueError when rows don't sum to 1."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.7, 0.2], [0.4, 0.6]],  # First row sums to 0.9
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        with pytest.raises(
            ValueError, match="Each row of transition_matrix must sum to 1"
        ):
            MarkovChain(transition_matrix=P, initial_distribution=pi)

    def test_constructor_negative_transition_probabilities(self):
        """Test that constructor raises ValueError for negative probabilities."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.7, 0.3], [-0.1, 1.1]],  # Negative probability
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        with pytest.raises(
            ValueError, match="All entries in transition_matrix must be non-negative"
        ):
            MarkovChain(transition_matrix=P, initial_distribution=pi)


class TestDataGeneration:

    def test_from_simulation_basic(self):
        """Test from_simulation with basic Markov chain."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.8, 0.2], [0.3, 0.7]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
        ).from_simulation(n_trajectories=100, length=5, random_state=42)

        assert mc.n_trajectories == 100
        assert len(mc.time) == 5
        assert mc.is_enumerated is False
        assert mc.data.isin(["A", "B"]).all().all()

    def test_from_simulation_with_existing_time(self):
        """Test from_simulation when time index is already provided."""
        state_space = SampleSpace().from_list(["rain", "sun"])
        P = pd.DataFrame(
            [[0.9, 0.1], [0.4, 0.6]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"rain": 0.25, "sun": 0.75})
        time = Time.discrete(length=3)

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
            index=time,
        ).from_simulation(n_trajectories=50, random_state=123)

        assert mc.time == time
        assert mc.n_trajectories == 50

    def test_from_simulation_creates_time_if_not_provided(self):
        """Test from_simulation creates time index when not provided."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.6, 0.4], [0.5, 0.5]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.7, "B": 0.3})

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
        ).from_simulation(n_trajectories=20, length=4, random_state=42)

        expected_time = Time.discrete(length=4)
        assert mc.time == expected_time

    def test_from_simulation_reproducibility(self):
        """Test that from_simulation with same random_state gives same results."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        mc1 = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
        ).from_simulation(n_trajectories=50, length=3, random_state=42)

        mc2 = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
        ).from_simulation(n_trajectories=50, length=3, random_state=42)

        pd.testing.assert_frame_equal(mc1.data, mc2.data)

    def test_from_enumeration_binary_states(self):
        """Test from_enumeration with two-state Markov chain."""
        state_space = SampleSpace().from_list([0, 1])
        P = pd.DataFrame(
            [[0.6, 0.4], [0.3, 0.7]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({0: 0.5, 1: 0.5})

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
        ).from_enumeration(length=2)

        assert mc.n_trajectories == 4  # 2^2 trajectories
        assert mc.is_enumerated is True

        trajectories = [tuple(row) for row in mc.data.values]
        assert (0, 0) in trajectories
        assert (0, 1) in trajectories
        assert (1, 0) in trajectories
        assert (1, 1) in trajectories


class TestProbabilityMeasure:

    @pytest.fixture
    def mc(self):
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [
                [0.8, 0.2],  # P(A | A) = 0.8,  P(B | A) = 0.2
                [0.3, 0.7],  # P(A | B) = 0.3,  P(B | B) = 0.7
            ],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.6, "B": 0.4})

        return MarkovChain(transition_matrix=P, initial_distribution=pi)

    @pytest.fixture
    def expected_probabilities(self, mc):
        return pd.Series(
            [
                0.8 * 0.6,  # P(A, A) = P(A | A) * P(A) = 0.8 * 0.6
                0.2 * 0.6,  # P(A, B) = P(B | A) * P(A) = 0.2 * 0.6
                0.3 * 0.4,  # P(B, A) = P(A | B) * P(B) = 0.3 * 0.4
                0.7 * 0.4,  # P(B, B) = P(B | B) * P(B) = 0.7 * 0.4
            ]
        )

    def test_exact_probability_measure_two_states(self, mc, expected_probabilities):
        """Test exact probability measure for enumerated two-state Markov chain."""
        mc.from_enumeration(length=2)
        P_mc = mc.probability_measure
        expected_probabilities.index = mc.domain.data

        assert all(np.isclose(P_mc.data, expected_probabilities))

    def test_empirical_probability_measure_from_simulation(
        self, mc, expected_probabilities
    ):
        """Test empirical probability measure for simulated Markov chain."""
        mc.from_simulation(n_trajectories=100_000, length=2, random_state=42)
        P_mc = mc.range.probability_measure
        expected_probabilities.index = mc.range.domain.data

        assert all(np.isclose(P_mc.data, expected_probabilities, atol=0.01))


class TestPlotTitle:

    def test_plot_title_for_enumerated_chain(self):
        """Test that _plot_title includes 'Enumerated' for enumerated chains."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
            name="MC",
        ).from_enumeration(length=2)

        title = mc._plot_title()

        assert "enumerated" in title.lower()
        assert "markov chain" in title.lower()
        assert "MC" in title

    def test_plot_title_for_simulated_chain(self):
        """Test that _plot_title shows 'Markov chain' for simulated chains."""
        state_space = SampleSpace().from_list(["A", "B"])
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]],
            index=state_space.data,
            columns=state_space.data,
        )
        pi = ProbabilityMeasure().from_dict({"A": 0.5, "B": 0.5})

        mc = MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
            name="MC",
        ).from_simulation(n_trajectories=10, length=2, random_state=42)

        title = mc._plot_title()

        assert "markov chain" in title.lower()
        assert "enumerated" not in title.lower()
        assert "MC" in title
