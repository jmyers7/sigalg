from itertools import product

import numpy as np
import pandas as pd
import pytest

from sigalg.core import Time
from sigalg.processes import RandomWalk


class TestConstructor:
    @pytest.fixture
    def time_discrete(self):
        return Time.discrete(length=5)

    @pytest.fixture
    def time_continuous(self):
        return Time.continuous(start=0.0, stop=1.0, num_points=5)

    def test_constructor_with_valid_parameters(self):
        """Test basic construction with valid probability parameter."""
        rw = RandomWalk(p=0.5, is_discrete_time=True, name="X")

        assert rw.name == "X"
        assert rw.p == 0.5
        assert rw.initial_state == 0
        assert rw.is_discrete_state is True
        assert rw.is_discrete_time is True

    def test_constructor_with_discrete_time_index(self, time_discrete):
        """Test construction with discrete time index."""
        rw = RandomWalk(p=0.6, time=time_discrete, name="W")

        assert rw.time == time_discrete
        assert rw.name == "W"

    def test_constructor_with_continuous_time_index(self, time_continuous):
        """Test construction with continuous time index."""
        rw = RandomWalk(p=0.7, time=time_continuous, name="B")

        assert rw.time == time_continuous
        assert rw.name == "B"

    def test_constructor_with_custom_initial_state(self):
        """Test construction with non-zero initial state."""
        rw = RandomWalk(p=0.5, initial_state=10, is_discrete_time=True)

        assert rw.initial_state == 10
        assert rw.p == 0.5

    def test_constructor_with_boundary_probabilities(self):
        """Test construction with probability values at boundaries."""
        rw_0 = RandomWalk(p=0.0, is_discrete_time=True)
        rw_1 = RandomWalk(p=1.0, is_discrete_time=True)

        assert rw_0.p == 0.0
        assert rw_1.p == 1.0

    def test_constructor_invalid_probability_below_zero(self):
        """Test that constructor raises TypeError for probability < 0."""
        with pytest.raises(TypeError, match="p must be a real number between 0 and 1"):
            RandomWalk(p=-0.1, is_discrete_time=True)

    def test_constructor_invalid_probability_above_one(self):
        """Test that constructor raises TypeError for probability > 1."""
        with pytest.raises(TypeError, match="p must be a real number between 0 and 1"):
            RandomWalk(p=1.5, is_discrete_time=True)

    def test_constructor_invalid_probability_type(self):
        """Test that constructor raises TypeError for non-numeric probability."""
        with pytest.raises(TypeError, match="p must be a real number between 0 and 1"):
            RandomWalk(p="0.5", is_discrete_time=True)


class TestDataGeneration:
    @pytest.fixture
    def time_discrete(self):
        return Time.discrete(length=4)

    @pytest.fixture
    def time_continuous(self):
        return Time.continuous(start=0.0, stop=1.0, num_points=4)

    def test_from_simulation_discrete_time(self, time_discrete):
        """Test from_simulation with discrete time."""
        rw = RandomWalk(p=0.6, time=time_discrete, name="X").from_simulation(
            n_trajectories=100, random_state=42
        )

        assert rw.n_trajectories == 100
        assert rw.time == time_discrete
        assert rw.is_discrete_state is True

    def test_from_simulation_continuous_time(self, time_continuous):
        """Test from_simulation with continuous time."""
        rw = RandomWalk(p=0.5, time=time_continuous, name="X").from_simulation(
            n_trajectories=50, random_state=123
        )

        assert rw.n_trajectories == 50
        assert rw.time == time_continuous
        assert rw.is_discrete_state is True

    def test_from_simulation_starts_at_initial_state(self):
        """Test that all trajectories start at initial_state."""
        initial_state = 10
        time = Time.discrete(length=4)
        rw = RandomWalk(p=0.5, initial_state=initial_state, time=time).from_simulation(
            n_trajectories=50, random_state=42
        )

        assert (rw.data.iloc[:, 0] == initial_state).all()

    def test_from_simulation_with_custom_initial_state(self):
        """Test from_simulation with non-zero initial state."""
        initial_state = -5
        time = Time.discrete(length=4)
        rw = RandomWalk(p=0.7, initial_state=initial_state, time=time).from_simulation(
            n_trajectories=30, random_state=42
        )

        assert (rw.data.iloc[:, 0] == initial_state).all()

    def test_from_enumeration_discrete_time(self):
        """Test from_enumeration with discrete time."""
        time = Time.discrete(length=2)
        rw = RandomWalk(p=0.75, time=time).from_enumeration()

        assert rw.n_trajectories == 4
        assert len(rw.time) == 3

    def test_from_enumeration_continuous_time(self):
        """Test from_enumeration with continuous time."""
        time = Time.continuous(start=0.0, stop=1.0, num_points=3)
        rw = RandomWalk(p=0.6, time=time).from_enumeration()

        assert rw.n_trajectories == 4

    def test_from_enumeration_starts_at_initial_state(self):
        """Test that all enumerated trajectories start at initial_state."""
        initial_state = 5
        time = Time.discrete(length=3)
        rw = RandomWalk(
            p=0.5, initial_state=initial_state, time=time
        ).from_enumeration()

        assert (rw.data.iloc[:, 0] == initial_state).all()


class TestTrajectoryProperties:
    def test_trajectory_steps_are_plus_or_minus_one(self):
        """Test that each step is either +1 or -1."""
        time = Time.discrete(length=5)
        rw = RandomWalk(p=0.5, time=time).from_simulation(
            n_trajectories=100, random_state=42
        )

        for i in range(len(rw.time) - 1):
            steps = rw.data.iloc[:, i + 1] - rw.data.iloc[:, i]
            assert steps.isin([-1, 1]).all()

    def test_trajectory_range_bounded_by_time(self):
        """Test that trajectory values are bounded by the number of steps."""
        time = Time.discrete(length=11)
        rw = RandomWalk(p=0.5, time=time).from_simulation(
            n_trajectories=100, random_state=42
        )

        for i in range(len(rw.time)):
            max_displacement = i
            assert (rw.data.iloc[:, i] >= -max_displacement).all()
            assert (rw.data.iloc[:, i] <= max_displacement).all()

    def test_symmetric_walk_has_expected_mean_near_zero(self):
        """Test that symmetric random walk has mean near zero."""
        time = Time.discrete(length=10)
        rw = RandomWalk(p=0.5, initial_state=4, time=time).from_simulation(
            n_trajectories=50_000, random_state=42
        )
        simulated_exp = rw.at[9].expectation()
        actual_exp = 4 + 9 * (2 * 0.5 - 1)

        assert np.all(abs(simulated_exp.data - actual_exp) < 1e-2)

    def test_biased_walk_drifts_in_expected_direction(self):
        """Test that biased random walk drifts in expected direction."""
        time = Time.discrete(length=10)
        rw = RandomWalk(p=0.8, initial_state=4, time=time).from_simulation(
            n_trajectories=50_000, random_state=42
        )
        simulated_exp = rw.at[9].expectation()
        actual_exp = 4 + 9 * (2 * 0.8 - 1)

        assert np.all(abs(simulated_exp.data - actual_exp) < 1e-2)


class TestProbabilityMeasure:
    def test_exact_probability_measure_symmetric_walk(self):
        """Test exact probability measure for symmetric random walk."""
        time = Time.discrete(length=2)
        rw = RandomWalk(p=0.5, time=time).from_enumeration()
        P = rw.prob_measure

        assert all(np.isclose(P.data, 0.25, atol=1e-9))

    def test_exact_probability_measure_biased_walk(self):
        """Test exact probability measure for biased random walk."""
        p = 0.75
        time = Time.discrete(length=3)
        rw = RandomWalk(p=p, time=time).from_enumeration()
        P = rw.prob_measure

        step_indicators = pd.Series(list(product([0, 1], repeat=3)))
        expected_probs = step_indicators.apply(
            lambda x: 0.75 ** sum(x) * 0.25 ** (3 - sum(x))
        )

        assert np.all(abs(P.data - expected_probs) < 1e-8)

    def test_empirical_probability_measure_from_simulation(self):
        """Test empirical probability measure for simulated random walk."""
        p = 0.6
        time = Time.discrete(length=3)
        rw = RandomWalk(p=p, time=time).from_simulation(
            n_trajectories=1000, random_state=42
        )
        P_empirical = rw.range.prob_measure

        assert np.isclose(P_empirical.data.sum(), 1.0)
        assert (P_empirical.data >= 0).all()


class TestPlotTitle:
    def test_plot_title_for_enumerated_walk(self):
        """Test that _plot_title includes 'Enumerated' for enumerated walks."""
        time = Time.discrete(length=3)
        rw = RandomWalk(p=0.5, time=time, name="W").from_enumeration()
        title = rw._plot_title()

        assert "random walk" in title.lower()
        assert "W" in title

    def test_plot_title_for_simulated_walk(self):
        """Test that _plot_title shows 'Random walk' for simulated walks."""
        time = Time.discrete(length=3)
        rw = RandomWalk(p=0.7, time=time, name="X").from_simulation(
            n_trajectories=10, random_state=42
        )
        title = rw._plot_title()

        assert "random walk" in title.lower()
        assert "X" in title


class TestSpecialCases:
    def test_deterministic_walk_p_equals_one(self):
        """Test random walk with p=1.0 (always steps right)."""
        time = Time.discrete(length=5)
        rw = RandomWalk(p=1.0, time=time).from_simulation(
            n_trajectories=50, random_state=42
        )

        for i in range(len(rw.time) - 1):
            steps = rw.data.iloc[:, i + 1] - rw.data.iloc[:, i]
            assert (steps == 1).all()

    def test_deterministic_walk_p_equals_zero(self):
        """Test random walk with p=0.0 (always steps left)."""
        time = Time.discrete(length=5)
        rw = RandomWalk(p=0.0, time=time).from_simulation(
            n_trajectories=50, random_state=42
        )

        for i in range(len(rw.time) - 1):
            steps = rw.data.iloc[:, i + 1] - rw.data.iloc[:, i]
            assert (steps == -1).all()
