from math import ceil, sqrt

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from sigalg.core import Time
from sigalg.processes import PoissonProcess


class TestConstructor:

    @pytest.fixture
    def time_continuous(self):
        return Time.continuous(start=0.0, stop=5.0, num_points=10)

    def test_constructor_with_valid_parameters(self, time_continuous):
        """Test basic construction with valid parameters."""
        pp = PoissonProcess(rate=2.5, max_count=20, time=time_continuous, name="N")

        assert pp.name == "N"
        assert pp.rate == 2.5
        assert pp.max_count == 20
        assert pp.time == time_continuous
        assert pp.is_discrete_state is True

    def test_constructor_with_integer_rate(self, time_continuous):
        """Test construction with integer rate parameter."""
        pp = PoissonProcess(rate=5, max_count=30, time=time_continuous)

        assert pp.rate == 5
        assert pp.max_count == 30

    def test_constructor_with_zero_rate(self, time_continuous):
        """Test construction with rate=0."""
        with pytest.raises(TypeError, match="rate must be a positive real number"):
            PoissonProcess(rate=0, max_count=20, time=time_continuous)

    def test_constructor_invalid_rate_negative(self, time_continuous):
        """Test that constructor raises TypeError for negative rate."""
        with pytest.raises(TypeError, match="rate must be a positive real number"):
            PoissonProcess(rate=-1.5, max_count=20, time=time_continuous)

    def test_constructor_invalid_rate_type(self, time_continuous):
        """Test that constructor raises TypeError for non-numeric rate."""
        with pytest.raises(TypeError, match="rate must be a positive real number"):
            PoissonProcess(rate="2.5", max_count=20, time=time_continuous)

    def test_constructor_invalid_max_count_zero(self, time_continuous):
        """Test that constructor raises TypeError for max_count=0."""
        with pytest.raises(TypeError, match="max_count must be a positive integer"):
            PoissonProcess(rate=2.5, max_count=0, time=time_continuous)

    def test_constructor_invalid_max_count_negative(self, time_continuous):
        """Test that constructor raises TypeError for negative max_count."""
        with pytest.raises(TypeError, match="max_count must be a positive integer"):
            PoissonProcess(rate=2.5, max_count=-10, time=time_continuous)

    def test_constructor_invalid_max_count_type(self, time_continuous):
        """Test that constructor raises TypeError for non-integer max_count."""
        with pytest.raises(TypeError, match="max_count must be a positive integer"):
            PoissonProcess(rate=2.5, max_count=20.5, time=time_continuous)


class TestDataGeneration:

    @pytest.fixture
    def time_continuous(self):
        return Time.continuous(start=0.0, stop=3.0, num_points=10)

    @pytest.fixture
    def rate(self):
        return 5.0

    @pytest.fixture
    def max_count(self, rate, time_continuous):
        t_stop = time_continuous[-1]
        return ceil(rate * t_stop + 3 * sqrt(rate * t_stop))

    def test_from_simulation_basic(self, rate, max_count, time_continuous):
        """Test from_simulation with basic Poisson process."""
        pp = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=50, random_state=42)

        assert pp.n_trajectories == 50
        assert pp._is_enumerated is False
        assert pp.is_discrete_state is True
        assert len(pp.time) <= len(time_continuous)

    def test_from_simulation_starts_at_zero(self, rate, max_count, time_continuous):
        """Test that all trajectories start at 0."""
        pp = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=30, random_state=42)

        assert (pp.data.iloc[:, 0] == 0).all()

    def test_from_simulation_reproducibility(self, rate, max_count, time_continuous):
        """Test that from_simulation with same random_state gives same results."""
        pp1 = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=20, random_state=42)

        pp2 = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=20, random_state=42)

        pd.testing.assert_frame_equal(pp1.data, pp2.data)

    def test_from_enumeration_raises_not_implemented(
        self, rate, max_count, time_continuous
    ):
        """Test that from_enumeration raises NotImplementedError."""
        pp = PoissonProcess(rate=rate, max_count=max_count, time=time_continuous)

        with pytest.raises(
            NotImplementedError, match="Not implemented for PoissonProcess"
        ):
            pp.from_enumeration()


class TestTrajectoryProperties:

    @pytest.fixture
    def time_continuous(self):
        return Time.continuous(start=0.0, stop=4.0, num_points=20)

    @pytest.fixture
    def rate(self):
        return 6.0

    @pytest.fixture
    def max_count(self, rate, time_continuous):
        t_stop = time_continuous[-1]
        return ceil(rate * t_stop + 3 * sqrt(rate * t_stop))

    def test_trajectories_are_non_decreasing(self, rate, max_count, time_continuous):
        """Test that all trajectories are non-decreasing."""
        pp = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=50, random_state=42)

        for i in range(len(pp.time) - 1):
            differences = pp.data.iloc[:, i + 1] - pp.data.iloc[:, i]
            assert (differences >= 0).all()

    def test_trajectories_are_non_negative(self, rate, max_count, time_continuous):
        """Test that all trajectory values are non-negative."""
        pp = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=50, random_state=42)

        assert (pp.data >= 0).all().all()

    def test_trajectories_are_integers(self, rate, max_count, time_continuous):
        """Test that all trajectory values are integers."""
        pp = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=50, random_state=42)

        assert (pp.data == pp.data.astype(int)).all().all()

    def test_final_counts_have_expected_mean(self, rate, max_count, time_continuous):
        """Test that final counts have approximately correct mean."""
        pp = PoissonProcess(
            rate=rate, max_count=max_count, time=time_continuous
        ).from_simulation(n_trajectories=10_000, random_state=42)

        final_time = pp.time[-1]
        expected_mean = rate * final_time
        observed_mean = pp.data.iloc[:, -1].mean()

        assert abs(observed_mean - expected_mean) < 0.5


class TestProbabilityMeasure:

    def test_empirical_probability_measure_from_simulation(self):
        """Test empirical probability measure for final counts."""
        time = Time.continuous(start=0.0, stop=3.0, num_points=10)
        rate = 8.0
        t_stop = time[-1]
        max_count = ceil(rate * t_stop + 3 * sqrt(rate * t_stop))

        pp = PoissonProcess(rate=rate, max_count=max_count, time=time).from_simulation(
            n_trajectories=50_000, random_state=42
        )

        final_counts = pp.last_rv.range
        P_empirical = final_counts.probability_measure

        assert np.isclose(P_empirical.data.sum(), 1.0)
        assert (P_empirical.data >= 0).all()

        final_time = pp.time[-1]
        theoretical_dist = poisson(mu=rate * final_time)
        observed_outputs = final_counts.sample_space.data
        theoretical_probs = theoretical_dist.pmf(observed_outputs)

        assert abs(P_empirical.data - theoretical_probs).sum() < 0.05


class TestPlotTitle:

    def test_plot_title_includes_name(self):
        """Test that _plot_title includes process name."""
        time = Time.continuous(start=0.0, stop=2.0, num_points=5)
        pp = PoissonProcess(
            rate=3.0, max_count=20, time=time, name="N"
        ).from_simulation(n_trajectories=10, random_state=42)

        title = pp._plot_title()

        assert "poisson" in title.lower()
        assert "N" in title


class TestSpecialCases:

    def test_small_max_count_truncates_time(self):
        """Test that small max_count leads to time truncation."""
        rate = 5.0
        max_count = 10
        time = Time.continuous(start=0.0, stop=10.0, num_points=50)

        pp = PoissonProcess(rate=rate, max_count=max_count, time=time).from_simulation(
            n_trajectories=20, random_state=42
        )

        assert len(pp.time) < len(time)

    def test_large_max_count_minimal_truncation(self):
        """Test that large max_count results in minimal time truncation."""
        rate = 2.0
        time = Time.continuous(start=0.0, stop=1.0, num_points=5)
        t_stop = time[-1]
        max_count = ceil(10 * rate * t_stop)

        pp = PoissonProcess(rate=rate, max_count=max_count, time=time).from_simulation(
            n_trajectories=20, random_state=42
        )

        assert len(pp.time) >= len(time) - 1


def test_is_discrete_state():
    """Test that PoissonProcess is always discrete state."""
    time = Time.continuous(start=0.0, stop=2.0, num_points=5)
    rate = 3.0
    max_count = 20

    pp = PoissonProcess(rate=rate, max_count=max_count, time=time).from_simulation(
        n_trajectories=10, random_state=42
    )

    assert pp.is_discrete_state is True
