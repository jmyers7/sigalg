import numpy as np
import pandas as pd
from scipy.stats import bernoulli, norm, poisson

from sigalg.core import Time
from sigalg.processes import IIDProcess


class TestConstructor:

    def test_constructor_with_distribution(self):
        """Test basic construction with a distribution."""
        dist = bernoulli(p=0.5)
        time = Time.discrete(length=3)

        X = IIDProcess(distribution=dist, index=time, name="X")

        assert X.name == "X"
        assert X.time == time
        assert X.distribution == dist

    def test_constructor_minimal(self):
        """Test construction with minimal parameters."""
        dist = bernoulli(p=0.3)
        X = IIDProcess(distribution=dist)

        assert X.name == "X"
        assert X.distribution == dist


class TestDataGeneration:

    def test_from_enumeration_bernoulli(self):
        """Test from_enumeration with Bernoulli distribution."""
        dist = bernoulli(p=0.6)
        time = Time.discrete(length=3)
        X = IIDProcess(
            distribution=dist, index=time, name="Bernoulli"
        ).from_enumeration(support=[0, 1], length=3)

        assert X.n_trajectories == 8
        assert X.is_enumerated is True

        trajectories = [tuple(row) for row in X.data.values]
        assert len(trajectories) == 8
        assert (0, 0, 0) in trajectories
        assert (1, 1, 1) in trajectories

    def test_from_enumeration_creates_time_if_not_provided(self):
        """Test from_enumeration creates time index when not provided."""
        dist = bernoulli(p=0.5)
        X = IIDProcess(distribution=dist).from_enumeration(support=[0, 1], length=2)
        expected_time = Time.discrete(length=2)

        assert X.time == expected_time
        assert X.n_trajectories == 4

    def test_from_simulation_bernoulli(self):
        """Test from_simulation with Bernoulli distribution."""
        dist = bernoulli(p=0.5)
        time = Time.discrete(length=5)
        X = IIDProcess(distribution=dist, index=time, name="Bernoulli").from_simulation(
            max_trajectories=100, random_state=42
        )

        assert len(X.data) == 100
        assert X.is_enumerated is False
        assert X.time == time
        assert X.data.isin([0, 1]).all().all()

    def test_from_simulation_poisson(self):
        """Test from_simulation with Poisson distribution."""
        dist = poisson(mu=2.0)
        time = Time.discrete(length=2)

        X = IIDProcess(distribution=dist, index=time, name="Poisson").from_simulation(
            max_trajectories=50, random_state=123
        )

        assert len(X.data) == 50
        assert X.is_enumerated is False
        assert X.time == time
        assert (X.data >= 0).all().all()

    def test_from_simulation_creates_time_if_not_provided(self):
        """Test from_simulation creates time index when not provided."""
        dist = bernoulli(p=0.4)
        X = IIDProcess(distribution=dist).from_simulation(
            max_trajectories=10, length=3, random_state=42
        )
        expected_time = Time.discrete(length=3)

        assert X.time == expected_time

    def test_from_simulation_reproducibility(self):
        """Test that from_simulation with same random_state gives same results."""
        dist = bernoulli(p=0.5)
        time = Time.discrete(length=3)
        X1 = IIDProcess(distribution=dist, index=time).from_simulation(
            max_trajectories=20, random_state=42
        )
        X2 = IIDProcess(distribution=dist, index=time).from_simulation(
            max_trajectories=20, random_state=42
        )

        pd.testing.assert_frame_equal(X1.data, X2.data)


def test_is_discrete_time_and_state():
    """Test is_discrete_time and is_discrete_state for Bernoulli IID process."""
    dist_discrete = bernoulli(p=0.5)
    dist_continuous = norm(loc=0.0, scale=1.0)
    time_discrete = Time.discrete(length=3)
    time_continuous = Time.continuous(start=0.0, stop=1.0, num_points=5)

    X = IIDProcess(distribution=dist_discrete, index=time_discrete).from_enumeration(
        support=[0, 1]
    )
    Y = IIDProcess(distribution=dist_discrete, index=time_continuous).from_enumeration(
        support=[0, 1]
    )
    Z = IIDProcess(distribution=dist_continuous, index=time_discrete).from_simulation(
        max_trajectories=2, random_state=42
    )
    W = IIDProcess(distribution=dist_continuous, index=time_continuous).from_simulation(
        max_trajectories=2, random_state=42
    )

    assert X.is_discrete_time is True
    assert X.is_discrete_state is True
    assert Y.is_discrete_time is False
    assert Y.is_discrete_state is True
    assert Z.is_discrete_time is True
    assert Z.is_discrete_state is False
    assert W.is_discrete_time is False
    assert W.is_discrete_state is False


class TestProbabilityMeasure:

    def test_exact_probability_measure_bernoulli(self):
        """Test exact probability measure for enumerated IID Bernoulli process."""
        p = 0.6
        dist = bernoulli(p=p)
        X = IIDProcess(distribution=dist).from_enumeration(support=[0, 1], length=2)
        P = X.probability_measure
        expected_probabilities = pd.Series(
            [0.16, 0.24, 0.24, 0.36], index=X.domain.data, name="probability"
        )

        pd.testing.assert_series_equal(P.data, expected_probabilities)

    def test_empirical_probability_measure_from_simulation(self):
        """Test empirical probability measure for simulated IID process."""
        dist = bernoulli(p=0.5)
        X = IIDProcess(distribution=dist).from_simulation(
            max_trajectories=100_000,
            length=2,
            random_state=42,
        )
        P_X = X.range.probability_measure

        assert all(np.isclose(P_X.data, 0.25, rtol=0, atol=0.01))


class TestTransformations:

    def test_cumsum_on_iid_process(self):
        """Test cumsum transformation on IID process."""
        dist = bernoulli(p=0.5)
        time = Time.discrete(length=3)
        X = IIDProcess(distribution=dist, index=time).from_enumeration(support=[0, 1])
        Y = X.cumsum()

        assert Y.is_monotonic(increasing=True)

    def test_pointwise_map_on_iid_process(self):
        """Test pointwise_map transformation on IID process."""
        dist = bernoulli(p=0.5)
        time = Time.discrete(length=2)
        X = IIDProcess(distribution=dist, index=time).from_enumeration(
            support=[0, 1], length=2
        )
        Y = X.pointwise_map(lambda x: x * 2)

        assert Y.data.isin([0, 2]).all().all()


class TestPlotTitle:

    def test_plot_title_includes_distribution_name(self):
        """Test that _plot_title includes distribution name."""
        dist = bernoulli(p=0.5)
        X = IIDProcess(distribution=dist, name="X").from_enumeration(
            support=[0, 1], length=2
        )
        title = X._plot_title()

        assert "bernoulli" in title.lower()
