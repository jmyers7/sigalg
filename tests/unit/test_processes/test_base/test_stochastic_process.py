import pandas as pd
import pytest

from sigalg.core import SampleSpace, Time
from sigalg.processes import StochasticProcess


class TestConstructor:

    def test_constructor(self):
        """Test basic construction."""
        time = Time.discrete(length=2)
        domain = SampleSpace().from_sequence(size=4, prefix="omega")
        X = StochasticProcess(domain=domain, name="X", index=time).from_dict(
            {
                "omega_0": (0, 0),
                "omega_1": (0, 1),
                "omega_2": (1, 0),
                "omega_3": (1, 1),
            }
        )

        expected_data = pd.DataFrame(
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            index=domain.data,
            columns=time.data,
        )

        assert X.time == time
        assert X.domain == domain
        assert X.name == "X"
        pd.testing.assert_frame_equal(X.data, expected_data)

    def test_constructor_minimal(self):
        """Test construction with minimal parameters."""
        X = StochasticProcess()

        assert X.name == "X"
        assert X.domain is None
        assert X.time is None


class TestProperties:

    def test_n_trajectories(self):
        """Test n_trajectories property."""
        X = StochasticProcess().from_enumeration(support=[0, 1], length=2)

        assert X.n_trajectories == 4

    def test_trajectory_counts(self):
        """Test trajectory_counts property."""
        X = StochasticProcess().from_enumeration(support=[0, 1], length=2)

        assert len(X.trajectory_counts) == 4
        assert (X.trajectory_counts == 1).all()


class TestDataGeneration:

    def test_from_enumeration(self):
        """Test from_enumeration method."""
        time = Time.discrete(length=2)
        X = StochasticProcess(index=time, name="X").from_enumeration(support=[0, 1])

        assert X.n_trajectories == 4
        assert X.is_enumerated is True

        trajectories = [tuple(row) for row in X.data.values]
        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert sorted(trajectories) == sorted(expected)

    def test_from_enumeration_without_time(self):
        """Test from_enumeration creates time index if not provided."""
        X = StochasticProcess(name="X").from_enumeration(support=[0, 1], length=3)
        expected_time = Time.discrete(length=3)

        assert X.time == expected_time
        assert len(X.time.data) == 3
        assert X.n_trajectories == 8

    def test_from_simulation(self):
        """Test from_simulation method (requires subclass implementation)."""
        X = StochasticProcess(name="X")
        with pytest.raises(NotImplementedError):
            X.from_simulation(max_trajectories=10, length=3, random_state=42)


class TestRVAt:

    def test_rv_at_with_discrete_times(self):
        """Test rv_at method with discrete time index."""
        time = Time.discrete(length=3)
        domain = SampleSpace().from_sequence(size=4, prefix="omega")
        X = StochasticProcess(domain=domain, name="X", index=time).from_dict(
            outputs={
                "omega_0": (0, 0, 0),
                "omega_1": (0, 1, 2),
                "omega_2": (1, 0, 1),
                "omega_3": (1, 1, 0),
            }
        )

        rv_at_1 = X.rv_at[1]

        expected_values = pd.Series(
            [0, 1, 0, 1],
            index=domain.data,
            name="X_1",
        )
        pd.testing.assert_series_equal(rv_at_1.data, expected_values)
        assert rv_at_1.name == "X_1"

    def test_rv_at_with_continuous_times(self):
        """Test rv_at method with continuous time index."""
        time = Time.continuous(start=0.0, stop=2.0, num_points=3)
        domain = SampleSpace().from_sequence(size=4, prefix="omega")
        X = StochasticProcess(domain=domain, name="X", index=time).from_dict(
            outputs={
                "omega_0": (0, 0, 1),
                "omega_1": (0, 2, 2),
                "omega_2": (1, 1, 1),
                "omega_3": (1, 2, 0),
            }
        )

        # Given time is exactly at a time point
        rv_at = X.rv_at[0.0]
        expected_values = pd.Series(
            [0, 0, 1, 1],
            index=domain.data,
            name="X_0.0",
        )
        pd.testing.assert_series_equal(rv_at.data, expected_values)
        assert rv_at.name == "X_0.0"

        # Given time is between two time points
        rv_at = X.rv_at[1.0]
        expected_values = pd.Series(
            [0, 2, 1, 2],
            index=domain.data,
            name="X_1.0",
        )
        pd.testing.assert_series_equal(rv_at.data, expected_values)
        assert rv_at.name == "X_1.0"

        # Given time is exactly at the last time point
        rv_at = X.rv_at[2.0]
        expected_values = pd.Series(
            [1, 2, 1, 0],
            index=domain.data,
            name="X_2.0",
        )
        pd.testing.assert_series_equal(rv_at.data, expected_values)
        assert rv_at.name == "X_2.0"

        # Given time is after the end of the time index
        with pytest.raises(ValueError, match="is after the end"):
            rv_at = X.rv_at[3.0]

        # Given time is before the start of the time index
        with pytest.raises(ValueError, match="is before the start"):
            rv_at = X.rv_at[-1.0]


class TestProbabilityMeasure:

    def test_probability_measure(self):
        """Test that probability measure."""
        X = StochasticProcess(name="X").from_dict(
            outputs={
                "omega_0": (0, 0),
                "omega_1": (0, 1),
                "omega_2": (1, 0),
                "omega_3": (1, 1),
            }
        )

        assert X.probability_measure is None

        Y = StochasticProcess(name="Y").from_enumeration(support=[0, 1], length=2)

        with pytest.raises(NotImplementedError, match="not implemented"):
            _ = Y.probability_measure


class TestTransformations:

    def test_cumsum(self):
        """Test cumsum transformation."""
        time = Time.discrete(length=3)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 2, 3),
                "omega_1": (2, 1, 2),
            }
        )

        Y = X.cumsum()

        expected_data = pd.DataFrame(
            [[1, 3, 6], [2, 3, 5]],
            index=domain.data,
            columns=time.data,
        )

        pd.testing.assert_frame_equal(Y.data, expected_data)
        assert Y.name == "X_cumsum"

    def test_increments(self):
        """Test increments transformation."""
        time = Time.discrete(length=3)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 3, 6),
                "omega_1": (2, 3, 5),
            }
        )

        Y = X.increments()
        print(Y.data)

        assert len(Y.data.columns) == 2

        expected_data = pd.DataFrame(
            [[2, 3], [1, 2]], index=domain.data, columns=time.data[1:]
        )

        pd.testing.assert_frame_equal(Y.data, expected_data)
        assert Y.name == "X_increments"

    def test_increments_raises_on_one_dimensional(self):
        """Test that increments raises error for 1D process."""
        time = Time.discrete(length=1)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time).from_dict(
            {
                "omega_0": 1,
                "omega_1": 2,
            }
        )

        with pytest.raises(ValueError, match="one-dimensional"):
            X.increments()

    def test_pointwise_map(self):
        """Test pointwise_map transformation."""
        time = Time.discrete(length=2)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 2),
                "omega_1": (3, 4),
            }
        )

        Y = X.pointwise_map(lambda x: x * 2)

        expected_data = pd.DataFrame(
            [[2, 4], [6, 8]],
            index=domain.data,
            columns=time.data,
        )

        pd.testing.assert_frame_equal(Y.data, expected_data)

    def test_timewise_map(self):
        """Test timewise_map transformation."""
        time = Time.discrete(length=2)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 2),
                "omega_1": (3, 4),
            }
        )

        Y = X.timewise_map(time=0, function=lambda x: x * 10)

        expected_data = pd.DataFrame(
            [[10, 2], [30, 4]],
            index=domain.data,
            columns=time.data,
        )

        pd.testing.assert_frame_equal(Y.data, expected_data)

    def test_max_value(self):
        """Test max_value method."""
        time = Time.discrete(length=2)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time).from_dict(
            {
                "omega_0": (1, 2),
                "omega_1": (3, 4),
            }
        )

        assert X.max_value() == 4

    def test_is_monotonic_increasing(self):
        """Test is_monotonic with increasing trajectories."""
        time = Time.discrete(length=3)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time).from_dict(
            {
                "omega_0": (1, 2, 3),
                "omega_1": (0, 1, 1),
            }
        )

        assert X.is_monotonic(increasing=True) is True
        assert X.is_monotonic(increasing=False) is False

    def test_is_monotonic_decreasing(self):
        """Test is_monotonic with decreasing trajectories."""
        time = Time.discrete(length=3)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time).from_dict(
            {
                "omega_0": (3, 2, 1),
                "omega_1": (1, 1, 0),
            }
        )

        assert X.is_monotonic(increasing=False) is True
        assert X.is_monotonic(increasing=True) is False

    def test_is_monotonic_non_monotonic(self):
        """Test is_monotonic with non-monotonic trajectories."""
        time = Time.discrete(length=3)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")
        X = StochasticProcess(domain=domain, index=time).from_dict(
            {
                "omega_0": (1, 3, 2),
                "omega_1": (2, 1, 3),
            }
        )

        assert X.is_monotonic(increasing=True) is False
        assert X.is_monotonic(increasing=False) is False


class TestEquality:

    def test_equality(self):
        """Test equality between stochastic processes."""
        time = Time.discrete(length=2)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")

        X1 = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 2),
                "omega_1": (3, 4),
            }
        )

        X2 = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 2),
                "omega_1": (3, 4),
            }
        )

        assert X1 == X2

    def test_inequality_different_data(self):
        """Test inequality with different data."""
        time = Time.discrete(length=2)
        domain = SampleSpace().from_sequence(size=2, prefix="omega")

        X1 = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 2),
                "omega_1": (3, 4),
            }
        )

        X2 = StochasticProcess(domain=domain, index=time, name="X").from_dict(
            {
                "omega_0": (1, 2),
                "omega_1": (3, 5),  # Different value
            }
        )

        assert X1 != X2
