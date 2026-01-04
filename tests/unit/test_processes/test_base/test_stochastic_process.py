import pandas as pd
import pytest

from sigalg.core import SampleSpace, Time
from sigalg.processes import StochasticProcess


class TestConstructor:

    def test_constructor(self):
        """Test basic construction."""
        time = Time.discrete(length=2)
        domain = SampleSpace().from_sequence(size=4, prefix="omega")
        X = StochasticProcess(domain=domain, name="X", vector_index=time).from_dict(
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


class TestRVAt:

    def test_rv_at_with_discrete_times(self):
        """Test rv_at method with discrete time index."""
        time = Time.discrete(length=3)
        domain = SampleSpace().from_sequence(size=4, prefix="omega")
        X = StochasticProcess(domain=domain, name="X", vector_index=time).from_dict(
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
        X = StochasticProcess(domain=domain, name="X", vector_index=time).from_dict(
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
