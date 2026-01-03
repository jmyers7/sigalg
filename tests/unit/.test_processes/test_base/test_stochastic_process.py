import pandas as pd
import pytest

from sigalg.core import SampleSpace, Time
from sigalg.processes import StochasticProcess


class TestConstructor:

    def test_constructor(self):
        time = Time.discrete(length=2)
        domain = SampleSpace.generate_default(size=4)
        X = StochasticProcess(
            outputs={
                "omega0": (0, 0),
                "omega1": (0, 1),
                "omega2": (1, 0),
                "omega3": (1, 1),
            },
            domain=domain,
            name="X",
            vector_index=time,
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
        time = Time.discrete(length=3)
        domain = SampleSpace.generate_default(size=4)
        X = StochasticProcess(
            outputs={
                "omega0": (0, 0, 0),
                "omega1": (0, 1, 2),
                "omega2": (1, 0, 1),
                "omega3": (1, 1, 0),
            },
            domain=domain,
            name="X",
            vector_index=time,
        )

        rv_at_1 = X.rv_at[1]

        expected_values = pd.Series(
            [0, 1, 0, 1],
            index=domain.data,
            name="X_1",
        )
        pd.testing.assert_series_equal(rv_at_1.data, expected_values)
        assert rv_at_1.name == "X_1"

    @pytest.mark.parametrize(
        "time_point, expected_values, expected_name",
        [
            pytest.param(
                0.0,
                [0.0, 0.0, 1.0, 1.0],
                "X_0.0",
                id="time_point_0",
            ),
            pytest.param(
                1.0,
                [0.5, 1.5, 0.5, 1.5],
                "X_1.0",
                id="time_point_1",
            ),
            pytest.param(
                2.0,
                [1.0, 2.0, 1.0, 0.0],
                "X_2.0",
                id="time_point_2",
            ),
        ],
    )
    def test_rv_at_with_continuous_times(
        self, time_point, expected_values, expected_name
    ):
        time = Time.continuous(start=0.0, stop=2.0, num_points=3)
        domain = SampleSpace.generate_default(size=4)
        X = StochasticProcess(
            outputs={
                "omega0": (0.0, 0.5, 1.0),
                "omega1": (0.0, 1.5, 2.0),
                "omega2": (1.0, 0.5, 1.0),
                "omega3": (1.0, 1.5, 0.0),
            },
            domain=domain,
            name="X",
            vector_index=time,
        )

        rv_at = X.rv_at[time_point]

        expected_values = pd.Series(
            expected_values,
            index=domain.data,
            name=expected_name,
        )
        pd.testing.assert_series_equal(rv_at.data, expected_values)
        assert rv_at.name == expected_name
