import pytest
import sigalg as sa
import pandas as pd
import numpy as np


class TestTimeInit:

    def test_time_init_with_non_num(self):
        time_points = ["t1", "t2", "t3"]

        with pytest.raises(ValueError):
            _ = sa.Time(data=time_points, dtype="object")

    def test_time_init_with_non_increasing(self):
        time_points = [0, 2, 1, 3]

        with pytest.raises(ValueError):
            _ = sa.Time(data=time_points, dtype="float")

    def test_time_init_with_one_time(self):
        time_points = [0]

        time = sa.Time(data=time_points, dtype="float")

        assert len(time) == 1
        assert time.to_list() == [0]

    def test_time_init_with_duplicates(self):
        time_points = [0, 1, 1, 2]

        with pytest.raises(ValueError):
            _ = sa.Time(data=time_points, dtype="float")


class TestTimeProperties:

    @pytest.fixture
    def time(self):
        time_points = [0, 1, 2, 3]
        return sa.Time(data=time_points, dtype="float", name="discrete_time")

    def test_name(self, time):
        assert time.name == "discrete_time"


class TestTimeMethods:

    @pytest.fixture
    def time(self):
        time_points = [0, 1, 2, 3]
        return sa.Time(data=time_points, dtype="float", name="discrete_time")

    def test_to_list(self, time):
        assert time.to_list() == [0, 1, 2, 3]

    def test_to_pandas(self, time):
        pd_index = time.to_pandas()
        expected_index = pd.Index([0, 1, 2, 3], dtype="float", name="discrete_time")
        pd.testing.assert_index_equal(pd_index, expected_index)

    def test_to_numpy(self, time):
        np_array = time.to_numpy()
        expected_array = pd.Index([0, 1, 2, 3], dtype="float").to_numpy()
        assert (np_array == expected_array).all()

    def test_cardinality(self, time):
        assert time.cardinality() == 4

    def test_getitem(self, time):
        assert time[1] == 1
        assert time[0:2].to_list() == [0, 1]

    def test_iter(self, time):
        times = [t for t in time]
        assert times == [0, 1, 2, 3]

    def test_contains(self, time):
        assert 2 in time
        assert 5 not in time

    def test_array(self, time):
        np_array = np.array(time)
        expected_array = pd.Index([0, 1, 2, 3], dtype="float").to_numpy()
        assert (np_array == expected_array).all()


class TestTimeClassMethods:

    def test_from_list(self):
        time_points = [0, 1, 2, 3]
        time = sa.Time.from_list(time_points, name="discrete_time")

        assert len(time) == 4
        assert time.to_list() == time_points
        assert time.name == "discrete_time"

    def test_from_list_empty(self):
        with pytest.raises(ValueError):
            _ = sa.Time.from_list([], name="empty_time")

    def test_from_continuous_params(self):
        time = sa.Time.from_continuous_params(
            initial_time=0.0, time_horizon=2.0, dt=0.5
        )

        assert len(time) == 5
        assert time.to_list() == [0.0, 0.5, 1.0, 1.5, 2.0]
        assert time.name == "continuous_time"

    def test_from_continuous_params_invalid_horizon(self):
        with pytest.raises(ValueError):
            _ = sa.Time.from_continuous_params(
                initial_time=2.0, time_horizon=1.0, dt=0.5
            )

    def test_from_continuous_params_nonnumeric_initial_time(self):
        with pytest.raises(ValueError):
            _ = sa.Time.from_continuous_params(
                initial_time="start", time_horizon=2.0, dt=0.5
            )

    def test_from_continuous_params_invalid_dt(self):
        with pytest.raises(ValueError):
            _ = sa.Time.from_continuous_params(
                initial_time=0.0, time_horizon=2.0, dt=-0.5
            )

    def test_from_discrete_params(self):
        time = sa.Time.from_discrete_params(initial_time=5, trajectory_length=4)

        assert len(time) == 4
        assert time.to_list() == [5, 6, 7, 8]
        assert time.name == "discrete_time"

    def test_from_discrete_params_nonnumeric_initial_time(self):
        with pytest.raises(ValueError):
            _ = sa.Time.from_discrete_params(initial_time=2.5, trajectory_length=4)

    def test_from_discrete_params_invalid_length(self):
        with pytest.raises(ValueError):
            _ = sa.Time.from_discrete_params(initial_time=5, trajectory_length=0)
