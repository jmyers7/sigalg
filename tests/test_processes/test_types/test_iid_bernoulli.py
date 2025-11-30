import pandas as pd
import pytest

import sigalg as sa


class TestConstruction:

    def test_construction(self):

        X = sa.IIDBernoulli(
            probability=0.25, max_trajectories=1, length=40, initial_time=3
        )
        assert X.probability == 0.25
        assert X.length == 40
        assert X.initial_time == 3


class TestReproducibility:

    def test_random_state(self):
        X = sa.IIDBernoulli(
            probability=0.3,
            max_trajectories=10,
            length=10,
            initial_time=2,
            name="X",
            random_state=42,
        )

        expected_trajectories = [
            [1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
        expected_index = [f"omega{i}" for i in range(10)]
        expected_trajectories = pd.DataFrame(
            expected_trajectories,
            columns=list(range(2, 12)),
            index=expected_index,
        )
        expected_trajectories.index.name = "trajectory"
        expected_trajectories.columns.name = "time"
        pd.testing.assert_frame_equal(
            X.process_trajectories.values, expected_trajectories
        )


class TestDataAccess:

    @pytest.fixture
    def process(self):
        return sa.IIDBernoulli(
            probability=0.3,
            max_trajectories=10,
            length=10,
            initial_time=1,
            name="Z",
            random_state=42,
        )

    def test_rv_at(self, process):
        Z1 = process.rv_at[1]
        assert isinstance(Z1, sa.RandomVariable)
        assert Z1.name == "Z1"
        assert Z1("omega0") == 1
        assert Z1("omega9") == 0

    def test_trajectory_at(self, process):
        expected_trajectory = pd.Series(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            name="omega9",
            index=pd.RangeIndex(start=1, stop=11, name="time"),
        )
        traj = process.trajectory_at[9]
        assert isinstance(traj, sa.Trajectory)
        pd.testing.assert_series_equal(traj.values, expected_trajectory)
        assert traj.value_at[1] == 0
        assert traj.value_at[10] == 1
