import numpy as np
import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:

    def test_basic_construction_with_numpy_array(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        assert mc.n_support == 2
        assert len(mc) == 10
        assert mc.initial_time == 0
        assert mc.name == "X"
        assert mc.max_trajectories == 1000
        assert mc.enumerate is False

    def test_construction_with_dataframe(self):
        P = pd.DataFrame([[0.7, 0.3], [0.4, 0.6]], index=[0, 1], columns=[0, 1])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        assert mc.n_support == 2
        pd.testing.assert_frame_equal(mc.transition_matrix, P)

    def test_construction_with_all_parameters(self):
        P = pd.DataFrame(
            [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        pi = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        time = sa.Time.discrete(start=5, length=15)
        mc = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            initial_distribution=pi,
            support=["A", "B", "C"],
            name="Y",
            max_trajectories=500,
            random_state=42,
            enumerate=False,
        )
        assert len(mc) == 15
        assert mc.initial_time == 5
        assert mc.name == "Y"
        assert mc.max_trajectories == 500
        assert mc.enumerate is False
        assert mc.support == ["A", "B", "C"]

    def test_construction_with_string_states(self):
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]], index=["Rain", "Sun"], columns=["Rain", "Sun"]
        )
        pi = pd.Series([0.6, 0.4], index=["Rain", "Sun"])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            initial_distribution=pi,
            support=["Rain", "Sun"],
        )
        assert mc.support == ["Rain", "Sun"]
        assert mc.initial_distribution["Rain"] == 0.6
        assert mc.initial_distribution["Sun"] == 0.4

    def test_construction_generates_trajectories(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, max_trajectories=10)
        assert mc.trajectories is not None
        assert isinstance(mc.trajectories, sa.Trajectories)

    def test_construction_with_enumerate_true(self):
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        assert mc.enumerate is True
        assert mc.n_trajectories == 8

    def test_construction_with_initial_distribution_series(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = pd.Series([0.6, 0.4], index=[0, 1])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time, initial_distribution=pi)
        assert np.allclose(mc.initial_distribution.values, pi.values)
        assert list(mc.initial_distribution.index) == list(pi.index)

    def test_construction_with_initial_distribution_dict(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = {0: 0.3, 1: 0.7}
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time, initial_distribution=pi)
        assert mc.initial_distribution[0] == 0.3
        assert mc.initial_distribution[1] == 0.7

    def test_construction_with_none_initial_distribution(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time, initial_distribution=None)
        assert abs(mc.initial_distribution[0] - 0.5) < 1e-10
        assert abs(mc.initial_distribution[1] - 0.5) < 1e-10

    def test_construction_reproducible_with_random_state(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc1 = sa.MarkovChain(
            transition_matrix=P, time=time, random_state=123, max_trajectories=50
        )
        time = sa.Time.discrete(start=0, length=10)
        mc2 = sa.MarkovChain(
            transition_matrix=P, time=time, random_state=123, max_trajectories=50
        )
        pd.testing.assert_frame_equal(mc1.trajectories.values, mc2.trajectories.values)


class TestProperties:

    @pytest.fixture
    def mc(self):
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
        pi = np.array([0.5, 0.3, 0.2])
        time = sa.Time.discrete(start=3, length=12)
        return sa.MarkovChain(
            transition_matrix=P,
            time=time,
            initial_distribution=pi,
            support=["A", "B", "C"],
            name="Z",
            max_trajectories=75,
            random_state=99,
        )

    def test_transition_matrix_property(self, mc):
        assert mc.transition_matrix.shape == (3, 3)
        assert list(mc.transition_matrix.index) == ["A", "B", "C"]

    def test_initial_distribution_property(self, mc):
        assert len(mc.initial_distribution) == 3
        assert abs(mc.initial_distribution.sum() - 1.0) < 1e-10

    def test_time_property(self, mc):
        assert isinstance(mc.time, sa.Time)

    def test_support_property(self, mc):
        assert mc.support == ["A", "B", "C"]

    def test_n_support_property(self, mc):
        assert mc.n_support == 3

    def test_initial_time_property(self, mc):
        assert mc.initial_time == 3

    def test_name_property(self, mc):
        assert mc.name == "Z"

    def test_max_trajectories_property(self, mc):
        assert mc.max_trajectories == 75

    def test_enumerate_property(self, mc):
        assert mc.enumerate is False

    def test_n_trajectories_property(self, mc):
        assert mc.n_trajectories > 0
        assert mc.n_trajectories <= mc.max_trajectories

    def test_time_index_property(self, mc):
        expected_time_index = list(range(3, 15))
        assert list(mc.time) == expected_time_index

    def test_probability_measure_property(self, mc):
        assert mc.probability_measure is not None
        assert isinstance(mc.probability_measure, sa.ProbabilityMeasure)


class TestSetters:

    def test_set_name(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time, name="X")
        mc.name = "NewName"
        assert mc.name == "NewName"


class TestValidation:

    def test_set_name_invalid_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time, name="X")
        with pytest.raises(TypeError, match="name must be a string"):
            mc.name = 123

    def test_invalid_transition_matrix_type(self):
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            TypeError, match="transition_matrix must be a numpy array or pandas"
        ):
            sa.MarkovChain(transition_matrix="not a matrix", time=time)

    def test_invalid_transition_matrix_not_2d(self):
        with pytest.raises(ValueError, match="transition_matrix must be a 2D array"):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=np.array([0.5, 0.5]), time=time)

    def test_invalid_transition_matrix_not_square(self):
        P = pd.DataFrame([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="transition_matrix must be square"):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time)

    def test_invalid_transition_matrix_rows_not_sum_to_one(self):
        P = np.array([[0.5, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="Each row of transition_matrix must sum"):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time)

    def test_invalid_transition_matrix_negative_entries(self):
        P = np.array([[1.2, -0.2], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="All entries in transition_matrix must be non-negative"
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time)

    def test_invalid_initial_distribution_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            TypeError,
            match="initial_distribution must be a numpy array, pandas Series, dict",
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(
                transition_matrix=P, time=time, initial_distribution="invalid"
            )

    def test_invalid_initial_distribution_not_sum_to_one(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = np.array([0.3, 0.5])
        with pytest.raises(ValueError, match="initial_distribution must sum to 1"):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, initial_distribution=pi)

    def test_invalid_initial_distribution_negative_entries(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = np.array([-0.2, 1.2])
        with pytest.raises(
            ValueError,
            match="All entries in initial_distribution must be non-negative",
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, initial_distribution=pi)

    def test_invalid_initial_distribution_wrong_length(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = np.array([0.5, 0.3, 0.2])
        with pytest.raises(
            ValueError, match="Length of initial_distribution .* must match"
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, initial_distribution=pi)

    def test_invalid_initial_distribution_missing_states(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = {"X": 0.6, "Y": 0.4}
        with pytest.raises(ValueError, match="initial_distribution missing states"):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(
                transition_matrix=P, time=time, initial_distribution=pi, support=[0, 1]
            )

    def test_invalid_support_wrong_length(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="Length of support .* must match"):
            time = sa.Time.discrete(start=0, length=5)
            sa.MarkovChain(transition_matrix=P, time=time, support=[0, 1, 2])

    def test_invalid_time_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(TypeError, match="time must be a Time object."):
            sa.MarkovChain(transition_matrix=P, time="not a time")

    def test_invalid_name_not_string(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(TypeError, match="name must be a string"):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, name=123)

    def test_invalid_max_trajectories_not_int(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, max_trajectories=10.5)

    def test_invalid_max_trajectories_negative(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, max_trajectories=-5)

    def test_invalid_max_trajectories_zero(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, max_trajectories=0)

    def test_invalid_random_state_not_int(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="random_state must be a non-negative integer or None"
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, random_state=12.5)

    def test_invalid_random_state_negative(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="random_state must be a non-negative integer or None"
        ):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, random_state=-1)

    def test_invalid_enumerate_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(TypeError, match="enumerate must be a boolean"):
            time = sa.Time.discrete(start=0, length=15)
            sa.MarkovChain(transition_matrix=P, time=time, enumerate="yes")


class TestSimulation:

    def test_simulation_produces_correct_shape(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=15)
        mc = sa.MarkovChain(
            transition_matrix=P, time=time, max_trajectories=100, random_state=42
        )
        assert mc.trajectories.shape[0] <= 100
        assert mc.trajectories.shape[1] == 15

    def test_simulation_with_initial_time(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=5, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        time = mc.time
        assert min(time) == 5
        assert max(time) == 14
        assert len(time) == 10

    def test_simulation_values_in_state_space(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(
            transition_matrix=P, time=time, support=["A", "B"], random_state=42
        )
        values = mc.trajectories.values.values.flatten()
        assert all(v in ["A", "B"] for v in values)


class TestTrajectories:

    @pytest.fixture
    def mc(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        return sa.MarkovChain(
            transition_matrix=P, time=time, max_trajectories=50, random_state=123
        )

    def test_trajectories_type(self, mc):
        assert isinstance(mc.trajectories, sa.Trajectories)

    def test_trajectories_has_sample_space(self, mc):
        assert hasattr(mc.trajectories, "sample_space")
        assert isinstance(mc.trajectories.sample_space, sa.SampleSpace)

    def test_trajectory_at_indexer(self, mc):
        trajectory = mc.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)
        assert len(trajectory) == len(mc)

    def test_rv_at_indexer(self, mc):
        rv = mc.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)
        assert rv.name == "X0"

    def test_rv_at_different_times(self, mc):
        rv0 = mc.rv_at[0]
        rv1 = mc.rv_at[1]
        assert rv0.name == "X0"
        assert rv1.name == "X1"


class TestEquality:

    def test_equal_markov_chains(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc1 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            random_state=42,
            max_trajectories=50,
            name="X",
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            random_state=42,
            max_trajectories=50,
            name="X",
        )
        assert mc1 == mc2

    def test_not_equal_different_name(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc1 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            random_state=42,
            max_trajectories=50,
            name="X",
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            random_state=42,
            max_trajectories=50,
            name="Y",
        )
        assert mc1 != mc2

    def test_not_equal_different_length(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=15)
        mc1 = sa.MarkovChain(transition_matrix=P, time=time, random_state=42, name="X")
        time = sa.Time.discrete(start=5, length=15)
        mc2 = sa.MarkovChain(transition_matrix=P, time=time, random_state=42, name="X")
        assert mc1 != mc2

    def test_not_equal_different_initial_time(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time1 = sa.Time.discrete(start=0, length=10)
        mc1 = sa.MarkovChain(transition_matrix=P, time=time1, random_state=42, name="X")
        time2 = sa.Time.discrete(start=5, length=10)
        mc2 = sa.MarkovChain(transition_matrix=P, time=time2, random_state=42, name="X")
        assert mc1 != mc2

    def test_not_equal_different_random_state(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc1 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            random_state=42,
            max_trajectories=100,
            name="X",
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            random_state=99,
            max_trajectories=100,
            name="X",
        )
        assert mc1 != mc2

    def test_not_equal_different_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time, name="X")
        assert mc != "not a markov chain"
        assert mc != 42
        assert mc is not None


class TestPlotting:

    @pytest.fixture
    def markov_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        return sa.MarkovChain(
            transition_matrix=P, time=time, max_trajectories=20, random_state=42
        )

    def test_plot_title_without_enumeration(self, markov_chain):
        title = markov_chain._plot_title()
        assert title == "Simulated Markov Chain"

    def test_plot_title_with_enumeration(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        title = mc._plot_title()
        assert title == "Enumerated Markov Chain"

    def test_plot_trajectories_creates_plot(self, markov_chain):
        ax = markov_chain.plot_trajectories()
        assert ax is not None

    def test_plot_trajectories_with_custom_labels(self, markov_chain):
        ax = markov_chain.plot_trajectories(
            x_label="Custom Time", y_label="Custom State"
        )
        assert ax.get_xlabel() == "Custom Time"
        assert ax.get_ylabel() == "Custom State"

    def test_plot_trajectories_with_colors(self, markov_chain):
        ax = markov_chain.plot_trajectories(colors=["red"])
        assert ax is not None

    def test_plot_trajectories_with_title(self, markov_chain):
        ax = markov_chain.plot_trajectories(title="Custom Title")
        assert ax.get_title() == "Custom Title"


class TestTrajectory:

    @pytest.fixture
    def mc(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=3, length=12)
        return sa.MarkovChain(
            transition_matrix=P,
            time=time,
            max_trajectories=30,
            random_state=42,
        )

    def test_trajectory_at_returns_trajectory(self, mc):
        trajectory = mc.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)

    def test_trajectory_has_correct_length(self, mc):
        trajectory = mc.trajectory_at[0]
        assert len(trajectory) == len(mc)

    def test_trajectory_has_correct_time_index(self, mc):
        trajectory = mc.trajectory_at[0]
        expected_times = list(range(3, 15))
        assert list(trajectory.values.index) == expected_times

    def test_trajectory_value_at_accessor(self, mc):
        trajectory = mc.trajectory_at[0]
        value = trajectory.value_at[3]
        assert value in [0, 1]

    def test_multiple_trajectories_are_different(self, mc):
        traj1 = mc.trajectory_at[0]
        traj2 = mc.trajectory_at[1]
        are_different = not all(traj1.values == traj2.values)
        assert are_different or mc.n_trajectories == 1


class TestRandomVariable:

    @pytest.fixture
    def mc(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        return sa.MarkovChain(
            transition_matrix=P, time=time, max_trajectories=100, random_state=42
        )

    def test_rv_at_returns_random_variable(self, mc):
        rv = mc.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)

    def test_rv_at_has_correct_name(self, mc):
        rv = mc.rv_at[5]
        assert rv.name == "X5"

    def test_rv_at_has_probability_space(self, mc):
        rv = mc.rv_at[0]
        assert rv.probability_space is not None

    def test_rv_at_invalid_time_raises_error(self, mc):
        with pytest.raises(ValueError, match="not in process time index"):
            mc.rv_at[100]


class TestEnumerationConstructor:

    def test_construction_with_enumerate_true(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        assert mc.enumerate is True
        assert mc.n_trajectories == 8

    def test_construction_enumerate_with_three_states(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        assert mc.enumerate is True
        assert mc.n_trajectories == 27

    def test_construction_enumerate_with_max_trajectories_limit(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(ValueError, match="greater than max_trajectories"):
            _ = sa.MarkovChain(
                transition_matrix=P, time=time, max_trajectories=50, enumerate=True
            )

    def test_construction_enumerate_reproducible(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=3)
        mc1 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            enumerate=True,
            random_state=42,
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P,
            time=time,
            enumerate=True,
            random_state=42,
        )
        pd.testing.assert_frame_equal(mc1.trajectories.values, mc2.trajectories.values)


class TestEnumerationValidation:

    def test_enumerate_large_trajectory_count_raises(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        with pytest.raises(ValueError, match="greater than max_trajectories"):
            time = sa.Time.discrete(start=0, length=10)
            sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)

    def test_enumerate_exceeds_max_trajectories_raises(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=15)
        with pytest.raises(ValueError, match="greater than max_trajectories"):
            sa.MarkovChain(
                transition_matrix=P, time=time, max_trajectories=100, enumerate=True
            )


class TestEnumerationExactProbabilities:

    def test_exact_probabilities_symmetric_chain(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        probs = list(mc.probability_measure.values)
        expected_prob = 0.125
        for prob in probs:
            assert abs(prob - expected_prob) < 1e-10

    def test_probabilities_sum_to_one(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=2)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        total_prob = sum(mc.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_exact_probabilities_three_states(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        total_prob = sum(mc.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10


class TestEnumerationTrajectories:

    def test_all_trajectories_enumerated_two_states(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=4)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        trajectories = mc.trajectories
        assert len(trajectories) == 16
        unique_trajectories = {tuple(traj) for traj in trajectories.iter_trajectories()}
        assert len(unique_trajectories) == 16

    def test_trajectories_contain_only_valid_states(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=3)
        mc = sa.MarkovChain(
            transition_matrix=P, time=time, support=["A", "B"], enumerate=True
        )
        values = mc.trajectories.values.values.flatten()
        assert all(v in ["A", "B"] for v in values)

    def test_enumeration_with_initial_time(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=5, length=3)
        mc = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        time = mc.time
        assert list(time) == [5, 6, 7]
        assert len(mc.trajectories.values) == 8


class TestEnumerationComparison:

    def test_simulation_vs_enumeration_different_mode(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=4)
        sim_chain = sa.MarkovChain(
            transition_matrix=P, time=time, enumerate=False, random_state=42
        )
        time = sa.Time.discrete(start=0, length=4)
        enum_chain = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        assert sim_chain.enumerate is False
        assert enum_chain.enumerate is True
        assert sim_chain != enum_chain

    def test_equal_enumerated_chains(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=3)
        mc1 = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        time = sa.Time.discrete(start=0, length=3)
        mc2 = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        assert mc1 == mc2

    def test_enumeration_produces_deterministic_order(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=3)
        mc1 = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        time = sa.Time.discrete(start=0, length=3)
        mc2 = sa.MarkovChain(transition_matrix=P, time=time, enumerate=True)
        pd.testing.assert_frame_equal(mc1.trajectories.values, mc2.trajectories.values)


class TestStationary:

    def test_stationary_distribution_symmetric_chain(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        pi = mc.stationary_distribution
        assert abs(pi[0] - 0.5) < 1e-6
        assert abs(pi[1] - 0.5) < 1e-6

    def test_stationary_distribution_sums_to_one(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        pi = mc.stationary_distribution
        assert abs(pi.sum() - 1.0) < 1e-6

    def test_stationary_distribution_three_states(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        pi = mc.stationary_distribution
        assert abs(pi.sum() - 1.0) < 1e-6
        assert len(pi) == 3


class TestIrreducibility:

    def test_irreducible_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        assert bool(mc.is_irreducible) is True

    def test_reducible_chain(self):
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        assert bool(mc.is_irreducible) is False

    def test_irreducible_three_state_chain(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        assert bool(mc.is_irreducible) is True


class TestAperiodicity:

    def test_aperiodic_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        assert mc.is_aperiodic is True

    def test_periodic_chain(self):
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        time = sa.Time.discrete(start=0, length=10)
        mc = sa.MarkovChain(transition_matrix=P, time=time)
        assert mc.is_aperiodic is False


class TestRandomWalkFactory:

    def test_random_walk_basic(self):
        mc = sa.MarkovChain.random_walk()
        assert mc.n_support == 3
        assert mc.support == [-1, 0, 1]
        assert len(mc) == 10

    def test_random_walk_with_probability(self):
        mc = sa.MarkovChain.random_walk(p=0.6)
        assert mc.transition_matrix.loc[0, 1] == 0.6
        assert mc.transition_matrix.loc[0, -1] == 0.4

    def test_random_walk_with_custom_states(self):
        mc = sa.MarkovChain.random_walk(support=[0, 1, 2])
        assert mc.support == [0, 1, 2]

    def test_random_walk_with_length(self):
        time = sa.Time.discrete(start=0, length=20)
        mc = sa.MarkovChain.random_walk(time=time)
        assert len(mc) == 20

    def test_random_walk_invalid_state_count(self):
        with pytest.raises(ValueError, match="Random walk requires exactly 3 states"):
            sa.MarkovChain.random_walk(support=[0, 1])
