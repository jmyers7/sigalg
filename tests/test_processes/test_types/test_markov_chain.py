import numpy as np
import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:

    def test_basic_construction_with_numpy_array(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P)
        assert mc.n_states == 2
        assert mc.length == 10
        assert mc.initial_time == 0
        assert mc.name == "X"
        assert mc.max_trajectories == 1000
        assert mc.enumerate is False

    def test_construction_with_dataframe(self):
        P = pd.DataFrame([[0.7, 0.3], [0.4, 0.6]], index=[0, 1], columns=[0, 1])
        mc = sa.MarkovChain(transition_matrix=P)
        assert mc.n_states == 2
        pd.testing.assert_frame_equal(mc.transition_matrix, P)

    def test_construction_with_all_parameters(self):
        P = pd.DataFrame(
            [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        pi = pd.Series([0.4, 0.3, 0.3], index=["A", "B", "C"])
        mc = sa.MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
            support=["A", "B", "C"],
            length=15,
            initial_time=5,
            name="Y",
            max_trajectories=500,
            random_state=42,
            enumerate=False,
        )
        assert mc.length == 15
        assert mc.initial_time == 5
        assert mc.name == "Y"
        assert mc.max_trajectories == 500
        assert mc.enumerate is False
        assert mc.states == ["A", "B", "C"]

    def test_construction_with_string_states(self):
        P = pd.DataFrame(
            [[0.7, 0.3], [0.4, 0.6]], index=["Rain", "Sun"], columns=["Rain", "Sun"]
        )
        pi = pd.Series([0.6, 0.4], index=["Rain", "Sun"])
        mc = sa.MarkovChain(
            transition_matrix=P, initial_distribution=pi, support=["Rain", "Sun"]
        )
        assert mc.states == ["Rain", "Sun"]
        assert mc.initial_distribution["Rain"] == 0.6
        assert mc.initial_distribution["Sun"] == 0.4

    def test_construction_generates_trajectories(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, max_trajectories=10, length=5)
        assert mc.trajectories is not None
        assert isinstance(mc.trajectories, sa.Trajectories)

    def test_construction_with_enumerate_true(self):
        P = np.array([[0.8, 0.2], [0.3, 0.7]])
        mc = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
        assert mc.enumerate is True
        print(mc.n_trajectories)
        assert mc.n_trajectories == 8

    def test_construction_with_initial_distribution_series(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = pd.Series([0.6, 0.4], index=[0, 1])
        mc = sa.MarkovChain(transition_matrix=P, initial_distribution=pi)
        assert np.allclose(mc.initial_distribution.values, pi.values)
        assert list(mc.initial_distribution.index) == list(pi.index)

    def test_construction_with_initial_distribution_dict(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = {0: 0.3, 1: 0.7}
        mc = sa.MarkovChain(transition_matrix=P, initial_distribution=pi)
        assert mc.initial_distribution[0] == 0.3
        assert mc.initial_distribution[1] == 0.7

    def test_construction_with_none_initial_distribution(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, initial_distribution=None)
        assert abs(mc.initial_distribution[0] - 0.5) < 1e-10
        assert abs(mc.initial_distribution[1] - 0.5) < 1e-10

    def test_construction_reproducible_with_random_state(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(transition_matrix=P, random_state=123, max_trajectories=50)
        mc2 = sa.MarkovChain(transition_matrix=P, random_state=123, max_trajectories=50)
        pd.testing.assert_frame_equal(mc1.trajectories.values, mc2.trajectories.values)


class TestProperties:

    @pytest.fixture
    def markov_chain(self):
        P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
        pi = np.array([0.5, 0.3, 0.2])
        return sa.MarkovChain(
            transition_matrix=P,
            initial_distribution=pi,
            support=["A", "B", "C"],
            length=12,
            initial_time=3,
            name="Z",
            max_trajectories=75,
            random_state=99,
        )

    def test_transition_matrix_property(self, markov_chain):
        assert markov_chain.transition_matrix.shape == (3, 3)
        assert list(markov_chain.transition_matrix.index) == ["A", "B", "C"]

    def test_initial_distribution_property(self, markov_chain):
        assert len(markov_chain.initial_distribution) == 3
        assert abs(markov_chain.initial_distribution.sum() - 1.0) < 1e-10

    def test_states_property(self, markov_chain):
        assert markov_chain.states == ["A", "B", "C"]

    def test_n_states_property(self, markov_chain):
        assert markov_chain.n_states == 3

    def test_length_property(self, markov_chain):
        assert markov_chain.length == 12

    def test_initial_time_property(self, markov_chain):
        assert markov_chain.initial_time == 3

    def test_name_property(self, markov_chain):
        assert markov_chain.name == "Z"

    def test_max_trajectories_property(self, markov_chain):
        assert markov_chain.max_trajectories == 75

    def test_enumerate_property(self, markov_chain):
        assert markov_chain.enumerate is False

    def test_n_possible_trajectories_property(self, markov_chain):
        assert markov_chain.n_possible_trajectories == 3**12

    def test_is_complete_enumeration_false_for_simulation(self, markov_chain):
        assert markov_chain.is_complete_enumeration is False

    def test_n_trajectories_property(self, markov_chain):
        assert markov_chain.n_trajectories > 0
        assert markov_chain.n_trajectories <= markov_chain.max_trajectories

    def test_time_index_property(self, markov_chain):
        expected_time_index = list(range(3, 15))
        assert list(markov_chain.time_index) == expected_time_index

    def test_probability_measure_property(self, markov_chain):
        assert markov_chain.probability_measure is not None
        assert isinstance(markov_chain.probability_measure, sa.ProbabilityMeasure)


class TestSetters:

    def test_set_name(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, name="X")
        mc.name = "NewName"
        assert mc.name == "NewName"

    def test_set_name_invalid_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, name="X")
        with pytest.raises(TypeError, match="name must be a string"):
            mc.name = 123


class TestValidation:

    def test_invalid_transition_matrix_type(self):
        with pytest.raises(
            TypeError, match="transition_matrix must be a numpy array or pandas"
        ):
            sa.MarkovChain(transition_matrix="not a matrix")

    def test_invalid_transition_matrix_not_2d(self):
        with pytest.raises(ValueError, match="transition_matrix must be a 2D array"):
            sa.MarkovChain(transition_matrix=np.array([0.5, 0.5]))

    def test_invalid_transition_matrix_not_square(self):
        P = pd.DataFrame([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
        with pytest.raises(ValueError, match="transition_matrix must be square"):
            sa.MarkovChain(transition_matrix=P)

    def test_invalid_transition_matrix_rows_not_sum_to_one(self):
        P = np.array([[0.5, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="Each row of transition_matrix must sum"):
            sa.MarkovChain(transition_matrix=P)

    def test_invalid_transition_matrix_negative_entries(self):
        P = np.array([[1.2, -0.2], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="All entries in transition_matrix must be non-negative"
        ):
            sa.MarkovChain(transition_matrix=P)

    def test_invalid_initial_distribution_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            TypeError,
            match="initial_distribution must be a numpy array, pandas Series, dict",
        ):
            sa.MarkovChain(transition_matrix=P, initial_distribution="invalid")

    def test_invalid_initial_distribution_not_sum_to_one(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = np.array([0.3, 0.5])
        with pytest.raises(ValueError, match="initial_distribution must sum to 1"):
            sa.MarkovChain(transition_matrix=P, initial_distribution=pi)

    def test_invalid_initial_distribution_negative_entries(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = np.array([-0.2, 1.2])
        with pytest.raises(
            ValueError,
            match="All entries in initial_distribution must be non-negative",
        ):
            sa.MarkovChain(transition_matrix=P, initial_distribution=pi)

    def test_invalid_initial_distribution_wrong_length(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = np.array([0.5, 0.3, 0.2])
        with pytest.raises(
            ValueError, match="Length of initial_distribution .* must match"
        ):
            sa.MarkovChain(transition_matrix=P, initial_distribution=pi)

    def test_invalid_initial_distribution_missing_states(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = {"X": 0.6, "Y": 0.4}
        with pytest.raises(ValueError, match="initial_distribution missing states"):
            sa.MarkovChain(transition_matrix=P, initial_distribution=pi, support=[0, 1])

    def test_invalid_support_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(TypeError, match="support must be a list or None"):
            sa.MarkovChain(transition_matrix=P, support="invalid")

    def test_invalid_support_wrong_length(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="Length of support .* must match"):
            sa.MarkovChain(transition_matrix=P, support=[0, 1, 2])

    def test_invalid_length_not_int(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="length must be a positive integer"):
            sa.MarkovChain(transition_matrix=P, length=5.5)

    def test_invalid_length_negative(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="length must be a positive integer"):
            sa.MarkovChain(transition_matrix=P, length=-3)

    def test_invalid_length_zero(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="length must be a positive integer"):
            sa.MarkovChain(transition_matrix=P, length=0)

    def test_invalid_initial_time_not_int(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(TypeError, match="initial_time must be an integer"):
            sa.MarkovChain(transition_matrix=P, initial_time=1.5)

    def test_invalid_name_not_string(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(TypeError, match="name must be a string"):
            sa.MarkovChain(transition_matrix=P, name=123)

    def test_invalid_max_trajectories_not_int(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.MarkovChain(transition_matrix=P, max_trajectories=10.5)

    def test_invalid_max_trajectories_negative(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.MarkovChain(transition_matrix=P, max_trajectories=-5)

    def test_invalid_max_trajectories_zero(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.MarkovChain(transition_matrix=P, max_trajectories=0)

    def test_invalid_random_state_not_int(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="random_state must be a non-negative integer or None"
        ):
            sa.MarkovChain(transition_matrix=P, random_state=12.5)

    def test_invalid_random_state_negative(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(
            ValueError, match="random_state must be a non-negative integer or None"
        ):
            sa.MarkovChain(transition_matrix=P, random_state=-1)

    def test_invalid_enumerate_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(TypeError, match="enumerate must be a boolean"):
            sa.MarkovChain(transition_matrix=P, enumerate="yes")


class TestSimulation:

    def test_simulation_produces_correct_shape(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(
            transition_matrix=P, max_trajectories=100, length=15, random_state=42
        )
        assert mc.trajectories.values.shape[0] <= 100
        assert mc.trajectories.values.shape[1] == 15

    def test_simulation_with_initial_time(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, length=10, initial_time=5)
        time_index = mc.time_index
        assert min(time_index) == 5
        assert max(time_index) == 14
        assert len(time_index) == 10

    def test_simulation_values_in_state_space(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, support=["A", "B"], random_state=42)
        values = mc.trajectories.values.values.flatten()
        assert all(v in ["A", "B"] for v in values)

    def test_simulation_produces_unique_trajectories(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc = sa.MarkovChain(
            transition_matrix=P, max_trajectories=1000, length=10, random_state=42
        )
        assert mc.n_trajectories > 1


class TestTrajectories:

    @pytest.fixture
    def markov_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        return sa.MarkovChain(
            transition_matrix=P, max_trajectories=50, length=8, random_state=123
        )

    def test_trajectories_type(self, markov_chain):
        assert isinstance(markov_chain.trajectories, sa.Trajectories)

    def test_trajectories_has_sample_space(self, markov_chain):
        assert hasattr(markov_chain.trajectories, "sample_space")
        assert isinstance(markov_chain.trajectories.sample_space, sa.SampleSpace)

    def test_trajectory_at_indexer(self, markov_chain):
        trajectory = markov_chain.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)
        assert len(trajectory) == markov_chain.length

    def test_rv_at_indexer(self, markov_chain):
        rv = markov_chain.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)
        assert rv.name == "X0"

    def test_rv_at_different_times(self, markov_chain):
        rv0 = markov_chain.rv_at[0]
        rv1 = markov_chain.rv_at[1]
        assert rv0.name == "X0"
        assert rv1.name == "X1"


class TestEquality:

    def test_equal_markov_chains(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(
            transition_matrix=P, random_state=42, max_trajectories=50, name="X"
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P, random_state=42, max_trajectories=50, name="X"
        )
        assert mc1 == mc2

    def test_not_equal_different_name(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(
            transition_matrix=P, random_state=42, max_trajectories=50, name="X"
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P, random_state=42, max_trajectories=50, name="Y"
        )
        assert mc1 != mc2

    def test_not_equal_different_length(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(transition_matrix=P, random_state=42, length=10, name="X")
        mc2 = sa.MarkovChain(transition_matrix=P, random_state=42, length=15, name="X")
        assert mc1 != mc2

    def test_not_equal_different_initial_time(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(
            transition_matrix=P, random_state=42, initial_time=0, name="X"
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P, random_state=42, initial_time=5, name="X"
        )
        assert mc1 != mc2

    def test_not_equal_different_random_state(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(
            transition_matrix=P, random_state=42, max_trajectories=100, name="X"
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P, random_state=99, max_trajectories=100, name="X"
        )
        assert mc1 != mc2

    def test_not_equal_different_type(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, name="X")
        assert mc != "not a markov chain"
        assert mc != 42
        assert mc is not None


class TestPlotting:

    @pytest.fixture
    def markov_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        return sa.MarkovChain(
            transition_matrix=P, max_trajectories=20, length=15, random_state=42
        )

    def test_plot_title_without_enumeration(self, markov_chain):
        title = markov_chain._plot_title()
        assert title == "Simulated Markov Chain"

    def test_plot_title_with_enumeration(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
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
    def markov_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        return sa.MarkovChain(
            transition_matrix=P,
            max_trajectories=30,
            length=12,
            initial_time=3,
            random_state=42,
        )

    def test_trajectory_at_returns_trajectory(self, markov_chain):
        trajectory = markov_chain.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)

    def test_trajectory_has_correct_length(self, markov_chain):
        trajectory = markov_chain.trajectory_at[0]
        assert len(trajectory) == markov_chain.length

    def test_trajectory_has_correct_time_index(self, markov_chain):
        trajectory = markov_chain.trajectory_at[0]
        expected_times = list(range(3, 15))
        assert list(trajectory.values.index) == expected_times

    def test_trajectory_value_at_accessor(self, markov_chain):
        trajectory = markov_chain.trajectory_at[0]
        value = trajectory.value_at[3]
        assert value in [0, 1]

    def test_multiple_trajectories_are_different(self, markov_chain):
        traj1 = markov_chain.trajectory_at[0]
        traj2 = markov_chain.trajectory_at[1]
        are_different = not all(traj1.values == traj2.values)
        assert are_different or markov_chain.n_trajectories == 1


class TestRandomVariable:

    @pytest.fixture
    def markov_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        return sa.MarkovChain(
            transition_matrix=P, max_trajectories=100, length=10, random_state=42
        )

    def test_rv_at_returns_random_variable(self, markov_chain):
        rv = markov_chain.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)

    def test_rv_at_has_correct_name(self, markov_chain):
        rv = markov_chain.rv_at[5]
        assert rv.name == "X5"

    def test_rv_at_has_probability_space(self, markov_chain):
        rv = markov_chain.rv_at[0]
        assert rv.probability_space is not None

    def test_rv_at_invalid_time_raises_error(self, markov_chain):
        with pytest.raises(ValueError, match="not in process time index"):
            markov_chain.rv_at[100]


class TestEnumerationConstructor:

    def test_construction_with_enumerate_true(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, length=4, enumerate=True)
        assert mc.enumerate is True
        assert mc.n_trajectories == 16

    def test_construction_enumerate_with_three_states(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        mc = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
        assert mc.enumerate is True
        assert mc.n_trajectories == 27

    def test_construction_enumerate_with_max_trajectories_limit(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="greater than max_trajectories"):
            _ = sa.MarkovChain(
                transition_matrix=P, length=8, max_trajectories=50, enumerate=True
            )

    def test_construction_enumerate_reproducible(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(
            transition_matrix=P,
            length=3,
            enumerate=True,
            random_state=42,
        )
        mc2 = sa.MarkovChain(
            transition_matrix=P,
            length=3,
            enumerate=True,
            random_state=42,
        )
        pd.testing.assert_frame_equal(mc1.trajectories.values, mc2.trajectories.values)


class TestEnumerationProperties:

    @pytest.fixture
    def complete_enum_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        return sa.MarkovChain(transition_matrix=P, length=4, enumerate=True, name="X")

    def test_enumerate_property(self, complete_enum_chain):
        assert complete_enum_chain.enumerate is True

    def test_n_possible_trajectories_two_states(self, complete_enum_chain):
        assert complete_enum_chain.n_possible_trajectories == 16

    def test_n_possible_trajectories_three_states(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        mc = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
        assert mc.n_possible_trajectories == 27

    def test_is_complete_enumeration_true(self, complete_enum_chain):
        assert complete_enum_chain.is_complete_enumeration is True

    def test_is_complete_enumeration_false_for_simulation(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, length=4, enumerate=False)
        assert mc.is_complete_enumeration is False

    def test_n_trajectories_equals_possible_for_complete(self, complete_enum_chain):
        assert (
            complete_enum_chain.n_trajectories
            == complete_enum_chain.n_possible_trajectories
        )


class TestEnumerationValidation:

    def test_enumerate_large_trajectory_count_raises(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        with pytest.raises(ValueError, match="too large to enumerate"):
            sa.MarkovChain(transition_matrix=P, length=15, enumerate=True)

    def test_enumerate_exceeds_max_trajectories_raises(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="greater than max_trajectories"):
            sa.MarkovChain(
                transition_matrix=P, length=10, max_trajectories=100, enumerate=True
            )


class TestEnumerationExactProbabilities:

    def test_exact_probabilities_symmetric_chain(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
        probs = list(mc.probability_measure.values)
        expected_prob = 0.125
        for prob in probs:
            assert abs(prob - expected_prob) < 1e-10

    def test_probabilities_sum_to_one(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, length=4, enumerate=True)
        total_prob = sum(mc.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_exact_probabilities_three_states(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        mc = sa.MarkovChain(transition_matrix=P, length=2, enumerate=True)
        total_prob = sum(mc.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_partial_enumeration_probabilities_sum_to_one(self):
        # Test that probabilities sum to 1 even for complete enumeration
        # Using a length that allows complete enumeration
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(
            transition_matrix=P,
            length=3,
            enumerate=True,
            random_state=42,
        )
        total_prob = sum(mc.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10


class TestEnumerationTrajectories:

    def test_all_trajectories_enumerated_two_states(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
        trajectories = mc.trajectories.values
        assert len(trajectories) == 8
        unique_trajectories = {tuple(row) for row in trajectories.values}
        assert len(unique_trajectories) == 8

    def test_trajectories_contain_only_valid_states(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(
            transition_matrix=P, support=["A", "B"], length=4, enumerate=True
        )
        values = mc.trajectories.values.values.flatten()
        assert all(v in ["A", "B"] for v in values)

    def test_partial_enumeration_has_correct_count(self):
        # Test that enumeration produces correct count when feasible
        # Using a length that allows complete enumeration
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(
            transition_matrix=P,
            length=3,
            enumerate=True,
            random_state=42,
        )
        assert len(mc.trajectories.values) == 8  # 2^3 = 8 trajectories
        assert mc.n_trajectories == 8

    def test_enumeration_with_initial_time(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(
            transition_matrix=P, length=3, initial_time=5, enumerate=True
        )
        time_index = mc.time_index
        assert list(time_index) == [5, 6, 7]
        assert len(mc.trajectories.values) == 8


class TestEnumerationComparison:

    def test_simulation_vs_enumeration_different_mode(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        sim_chain = sa.MarkovChain(
            transition_matrix=P, length=4, enumerate=False, random_state=42
        )
        enum_chain = sa.MarkovChain(transition_matrix=P, length=4, enumerate=True)
        assert sim_chain.enumerate is False
        assert enum_chain.enumerate is True
        assert sim_chain != enum_chain

    def test_equal_enumerated_chains(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(transition_matrix=P, length=4, enumerate=True)
        mc2 = sa.MarkovChain(transition_matrix=P, length=4, enumerate=True)
        assert mc1 == mc2

    def test_enumeration_produces_deterministic_order(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc1 = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
        mc2 = sa.MarkovChain(transition_matrix=P, length=3, enumerate=True)
        pd.testing.assert_frame_equal(mc1.trajectories.values, mc2.trajectories.values)


class TestStationary:

    def test_stationary_distribution_symmetric_chain(self):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        mc = sa.MarkovChain(transition_matrix=P)
        pi = mc.stationary_distribution
        assert abs(pi[0] - 0.5) < 1e-6
        assert abs(pi[1] - 0.5) < 1e-6

    def test_stationary_distribution_sums_to_one(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P)
        pi = mc.stationary_distribution
        assert abs(pi.sum() - 1.0) < 1e-6

    def test_stationary_distribution_three_states(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        mc = sa.MarkovChain(transition_matrix=P)
        pi = mc.stationary_distribution
        assert abs(pi.sum() - 1.0) < 1e-6
        assert len(pi) == 3


class TestIrreducibility:

    def test_irreducible_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P)
        assert bool(mc.is_irreducible) is True

    def test_reducible_chain(self):
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        mc = sa.MarkovChain(transition_matrix=P)
        assert bool(mc.is_irreducible) is False

    def test_irreducible_three_state_chain(self):
        P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        mc = sa.MarkovChain(transition_matrix=P)
        assert bool(mc.is_irreducible) is True


class TestAperiodicity:

    def test_aperiodic_chain(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = sa.MarkovChain(transition_matrix=P)
        assert mc.is_aperiodic is True

    def test_periodic_chain(self):
        P = np.array([[0.0, 1.0], [1.0, 0.0]])
        mc = sa.MarkovChain(transition_matrix=P)
        assert mc.is_aperiodic is False


class TestRandomWalkFactory:

    def test_random_walk_basic(self):
        mc = sa.MarkovChain.random_walk()
        assert mc.n_states == 3
        assert mc.states == [-1, 0, 1]
        assert mc.length == 10

    def test_random_walk_with_probability(self):
        mc = sa.MarkovChain.random_walk(p=0.6)
        assert mc.transition_matrix.loc[0, 1] == 0.6
        assert mc.transition_matrix.loc[0, -1] == 0.4

    def test_random_walk_with_custom_states(self):
        mc = sa.MarkovChain.random_walk(support=[0, 1, 2])
        assert mc.states == [0, 1, 2]

    def test_random_walk_with_length(self):
        mc = sa.MarkovChain.random_walk(length=20)
        assert mc.length == 20

    def test_random_walk_invalid_state_count(self):
        with pytest.raises(ValueError, match="Random walk requires exactly 3 states"):
            sa.MarkovChain.random_walk(support=[0, 1])


class TestBirthDeathFactory:

    def test_birth_death_basic(self):
        mc = sa.MarkovChain.birth_death(birth_rate=0.3, death_rate=0.2)
        assert mc.n_states == 11
        assert mc.length == 10

    def test_birth_death_with_max_population(self):
        mc = sa.MarkovChain.birth_death(
            birth_rate=0.3, death_rate=0.2, max_population=5
        )
        assert mc.n_states == 6
        assert mc.states == [0, 1, 2, 3, 4, 5]

    def test_birth_death_transition_matrix_structure(self):
        mc = sa.MarkovChain.birth_death(
            birth_rate=0.3, death_rate=0.2, max_population=3
        )
        P = mc.transition_matrix
        assert P.loc[0, 0] == 0.7
        assert P.loc[0, 1] == 0.3
        assert P.loc[3, 2] == 0.2
        assert P.loc[3, 3] == 0.8

    def test_birth_death_initial_distribution(self):
        mc = sa.MarkovChain.birth_death(birth_rate=0.3, death_rate=0.2)
        assert mc.initial_distribution[0] == 1.0
        assert all(mc.initial_distribution[i] == 0.0 for i in range(1, 11))


class TestEhrenfestUrnFactory:

    def test_ehrenfest_urn_basic(self):
        mc = sa.MarkovChain.ehrenfest_urn()
        assert mc.n_states == 11
        assert mc.length == 10

    def test_ehrenfest_urn_with_n_balls(self):
        mc = sa.MarkovChain.ehrenfest_urn(n_balls=4)
        assert mc.n_states == 5
        assert mc.states == [0, 1, 2, 3, 4]

    def test_ehrenfest_urn_transition_matrix_structure(self):
        mc = sa.MarkovChain.ehrenfest_urn(n_balls=4)
        P = mc.transition_matrix
        assert P.loc[0, 1] == 1.0
        assert P.loc[4, 3] == 1.0
        assert P.loc[2, 1] == 0.5
        assert P.loc[2, 3] == 0.5

    def test_ehrenfest_urn_initial_distribution(self):
        mc = sa.MarkovChain.ehrenfest_urn(n_balls=4)
        assert mc.initial_distribution[2] == 1.0
        assert all(mc.initial_distribution[i] == 0.0 for i in [0, 1, 3, 4])

    def test_ehrenfest_urn_stationary_close_to_uniform(self):
        mc = sa.MarkovChain.ehrenfest_urn(n_balls=4)
        pi = mc.stationary_distribution
        expected_values = [1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16]
        for i, expected in enumerate(expected_values):
            assert abs(pi[i] - expected) < 0.1
