import pandas as pd
import pytest
from scipy.stats import bernoulli, norm, poisson

import sigalg as sa


class TestConstructor:

    def test_basic_construction_with_bernoulli(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, name="X")
        assert process.rv == rv
        assert process.max_trajectories == 1000
        assert process.length == 10
        assert process.initial_time == 0
        assert process.name == "X"
        assert process.rv_type is None
        assert process.random_state is None

    def test_construction_with_all_parameters(self):
        rv = norm(loc=0, scale=1)
        process = sa.IIDProcess(
            rv=rv,
            max_trajectories=500,
            length=20,
            initial_time=5,
            name="Y",
            rv_type="Gaussian",
            random_state=42,
        )
        assert process.rv == rv
        assert process.max_trajectories == 500
        assert process.length == 20
        assert process.initial_time == 5
        assert process.name == "Y"
        assert process.rv_type == "Gaussian"
        assert process.random_state == 42

    def test_construction_with_poisson(self):
        rv = poisson(mu=3)
        process = sa.IIDProcess(rv=rv, length=15, name="N")
        assert process.rv == rv
        assert process.length == 15
        assert process.name == "N"

    def test_construction_generates_process_trajectories(self):
        rv = bernoulli(0.3)
        process = sa.IIDProcess(rv=rv, max_trajectories=10, length=5)
        assert process.process_trajectories is not None
        assert isinstance(process.process_trajectories, sa.ProcessTrajectories)

    def test_construction_with_random_state_reproducible(self):
        rv = bernoulli(0.5)
        process1 = sa.IIDProcess(rv=rv, random_state=123, max_trajectories=100)
        process2 = sa.IIDProcess(rv=rv, random_state=123, max_trajectories=100)
        pd.testing.assert_frame_equal(
            process1.process_trajectories.values, process2.process_trajectories.values
        )


class TestProperties:

    @pytest.fixture
    def process(self):
        rv = bernoulli(0.6)
        return sa.IIDProcess(
            rv=rv,
            max_trajectories=50,
            length=8,
            initial_time=2,
            name="Z",
            rv_type="Bernoulli",
            random_state=99,
        )

    def test_rv_property(self, process):
        assert isinstance(process.rv, type(bernoulli(0.5)))

    def test_max_trajectories_property(self, process):
        assert process.max_trajectories == 50

    def test_length_property(self, process):
        assert process.length == 8

    def test_initial_time_property(self, process):
        assert process.initial_time == 2

    def test_name_property(self, process):
        assert process.name == "Z"

    def test_rv_type_property(self, process):
        assert process.rv_type == "Bernoulli"

    def test_random_state_property(self, process):
        assert process.random_state == 99

    def test_n_trajectories_property(self, process):
        assert process.n_trajectories > 0
        assert process.n_trajectories <= process.max_trajectories

    def test_time_index_property(self, process):
        expected_time_index = list(range(2, 10))
        assert list(process.time_index) == expected_time_index

    def test_probability_measure_property(self, process):
        assert process.probability_measure is not None
        assert isinstance(process.probability_measure, sa.ProbabilityMeasure)


class TestSetters:

    def test_set_name(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, name="X")
        process.name = "NewName"
        assert process.name == "NewName"

    def test_set_name_invalid_type(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, name="X")
        with pytest.raises(TypeError, match="name must be a string"):
            process.name = 123

    def test_set_rv_type(self):
        rv = norm(0, 1)
        process = sa.IIDProcess(rv=rv, name="X")
        process.rv_type = "Normal"
        assert process.rv_type == "Normal"

    def test_set_rv_type_to_none(self):
        rv = norm(0, 1)
        process = sa.IIDProcess(rv=rv, name="X", rv_type="Normal")
        process.rv_type = None
        assert process.rv_type is None

    def test_set_rv_type_invalid_type(self):
        rv = norm(0, 1)
        process = sa.IIDProcess(rv=rv, name="X")
        with pytest.raises(TypeError, match="rv_type must be a string or None"):
            process.rv_type = 123


class TestValidation:

    def test_invalid_rv_type(self):
        with pytest.raises(TypeError, match="must be an instance of.*rv_frozen"):
            sa.IIDProcess(rv="not a distribution", name="X")

    def test_invalid_max_trajectories_not_int(self):
        rv = bernoulli(0.5)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, max_trajectories=10.5, name="X")

    def test_invalid_max_trajectories_negative(self):
        rv = bernoulli(0.5)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, max_trajectories=-5, name="X")

    def test_invalid_max_trajectories_zero(self):
        rv = bernoulli(0.5)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, max_trajectories=0, name="X")

    def test_invalid_length_not_int(self):
        rv = bernoulli(0.5)
        with pytest.raises(ValueError, match="length must be a positive integer"):
            sa.IIDProcess(rv=rv, length=5.5, name="X")

    def test_invalid_length_negative(self):
        rv = bernoulli(0.5)
        with pytest.raises(ValueError, match="length must be a positive integer"):
            sa.IIDProcess(rv=rv, length=-3, name="X")

    def test_invalid_length_zero(self):
        rv = bernoulli(0.5)
        with pytest.raises(ValueError, match="length must be a positive integer"):
            sa.IIDProcess(rv=rv, length=0, name="X")

    def test_invalid_initial_time_not_int(self):
        rv = bernoulli(0.5)
        with pytest.raises(TypeError, match="initial_time must be an integer"):
            sa.IIDProcess(rv=rv, initial_time=1.5, name="X")

    def test_invalid_name_not_string(self):
        rv = bernoulli(0.5)
        with pytest.raises(TypeError, match="name must be a string"):
            sa.IIDProcess(rv=rv, name=123)

    def test_invalid_random_state_not_int(self):
        rv = bernoulli(0.5)
        with pytest.raises(
            ValueError, match="random_state must be a non-negative integer or None"
        ):
            sa.IIDProcess(rv=rv, random_state=12.5, name="X")

    def test_invalid_random_state_negative(self):
        rv = bernoulli(0.5)
        with pytest.raises(
            ValueError, match="random_state must be a non-negative integer or None"
        ):
            sa.IIDProcess(rv=rv, random_state=-1, name="X")


class TestSimulation:

    def test_simulation_produces_correct_shape(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, max_trajectories=100, length=15)
        assert process.process_trajectories.values.shape[1] == 15

    def test_simulation_with_initial_time(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=10, initial_time=5)
        time_index = process.time_index
        assert min(time_index) == 5
        assert max(time_index) == 14
        assert len(time_index) == 10

    def test_simulation_bernoulli_values_in_range(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, random_state=42)
        values = process.process_trajectories.values.values.flatten()
        assert all(v in [0, 1] for v in values)

    def test_simulation_produces_unique_trajectories(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(
            rv=rv, max_trajectories=1000, length=10, random_state=42
        )
        assert process.n_trajectories > 1


class TestProcessTrajectories:

    @pytest.fixture
    def process(self):
        rv = bernoulli(0.5)
        return sa.IIDProcess(rv=rv, max_trajectories=50, length=8, random_state=123)

    def test_process_trajectories_type(self, process):
        assert isinstance(process.process_trajectories, sa.ProcessTrajectories)

    def test_process_trajectories_has_sample_space(self, process):
        assert hasattr(process.process_trajectories, "sample_space")
        assert isinstance(process.process_trajectories.sample_space, sa.SampleSpace)

    def test_process_trajectories_has_probability_measure(self, process):
        assert hasattr(process.process_trajectories, "probability_measure")
        assert isinstance(
            process.process_trajectories.probability_measure, sa.ProbabilityMeasure
        )

    def test_process_trajectories_probabilities_sum_to_one(self, process):
        total_prob = sum(process.process_trajectories.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_trajectory_at_indexer(self, process):
        trajectory = process.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)
        assert len(trajectory) == process.length

    def test_rv_at_indexer(self, process):
        rv = process.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)
        assert rv.name == "X0"

    def test_rv_at_different_times(self, process):
        rv0 = process.rv_at[0]
        rv1 = process.rv_at[1]
        assert rv0.name == "X0"
        assert rv1.name == "X1"


class TestEquality:

    def test_equal_processes(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        process1 = sa.IIDProcess(rv=rv1, random_state=42, max_trajectories=50, name="X")
        process2 = sa.IIDProcess(rv=rv2, random_state=42, max_trajectories=50, name="X")
        assert process1 == process2

    def test_not_equal_different_name(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        process1 = sa.IIDProcess(rv=rv1, random_state=42, max_trajectories=50, name="X")
        process2 = sa.IIDProcess(rv=rv2, random_state=42, max_trajectories=50, name="Y")
        assert process1 != process2

    def test_not_equal_different_length(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        process1 = sa.IIDProcess(rv=rv1, random_state=42, length=10, name="X")
        process2 = sa.IIDProcess(rv=rv2, random_state=42, length=15, name="X")
        assert process1 != process2

    def test_not_equal_different_initial_time(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        process1 = sa.IIDProcess(rv=rv1, random_state=42, initial_time=0, name="X")
        process2 = sa.IIDProcess(rv=rv2, random_state=42, initial_time=5, name="X")
        assert process1 != process2

    def test_not_equal_different_random_state(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        process1 = sa.IIDProcess(
            rv=rv1, random_state=42, max_trajectories=100, name="X"
        )
        process2 = sa.IIDProcess(
            rv=rv2, random_state=99, max_trajectories=100, name="X"
        )
        assert process1 != process2

    def test_not_equal_different_type(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, name="X")
        assert process != "not a process"
        assert process != 42
        assert process is not None


class TestPlotting:

    @pytest.fixture
    def process(self):
        rv = bernoulli(0.7)
        return sa.IIDProcess(rv=rv, max_trajectories=20, length=15, random_state=42)

    def test_plot_title_without_rv_type(self, process):
        title = process._plot_title()
        assert title == "IID Process"

    def test_plot_title_with_rv_type(self):
        rv = norm(0, 1)
        process = sa.IIDProcess(rv=rv, rv_type="Gaussian", name="X")
        title = process._plot_title()
        assert title == "IID Gaussian Process"

    def test_plot_trajectories_creates_plot(self, process):
        ax = process.plot_trajectories()
        assert ax is not None

    def test_plot_trajectories_with_custom_labels(self, process):
        ax = process.plot_trajectories(x_label="Custom Time", y_label="Custom State")
        assert ax.get_xlabel() == "Custom Time"
        assert ax.get_ylabel() == "Custom State"

    def test_plot_trajectories_with_colors(self, process):
        ax = process.plot_trajectories(colors=["red"])
        assert ax is not None

    def test_plot_trajectories_with_title(self, process):
        ax = process.plot_trajectories(title="Custom Title")
        assert ax.get_title() == "Custom Title"


class TestTrajectory:

    @pytest.fixture
    def process(self):
        rv = bernoulli(0.5)
        return sa.IIDProcess(
            rv=rv, max_trajectories=30, length=12, initial_time=3, random_state=42
        )

    def test_trajectory_at_returns_trajectory(self, process):
        trajectory = process.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)

    def test_trajectory_has_correct_length(self, process):
        trajectory = process.trajectory_at[0]
        assert len(trajectory) == process.length

    def test_trajectory_has_correct_time_index(self, process):
        trajectory = process.trajectory_at[0]
        expected_times = list(range(3, 15))
        assert list(trajectory.values.index) == expected_times

    def test_trajectory_value_at_accessor(self, process):
        trajectory = process.trajectory_at[0]
        value = trajectory.value_at[3]
        assert value in [0, 1]

    def test_multiple_trajectories_are_different(self, process):
        traj1 = process.trajectory_at[0]
        traj2 = process.trajectory_at[1]
        are_different = not all(traj1.values == traj2.values)
        assert are_different or process.n_trajectories == 1


class TestRandomVariable:

    @pytest.fixture
    def process(self):
        rv = bernoulli(0.5)
        return sa.IIDProcess(rv=rv, max_trajectories=100, length=10, random_state=42)

    def test_rv_at_returns_random_variable(self, process):
        rv = process.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)

    def test_rv_at_has_correct_name(self, process):
        rv = process.rv_at[5]
        assert rv.name == "X5"

    def test_rv_at_has_probability_space(self, process):
        rv = process.rv_at[0]
        assert rv.probability_space is not None

    def test_rv_at_different_times_independent(self, process):
        rv0 = process.rv_at[0]
        rv1 = process.rv_at[1]
        assert rv0.name != rv1.name

    def test_rv_at_invalid_time_raises_error(self, process):
        with pytest.raises(ValueError, match="not in process time index"):
            process.rv_at[100]
