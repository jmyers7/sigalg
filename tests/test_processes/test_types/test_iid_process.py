import pandas as pd
import pytest
from scipy.stats import bernoulli, binom, norm, poisson, randint

import sigalg as sa


class TestConstructor:

    def test_basic_construction_with_bernoulli(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time, name="X")
        assert X.rv == rv
        assert X.max_trajectories == 1000
        assert X.length == 10
        assert X.initial_time == 0
        assert X.name == "X"

    def test_construction_with_all_parameters(self):
        rv = norm(loc=0, scale=1)
        time = sa.Time.discrete(start=5, length=20)
        Y = sa.IIDProcess(
            rv=rv,
            time=time,
            max_trajectories=500,
            name="Y",
            random_state=42,
        )
        assert Y.rv == rv
        assert Y.max_trajectories == 500
        assert Y.length == 20
        assert Y.initial_time == 5
        assert Y.name == "Y"

    def test_construction_generates_trajectories(self):
        rv = bernoulli(0.3)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time, max_trajectories=10)
        assert X.trajectories is not None
        assert isinstance(X.trajectories, sa.Trajectories)

    def test_construction_with_random_state_reproducible(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        process1 = sa.IIDProcess(
            rv=rv, time=time, random_state=123, max_trajectories=100
        )
        process2 = sa.IIDProcess(
            rv=rv, time=time, random_state=123, max_trajectories=100
        )
        pd.testing.assert_frame_equal(
            process1.trajectories.values, process2.trajectories.values
        )


class TestProperties:

    @pytest.fixture
    def Z(self):
        rv = bernoulli(0.6)
        time = sa.Time.discrete(start=2, length=8)
        return sa.IIDProcess(
            rv=rv,
            time=time,
            max_trajectories=50,
            name="Z",
            random_state=99,
        )

    def test_rv_property(self, Z):
        assert isinstance(Z.rv, type(bernoulli(0.5)))

    def test_max_trajectories_property(self, Z):
        assert Z.max_trajectories == 50

    def test_initial_time_property(self, Z):
        assert Z.initial_time == 2

    def test_name_property(self, Z):
        assert Z.name == "Z"

    def test_n_trajectories_property(self, Z):
        assert Z.n_trajectories > 0
        assert Z.n_trajectories <= Z.max_trajectories

    def test_probability_measure_property(self, Z):
        assert Z.probability_measure is not None
        assert isinstance(Z.probability_measure, sa.ProbabilityMeasure)


class TestSetters:

    def test_set_name(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time, name="X")
        X.name = "NewName"
        assert X.name == "NewName"


class TestValidation:

    def test_set_name_invalid_type(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time, name="X")
        with pytest.raises(TypeError, match="name must be a string"):
            X.name = 123

    def test_invalid_rv_type(self):
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(TypeError, match="must be an instance of.*rv_frozen"):
            sa.IIDProcess(rv="not a distribution", time=time, name="X")

    def test_invalid_time_type(self):
        rv = bernoulli(0.5)
        with pytest.raises(TypeError, match="time must be a Time object."):
            sa.IIDProcess(rv=rv, time="not a time", name="X")

    def test_invalid_max_trajectories_not_int(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, time=time, max_trajectories=10.5, name="X")

    def test_invalid_max_trajectories_negative(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, time=time, max_trajectories=-5, name="X")

    def test_invalid_max_trajectories_zero(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, time=time, max_trajectories=0, name="X")

    def test_invalid_name_not_string(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(TypeError, match="name must be a string"):
            sa.IIDProcess(rv=rv, time=time, name=123)

    def test_invalid_random_state_not_int(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            TypeError, match="random_state must be a non-negative integer or None."
        ):
            sa.IIDProcess(rv=rv, time=time, random_state=12.5, name="X")

    def test_invalid_random_state_negative(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            TypeError, match="random_state must be a non-negative integer or None."
        ):
            sa.IIDProcess(rv=rv, time=time, random_state=-1, name="X")


class TestSimulation:

    def test_simulation_produces_correct_shape(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=15)
        X = sa.IIDProcess(rv=rv, time=time, max_trajectories=100)
        assert X.trajectories.values.shape[1] == 15

    def test_simulation_with_initial_time(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=5, length=10)
        X = sa.IIDProcess(rv=rv, time=time)
        time = X.time
        assert min(time) == 5
        assert max(time) == 14
        assert len(time) == 10

    def test_simulation_bernoulli_values_in_range(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time, random_state=42)
        values = X.trajectories.values.values.flatten()
        assert all(v in [0, 1] for v in values)


class TestTrajectories:

    @pytest.fixture
    def X(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=8)
        return sa.IIDProcess(rv=rv, time=time, max_trajectories=50, random_state=123)

    def test_trajectories_type(self, X):
        assert isinstance(X.trajectories, sa.Trajectories)

    def test_trajectories_has_sample_space(self, X):
        assert hasattr(X.trajectories, "sample_space")
        assert isinstance(X.trajectories.sample_space, sa.SampleSpace)

    def test_trajectory_at_indexer(self, X):
        trajectory = X.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)
        assert len(trajectory) == X.length

    def test_rv_at_indexer(self, X):
        rv = X.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)
        assert rv.name == "X0"

    def test_rv_at_different_times(self, X):
        rv0 = X.rv_at[0]
        rv1 = X.rv_at[1]
        assert rv0.name == "X0"
        assert rv1.name == "X1"


class TestPlotting:

    @pytest.fixture
    def X(self):
        rv = bernoulli(0.7)
        time = sa.Time.discrete(start=0, length=10)
        return sa.IIDProcess(rv=rv, time=time, max_trajectories=20, random_state=42)

    def test_plot_trajectories_creates_plot(self, X):
        ax = X.plot_trajectories()
        assert ax is not None

    def test_plot_trajectories_with_custom_labels(self, X):
        ax = X.plot_trajectories(x_label="Custom Time", y_label="Custom State")
        assert ax.get_xlabel() == "Custom Time"
        assert ax.get_ylabel() == "Custom State"

    def test_plot_trajectories_with_colors(self, X):
        ax = X.plot_trajectories(colors=["red"])
        assert ax is not None

    def test_plot_trajectories_with_title(self, X):
        ax = X.plot_trajectories(title="Custom Title")
        assert ax.get_title() == "Custom Title"


class TestTrajectory:

    @pytest.fixture
    def X(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=3, length=12)
        return sa.IIDProcess(rv=rv, time=time, max_trajectories=30, random_state=42)

    def test_trajectory_at_returns_trajectory(self, X):
        trajectory = X.trajectory_at[0]
        assert isinstance(trajectory, sa.Trajectory)

    def test_trajectory_has_correct_length(self, X):
        trajectory = X.trajectory_at[0]
        assert len(trajectory) == X.length

    def test_trajectory_has_correct_time(self, X):
        trajectory = X.trajectory_at[0]
        expected_times = list(range(3, 15))
        assert list(trajectory.values.index) == expected_times

    def test_trajectory_value_at_accessor(self, X):
        trajectory = X.trajectory_at[0]
        value = trajectory.value_at[3]
        assert value in [0, 1]

    def test_multiple_trajectories_are_different(self, X):
        traj1 = X.trajectory_at[0]
        traj2 = X.trajectory_at[1]
        are_different = not all(traj1.values == traj2.values)
        assert are_different or X.n_trajectories == 1


class TestRandomVariable:

    @pytest.fixture
    def X(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        return sa.IIDProcess(rv=rv, time=time, max_trajectories=100, random_state=42)

    def test_rv_at_returns_random_variable(self, X):
        rv = X.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)

    def test_rv_at_has_correct_name(self, X):
        rv = X.rv_at[5]
        assert rv.name == "X5"

    def test_rv_at_has_probability_space(self, X):
        rv = X.rv_at[0]
        assert rv.probability_space is not None

    def test_rv_at_different_times_independent(self, X):
        rv0 = X.rv_at[0]
        rv1 = X.rv_at[1]
        assert rv0.name != rv1.name

    def test_rv_at_invalid_time_raises_error(self, X):
        with pytest.raises(ValueError, match="not in process time index"):
            X.rv_at[100]


class TestEnumerationConstructor:

    def test_construction_with_enumerate_true(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=4)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        assert X.enumerate is True
        assert X.n_trajectories == 16

    def test_construction_enumerate_with_binom(self):
        rv = binom(n=2, p=0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1, 2], enumerate=True)
        assert X.enumerate is True
        assert X.n_trajectories == 27


class TestEnumerationValidation:

    def test_enumerate_without_support_raises_error(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        with pytest.raises(
            ValueError, match="Cannot enumerate trajectories without explicit support"
        ):
            sa.IIDProcess(rv=rv, time=time, enumerate=True)

    def test_enumerate_invalid_type(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        with pytest.raises(TypeError, match="enumerate must be a boolean"):
            sa.IIDProcess(rv=rv, time=time, enumerate="yes")

    def test_enumerate_large_trajectory_raises(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=25)
        with pytest.raises(ValueError, match="The number of"):
            sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)


class TestEnumerationExactProbabilities:

    def test_exact_probabilities_bernoulli_fair(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        probs = list(X.probability_measure.values)
        expected_prob = 0.125
        for prob in probs:
            assert abs(prob - expected_prob) < 1e-10

    def test_exact_probabilities_bernoulli_biased(self):
        rv = bernoulli(0.3)
        time = sa.Time.discrete(start=0, length=2)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        probs = X.probability_measure.values
        expected_probs = [
            0.3 * 0.3,
            0.3 * 0.7,
            0.7 * 0.3,
            0.7 * 0.7,
        ]
        sorted_probs = sorted(probs)
        sorted_expected = sorted(expected_probs)
        for prob, expected in zip(sorted_probs, sorted_expected):
            assert abs(prob - expected) < 1e-10

    def test_probabilities_sum_to_one(self):
        rv = bernoulli(0.4)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        total_prob = sum(X.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_exact_probabilities_binom(self):
        rv = binom(n=2, p=0.5)
        time = sa.Time.discrete(start=0, length=2)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1, 2], enumerate=True)
        total_prob = sum(X.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10


class TestEnumerationTrajectories:

    def test_all_trajectories_enumerated_bernoulli(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        trajectories = X.trajectories
        assert len(trajectories) == 8
        unique_trajectories = {tuple(traj) for traj in trajectories.iter_trajectories()}
        assert len(unique_trajectories) == 8

    def test_trajectories_contain_only_valid_values_bernoulli(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        values = X.trajectories.values.values.flatten()
        assert all(v in [0, 1] for v in values)

    def test_trajectories_contain_only_valid_values_binom(self):
        rv = binom(n=2, p=0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1, 2], enumerate=True)
        values = X.trajectories.values.values.flatten()
        assert all(v in [0, 1, 2] for v in values)

    def test_enumeration_with_initial_time(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=5, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        time = X.time
        assert list(time) == [5, 6, 7]
        assert len(X.trajectories) == 8


class TestEnumerationComparison:

    def test_simulation_vs_enumeration_different_mode(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=5)
        sim_process = sa.IIDProcess(rv=rv1, time=time, enumerate=False, random_state=42)
        enum_process = sa.IIDProcess(rv=rv2, time=time, support=[0, 1], enumerate=True)
        assert sim_process.enumerate is False
        assert enum_process.enumerate is True
        assert sim_process != enum_process

    def test_equal_enumerated_processes(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=5)
        process1 = sa.IIDProcess(rv=rv1, time=time, support=[0, 1], enumerate=True)
        process2 = sa.IIDProcess(rv=rv2, time=time, support=[0, 1], enumerate=True)
        assert process1 == process2

    def test_enumeration_produces_deterministic_order(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        process1 = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        process2 = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        pd.testing.assert_frame_equal(
            process1.trajectories.values, process2.trajectories.values
        )


class TestEnumerationPlotting:

    def test_plot_title_with_enumeration(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        title = X._plot_title()
        assert title == "Enumerated IID Bernoulli Process X"

    def test_plot_trajectories_with_enumeration(self):
        rv = bernoulli(0.7)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        ax = X.plot_trajectories()
        assert ax is not None


class TestEnumerationDiscreteSupport:

    def test_bernoulli_support(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        values = set(X.trajectories.values.values.flatten())
        assert values == {0, 1}

    def test_binom_support(self):
        rv = binom(n=3, p=0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1, 2, 3], enumerate=True)
        values = set(X.trajectories.values.values.flatten())
        assert values.issubset({0, 1, 2, 3})

    def test_poisson_support(self):
        rv = poisson(mu=2)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=list(range(5)), enumerate=True)
        values = X.trajectories.values.values.flatten()
        assert all(v >= 0 for v in values)
        assert all(isinstance(int(v), int) for v in values)

    def test_randint_support(self):
        rv = randint(low=0, high=3)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1, 2], enumerate=True)
        values = set(X.trajectories.values.values.flatten())
        assert values.issubset({0, 1, 2})


class TestEnumerationRandomVariable:

    @pytest.fixture
    def enum_process(self):
        rv = bernoulli(0.6)
        time = sa.Time.discrete(start=0, length=3)
        return sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)

    def test_rv_at_with_enumeration(self, enum_process):
        rv = enum_process.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)
        assert rv.probability_space is not None

    def test_rv_at_has_exact_probabilities(self, enum_process):
        rv = enum_process.rv_at[0]
        prob_0 = rv.P(0)
        prob_1 = rv.P(1)
        assert abs(prob_0 + prob_1 - 1.0) < 1e-10
