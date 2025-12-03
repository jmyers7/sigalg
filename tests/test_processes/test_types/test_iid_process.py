import pandas as pd
import pytest
from scipy.stats import bernoulli, binom, norm, poisson, randint

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
        assert process.random_state is None

    def test_construction_with_all_parameters(self):
        rv = norm(loc=0, scale=1)
        process = sa.IIDProcess(
            rv=rv,
            max_trajectories=500,
            length=20,
            initial_time=5,
            name="Y",
            random_state=42,
        )
        assert process.rv == rv
        assert process.max_trajectories == 500
        assert process.length == 20
        assert process.initial_time == 5
        assert process.name == "Y"
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

    def test_plot_title_uses_dist_name(self, process):
        title = process._plot_title()
        assert title == "IID Bernoulli Process"

    def test_plot_title_with_norm(self):
        rv = norm(0, 1)
        process = sa.IIDProcess(rv=rv, name="X")
        title = process._plot_title()
        assert title == "IID Norm Process"

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


class TestEnumerationConstructor:

    def test_construction_with_enumerate_true(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=4, enumerate=True)
        assert process.enumerate is True
        assert process.n_trajectories == 16

    def test_construction_enumerate_with_binom(self):
        rv = binom(n=2, p=0.5)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        assert process.enumerate is True
        assert process.n_trajectories == 27

    def test_construction_enumerate_with_max_trajectories_limit(self):
        rv = bernoulli(0.5)
        with pytest.warns(RuntimeWarning, match="exceeds max_trajectories"):
            process = sa.IIDProcess(
                rv=rv, length=6, max_trajectories=20, enumerate=True
            )
        assert process.enumerate is True
        assert process.n_trajectories == 20

    def test_construction_enumerate_reproducible(self):
        rv = bernoulli(0.5)
        with pytest.warns(RuntimeWarning, match="exceeds max_trajectories"):
            process1 = sa.IIDProcess(
                rv=rv, length=5, max_trajectories=10, enumerate=True, random_state=42
            )
            process2 = sa.IIDProcess(
                rv=rv, length=5, max_trajectories=10, enumerate=True, random_state=42
            )
        pd.testing.assert_frame_equal(
            process1.process_trajectories.values, process2.process_trajectories.values
        )


class TestEnumerationProperties:

    @pytest.fixture
    def complete_enum_process(self):
        rv = bernoulli(0.6)
        return sa.IIDProcess(rv=rv, length=4, enumerate=True, name="X")

    @pytest.fixture
    def partial_enum_process(self):
        rv = bernoulli(0.5)
        with pytest.warns(RuntimeWarning, match="exceeds max_trajectories"):
            process = sa.IIDProcess(
                rv=rv, length=8, max_trajectories=50, enumerate=True, random_state=123
            )
        return process

    def test_enumerate_property(self, complete_enum_process):
        assert complete_enum_process.enumerate is True

    def test_n_possible_trajectories_bernoulli(self, complete_enum_process):
        assert complete_enum_process.n_possible_trajectories == 16

    def test_n_possible_trajectories_binom(self):
        rv = binom(n=2, p=0.5)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        assert process.n_possible_trajectories == 27

    def test_is_complete_enumeration_true(self, complete_enum_process):
        assert complete_enum_process.is_complete_enumeration is True

    def test_is_complete_enumeration_false(self, partial_enum_process):
        assert partial_enum_process.is_complete_enumeration is False

    def test_is_complete_enumeration_false_for_simulation(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=4, enumerate=False)
        assert process.is_complete_enumeration is False

    def test_n_trajectories_equals_possible_for_complete(self, complete_enum_process):
        assert (
            complete_enum_process.n_trajectories
            == complete_enum_process.n_possible_trajectories
        )

    def test_n_trajectories_less_than_possible_for_partial(self, partial_enum_process):
        assert (
            partial_enum_process.n_trajectories
            < partial_enum_process.n_possible_trajectories
        )

    def test_n_possible_trajectories_continuous_distribution(self):
        rv = norm(0, 1)
        process = sa.IIDProcess(rv=rv, length=5)
        assert process.n_possible_trajectories == float("inf")


class TestEnumerationValidation:

    def test_enumerate_continuous_distribution_raises_error(self):
        rv = norm(0, 1)
        with pytest.raises(ValueError, match="Cannot enumerate.*continuous"):
            sa.IIDProcess(rv=rv, enumerate=True)

    def test_enumerate_invalid_type(self):
        rv = bernoulli(0.5)
        with pytest.raises(TypeError, match="enumerate must be a boolean"):
            sa.IIDProcess(rv=rv, enumerate="yes")

    def test_enumerate_large_trajectory_count_warns(self):
        rv = bernoulli(0.5)
        with pytest.warns(RuntimeWarning):
            sa.IIDProcess(rv=rv, length=21, enumerate=True)

    def test_enumerate_exceeds_max_trajectories_warns(self):
        rv = bernoulli(0.5)
        with pytest.warns(RuntimeWarning, match="exceeds max_trajectories"):
            sa.IIDProcess(rv=rv, length=10, max_trajectories=100, enumerate=True)


class TestEnumerationExactProbabilities:

    def test_exact_probabilities_bernoulli_fair(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        probs = list(process.probability_measure.values)
        expected_prob = 0.125
        for prob in probs:
            assert abs(prob - expected_prob) < 1e-10

    def test_exact_probabilities_bernoulli_biased(self):
        rv = bernoulli(0.3)
        process = sa.IIDProcess(rv=rv, length=2, enumerate=True)
        probs = process.probability_measure.values
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
        process = sa.IIDProcess(rv=rv, length=4, enumerate=True)
        total_prob = sum(process.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_exact_probabilities_binom(self):
        rv = binom(n=2, p=0.5)
        process = sa.IIDProcess(rv=rv, length=2, enumerate=True)
        total_prob = sum(process.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_partial_enumeration_probabilities_sum_to_one(self):
        rv = bernoulli(0.5)
        with pytest.warns(RuntimeWarning, match="exceeds max_trajectories"):
            process = sa.IIDProcess(
                rv=rv, length=6, max_trajectories=20, enumerate=True, random_state=42
            )
        total_prob = sum(process.probability_measure.values)
        assert abs(total_prob - 1.0) < 1e-10


class TestEnumerationTrajectories:

    def test_all_trajectories_enumerated_bernoulli(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        trajectories = process.process_trajectories.values
        assert len(trajectories) == 8
        unique_trajectories = {tuple(row) for row in trajectories.values}
        assert len(unique_trajectories) == 8

    def test_trajectories_contain_only_valid_values_bernoulli(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=4, enumerate=True)
        values = process.process_trajectories.values.values.flatten()
        assert all(v in [0, 1] for v in values)

    def test_trajectories_contain_only_valid_values_binom(self):
        rv = binom(n=2, p=0.5)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        values = process.process_trajectories.values.values.flatten()
        assert all(v in [0, 1, 2] for v in values)

    def test_partial_enumeration_has_correct_count(self):
        rv = bernoulli(0.5)
        with pytest.warns(RuntimeWarning, match="exceeds max_trajectories"):
            process = sa.IIDProcess(
                rv=rv, length=6, max_trajectories=20, enumerate=True, random_state=42
            )
        assert len(process.process_trajectories.values) == 20
        assert process.n_trajectories == 20

    def test_enumeration_with_initial_time(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=3, initial_time=5, enumerate=True)
        time_index = process.time_index
        assert list(time_index) == [5, 6, 7]
        assert len(process.process_trajectories.values) == 8


class TestEnumerationComparison:

    def test_simulation_vs_enumeration_different_mode(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        sim_process = sa.IIDProcess(rv=rv1, length=4, enumerate=False, random_state=42)
        enum_process = sa.IIDProcess(rv=rv2, length=4, enumerate=True)
        assert sim_process.enumerate is False
        assert enum_process.enumerate is True
        assert sim_process != enum_process

    def test_equal_enumerated_processes(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        process1 = sa.IIDProcess(rv=rv1, length=4, enumerate=True)
        process2 = sa.IIDProcess(rv=rv2, length=4, enumerate=True)
        assert process1 == process2

    def test_enumeration_produces_deterministic_order(self):
        rv = bernoulli(0.5)
        process1 = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        process2 = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        pd.testing.assert_frame_equal(
            process1.process_trajectories.values, process2.process_trajectories.values
        )


class TestEnumerationPlotting:

    def test_plot_title_with_enumeration(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        title = process._plot_title()
        assert title == "Enumerated IID Bernoulli Process"

    def test_plot_title_uses_dist_name(self):
        rv = binom(n=3, p=0.5)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        title = process._plot_title()
        assert title == "Enumerated IID Binom Process"

    def test_plot_trajectories_with_enumeration(self):
        rv = bernoulli(0.7)
        process = sa.IIDProcess(rv=rv, length=3, enumerate=True)
        ax = process.plot_trajectories()
        assert ax is not None


class TestEnumerationDiscreteSupport:

    def test_bernoulli_support(self):
        rv = bernoulli(0.5)
        process = sa.IIDProcess(rv=rv, length=2, enumerate=True)
        values = set(process.process_trajectories.values.values.flatten())
        assert values == {0, 1}

    def test_binom_support(self):
        rv = binom(n=3, p=0.5)
        process = sa.IIDProcess(rv=rv, length=2, enumerate=True)
        values = set(process.process_trajectories.values.values.flatten())
        assert values.issubset({0, 1, 2, 3})

    def test_poisson_support(self):
        rv = poisson(mu=2)
        process = sa.IIDProcess(rv=rv, length=2, enumerate=True)
        values = process.process_trajectories.values.values.flatten()
        assert all(v >= 0 for v in values)
        assert all(isinstance(int(v), int) for v in values)

    def test_randint_support(self):
        rv = randint(low=0, high=3)
        process = sa.IIDProcess(rv=rv, length=2, enumerate=True)
        values = set(process.process_trajectories.values.values.flatten())
        assert values.issubset({0, 1, 2})


class TestEnumerationRandomVariable:

    @pytest.fixture
    def enum_process(self):
        rv = bernoulli(0.6)
        return sa.IIDProcess(rv=rv, length=3, enumerate=True)

    def test_rv_at_with_enumeration(self, enum_process):
        rv = enum_process.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)
        assert rv.probability_space is not None

    def test_rv_at_has_exact_probabilities(self, enum_process):
        rv = enum_process.rv_at[0]
        prob_0 = rv.P(0)
        prob_1 = rv.P(1)
        assert abs(prob_0 + prob_1 - 1.0) < 1e-10
