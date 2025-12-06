import pandas as pd
import pytest
from scipy.stats import bernoulli, binom, norm

import sigalg as sa


class TestConstructor:

    def test_basic_construction_with_bernoulli(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time)
        assert X.rv == rv
        assert X.max_trajectories == 1000
        assert len(X.time) == 10
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
        assert len(Y.time) == 20
        assert Y.initial_time == 5
        assert Y.name == "Y"

    def test_construction_with_support(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1])
        assert X.support == [0, 1]
        assert X.n_support == 2

    def test_construction_generates_trajectories(self):
        rv = bernoulli(0.3)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time, max_trajectories=10)
        assert X.trajectories is not None
        assert isinstance(X.trajectories, sa.Trajectories)

    def test_construction_with_random_state_reproducible(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X1 = sa.IIDProcess(rv=rv, time=time, random_state=123, max_trajectories=100)
        X2 = sa.IIDProcess(rv=rv, time=time, random_state=123, max_trajectories=100)
        pd.testing.assert_frame_equal(X1.trajectories.values, X2.trajectories.values)


class TestProperties:

    @pytest.fixture
    def X(self):
        rv = bernoulli(0.6)
        time = sa.Time.discrete(start=2, length=8)
        return sa.IIDProcess(
            rv=rv,
            time=time,
            max_trajectories=50,
            name="Z",
            random_state=99,
        )

    def test_rv_property(self, X):
        assert isinstance(X.rv, type(bernoulli(0.5)))

    def test_max_trajectories_property(self, X):
        assert X.max_trajectories == 50

    def test_initial_time_property(self, X):
        assert X.initial_time == 2

    def test_name_property(self, X):
        assert X.name == "Z"

    def test_n_trajectories_property(self, X):
        assert X.n_trajectories > 0
        assert X.n_trajectories <= X.max_trajectories

    def test_time_property(self, X):
        assert isinstance(X.time, sa.Time)
        assert len(X.time) == 8

    def test_length_property(self, X):
        assert X.length == 8

    def test_support_property(self, X):
        assert X.support is None

    def test_n_support_property(self, X):
        assert X.n_support is None

    def test_enumerate_property(self, X):
        assert X.enumerate is False

    def test_random_state_property(self, X):
        assert X.random_state == 99

    def test_probability_space_property(self, X):
        assert X.probability_space is not None
        assert isinstance(X.probability_space, sa.ProbabilitySpace)

    def test_sample_space_property(self, X):
        assert X.sample_space is not None
        assert isinstance(X.sample_space, sa.SampleSpace)

    def test_sigma_algebra_property(self, X):
        assert X.sigma_algebra is not None
        assert isinstance(X.sigma_algebra, sa.SigmaAlgebra)

    def test_probability_measure_property(self, X):
        assert X.probability_measure is not None
        assert isinstance(X.probability_measure, sa.ProbabilityMeasure)


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
            sa.IIDProcess(rv="not a distribution", time=time)

    def test_invalid_time_type(self):
        rv = bernoulli(0.5)
        with pytest.raises(TypeError, match="time must be a Time object"):
            sa.IIDProcess(rv=rv, time="not a time")

    def test_invalid_max_trajectories_not_int(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, time=time, max_trajectories=10.5)

    def test_invalid_max_trajectories_negative(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, time=time, max_trajectories=-5)

    def test_invalid_max_trajectories_zero(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            ValueError, match="max_trajectories must be a positive integer"
        ):
            sa.IIDProcess(rv=rv, time=time, max_trajectories=0)

    def test_invalid_name_not_string(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(TypeError, match="name must be a string"):
            sa.IIDProcess(rv=rv, time=time, name=123)

    def test_invalid_random_state_not_int(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            TypeError, match="random_state must be a non-negative integer or None"
        ):
            sa.IIDProcess(rv=rv, time=time, random_state=12.5)

    def test_invalid_random_state_negative(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(
            TypeError, match="random_state must be a non-negative integer or None"
        ):
            sa.IIDProcess(rv=rv, time=time, random_state=-1)

    def test_invalid_support_not_list(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(TypeError, match="support must be a list or None"):
            sa.IIDProcess(rv=rv, time=time, support="invalid")

    def test_invalid_enumerate_not_bool(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        with pytest.raises(TypeError, match="enumerate must be a boolean"):
            sa.IIDProcess(rv=rv, time=time, enumerate="yes")


class TestEquality:

    def test_equal_processes(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X1 = sa.IIDProcess(rv=rv1, time=time, random_state=42, max_trajectories=50)
        X2 = sa.IIDProcess(rv=rv2, time=time, random_state=42, max_trajectories=50)
        assert X1 == X2

    def test_not_equal_different_random_state(self):
        rv1 = bernoulli(0.5)
        rv2 = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X1 = sa.IIDProcess(rv=rv1, time=time, random_state=42, max_trajectories=100)
        X2 = sa.IIDProcess(rv=rv2, time=time, random_state=99, max_trajectories=100)
        assert X1 != X2

    def test_not_equal_different_type(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        X = sa.IIDProcess(rv=rv, time=time)
        assert X != "not a process"
        assert X != 42
        assert X is not None


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
        assert X.initial_time == 5
        assert len(X.time) == 10

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
        assert len(trajectory) == len(X.time)

    def test_rv_at_indexer(self, X):
        rv = X.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)

    def test_rv_at_different_times(self, X):
        rv0 = X.rv_at[0]
        rv1 = X.rv_at[1]
        assert rv0.name != rv1.name


class TestRandomVariable:

    @pytest.fixture
    def X(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=10)
        return sa.IIDProcess(rv=rv, time=time, max_trajectories=100, random_state=42)

    def test_rv_at_returns_random_variable(self, X):
        rv = X.rv_at[0]
        assert isinstance(rv, sa.RandomVariable)

    def test_rv_at_has_probability_space(self, X):
        rv = X.rv_at[0]
        assert rv.probability_space is not None

    def test_rv_at_invalid_time_raises_error(self, X):
        with pytest.raises(ValueError, match="not in process time index"):
            X.rv_at[100]


class TestEnumeration:

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

    def test_enumerate_without_support_raises_error(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        with pytest.raises(
            ValueError, match="Cannot enumerate trajectories without explicit support"
        ):
            sa.IIDProcess(rv=rv, time=time, enumerate=True)

    def test_enumerate_large_trajectory_raises(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=25)
        with pytest.raises(ValueError, match="too large to enumerate"):
            sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)

    def test_exact_probabilities_bernoulli_fair(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        probs = list(X.probability_measure.values.values)
        expected_prob = 0.125
        for prob in probs:
            assert abs(prob - expected_prob) < 1e-10

    def test_probabilities_sum_to_one(self):
        rv = bernoulli(0.4)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        total_prob = sum(X.probability_measure.values.values)
        assert abs(total_prob - 1.0) < 1e-10

    def test_all_trajectories_enumerated(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        assert X.n_trajectories == 8

    def test_trajectories_contain_only_valid_values(self):
        rv = bernoulli(0.5)
        time = sa.Time.discrete(start=0, length=3)
        X = sa.IIDProcess(rv=rv, time=time, support=[0, 1], enumerate=True)
        values = X.trajectories.values.values.flatten()
        assert all(v in [0, 1] for v in values)


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
