import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from sigalg.core import (
    Index,
    ProbabilityMeasure,
    RandomVariable,
    SampleSpace,
    Time,
)
from sigalg.processes import BrownianMotion, RandomWalk, StochasticProcess


class TestConstructor:

    def test_constructor_with_time_and_domain(self):
        """Test StochasticProcess constructor with time and domain."""
        domain = SampleSpace().from_sequence(size=3)
        time = Time.discrete(length=4)
        X = StochasticProcess(domain=domain, time=time)

        assert X.domain == domain
        assert X.time == time
        assert X.is_discrete_time is True
        assert X.name == "X"

    def test_constructor_with_time_only(self):
        """Test StochasticProcess constructor with time only."""
        time = Time.discrete(length=5)
        X = StochasticProcess(time=time)

        assert X.domain is None
        assert X.time == time
        assert X.is_discrete_time is True
        assert X.name == "X"

    def test_constructor_with_is_discrete_time_only(self):
        """Test StochasticProcess constructor with is_discrete_time only."""
        X = StochasticProcess(is_discrete_time=True)

        assert X.domain is None
        assert X.time is None
        assert X.is_discrete_time is True
        assert X.name == "X"

    def test_constructor_with_custom_name(self):
        """Test StochasticProcess constructor with custom name."""
        time = Time.discrete(length=3)
        Y = StochasticProcess(time=time, name="Y")

        assert Y.name == "Y"
        assert Y.time == time

    def test_constructor_with_none_name(self):
        """Test StochasticProcess constructor with None name."""
        time = Time.discrete(length=3)
        X = StochasticProcess(time=time, name=None)

        assert X.name is None
        assert X.time == time

    def test_constructor_with_custom_is_discrete_state(self):
        """Test StochasticProcess constructor with custom is_discrete_state."""
        time = Time.discrete(length=3)
        X = StochasticProcess(time=time, is_discrete_state=True)

        assert X.is_discrete_state is True

    def test_constructor_with_none_is_discrete_state(self):
        """Test StochasticProcess constructor with None is_discrete_state."""
        time = Time.discrete(length=3)
        X = StochasticProcess(time=time, is_discrete_state=None)

        assert X.is_discrete_state is None

    def test_constructor_time_not_time_type_raises(self):
        """Test that time parameter must be Time instance or None."""
        idx = Index().from_sequence(size=3)

        with pytest.raises(TypeError, match="time must be an instance of Time or None"):
            StochasticProcess(time=idx, is_discrete_time=True)

    def test_constructor_is_discrete_time_not_bool_raises(self):
        """Test that is_discrete_time parameter must be bool or None."""
        with pytest.raises(
            TypeError, match="is_discrete_time must be a boolean or None"
        ):
            StochasticProcess(time=Time.discrete(length=3), is_discrete_time="True")

    def test_constructor_inconsistent_time_and_is_discrete_time_raises(self):
        """Test that time and is_discrete_time must be consistent."""
        time = Time.discrete(length=3)
        with pytest.raises(
            ValueError,
            match="is_discrete_time property must be consistent with the discreteness",
        ):
            StochasticProcess(time=time, is_discrete_time=False)

    def test_constructor_no_time_or_is_discrete_time_raises(self):
        """Test that at least one of time or is_discrete_time must be provided."""
        with pytest.raises(
            ValueError,
            match="At least one of time or is_discrete_time must be provided",
        ):
            StochasticProcess()


class TestProperties:

    def test_time_property(self):
        """Test time property returns the time index."""
        time = Time.discrete(length=4)
        X = StochasticProcess(time=time)

        assert X.time == time

    def test_time_property_none(self):
        """Test time property returns None when not set."""
        X = StochasticProcess(is_discrete_time=True)

        assert X.time is None

    def test_time_setter(self):
        """Test time setter updates the time index."""
        X = StochasticProcess(is_discrete_time=True)
        time = Time.discrete(length=5)
        X.time = time

        assert X.time == time

    def test_time_setter_invalid_type_raises(self):
        """Test that time setter raises TypeError for non-Time objects."""
        X = StochasticProcess(is_discrete_time=True)
        with pytest.raises(TypeError):
            X.time = "not a Time instance"

    def test_n_trajectories_with_data(self):
        """Test n_trajectories property with data."""
        domain = SampleSpace().from_sequence(size=3)
        time = Time.discrete(length=3)
        X = StochasticProcess(domain=domain, time=time).from_dict(
            {
                0: (1, 2, 3, 4),
                1: (5, 6, 7, 8),
                2: (9, 10, 11, 12),
            }
        )

        assert X.n_trajectories == 3

    def test_n_trajectories_without_data(self):
        """Test n_trajectories property without data."""
        time = Time.discrete(length=4)
        X = StochasticProcess(time=time)

        assert X.n_trajectories is None

    def test_probability_measure_raises_without_data(self):
        """Test that probability_measure raises ValueError without data."""
        time = Time.discrete(length=4)
        X = StochasticProcess(time=time)

        with pytest.raises(ValueError):
            _ = X.probability_measure

    def test_probability_measure_with_non_generated_data_returns_uniform(self):
        """Test that probability_measure returns uniform measure with non-generated data."""
        domain = SampleSpace().from_sequence(size=3)
        time = Time.discrete(length=2)
        X = StochasticProcess(domain=domain, time=time).from_dict(
            {
                0: (1, 2, 3),
                1: (3, 4, 5),
                2: (5, 6, 7),
            }
        )

        expected_measure = ProbabilityMeasure.uniform(sample_space=domain)
        assert X.probability_measure == expected_measure


class TestFromConstant:

    def test_from_constant_with_domain_and_time(self):
        """Test from_constant method with domain and time."""
        domain = SampleSpace().from_sequence(size=2)
        time = Time.discrete(length=2)
        X = StochasticProcess(domain=domain, time=time).from_constant(value=5)

        expected_data = pd.DataFrame(
            [[5, 5, 5], [5, 5, 5]],
            index=domain.data,
            columns=time.data,
        )

        pd.testing.assert_frame_equal(X.data, expected_data)

    def test_from_constant_with_length_parameter(self):
        """Test from_constant method with length parameter."""
        domain = SampleSpace().from_sequence(size=2)
        X = StochasticProcess(domain=domain, is_discrete_time=True).from_constant(
            value=10, length=2
        )

        expected_time = Time().discrete(length=2)
        expected_data = pd.DataFrame(
            [[10, 10, 10], [10, 10, 10]],
            index=domain.data,
            columns=expected_time.data,
        )

        pd.testing.assert_frame_equal(X.data, expected_data)

    def test_from_constant_sets_is_enumerated(self):
        """Test that from_constant sets is_enumerated to True."""
        domain = SampleSpace().from_sequence(size=2)
        time = Time.discrete(length=3)
        X = StochasticProcess(domain=domain, time=time).from_constant(value=1)

        assert X.is_enumerated is True

    def test_from_constant_sets_uniform_probability_measure(self):
        """Test that from_constant sets uniform probability measure."""
        domain = SampleSpace().from_sequence(size=2)
        time = Time.discrete(length=3)
        X = StochasticProcess(domain=domain, time=time).from_constant(value=1)

        expected_measure = ProbabilityMeasure.uniform(sample_space=domain)
        assert X.probability_measure == expected_measure

    def test_from_constant_invalid_length_raises(self):
        """Test that invalid length parameter raises ValueError."""
        domain = SampleSpace.generate_sequence(size=2)
        time = Time.discrete(length=3)
        X = StochasticProcess(domain=domain, time=time)

        with pytest.raises(ValueError):
            X.from_constant(value=1, length=-1)

    def test_from_constant_without_domain_raises(self):
        """Test that from_constant raises ValueError without domain."""
        time = Time.discrete(length=3)
        X = StochasticProcess(time=time)

        with pytest.raises(ValueError):
            X.from_constant(value=1)

    def test_from_constant_non_numeric_value_raises(self):
        """Test that non-numeric value raises TypeError."""
        domain = SampleSpace().from_sequence(size=2)
        time = Time.discrete(length=3)
        X = StochasticProcess(domain=domain, time=time)

        with pytest.raises(TypeError):
            X.from_constant(value="not a number")


class TestDataAccess:

    @pytest.fixture
    def process(self):
        domain = SampleSpace().from_sequence(size=3)
        time = Time.discrete(start=1, length=3)
        return StochasticProcess(domain=domain, time=time, name="X").from_dict(
            {
                0: (1, 2, 3, 4),
                1: (5, 6, 7, 8),
                2: (9, 10, 11, 12),
            }
        )

    def test_getitem_returns_random_variable(self, process):
        """Test __getitem__ returns a RandomVariable."""
        rv = process[2]
        expected_data = pd.Series(
            [2, 6, 10],
            index=process.domain.data,
            name="X_2",
        )

        pd.testing.assert_series_equal(rv.data, expected_data)

    def test_getitem_without_time_raises(self):
        """Test that __getitem__ raises ValueError without time."""
        X = StochasticProcess(is_discrete_time=True)

        with pytest.raises(ValueError):
            _ = X[0]

    def test_iter_returns_random_variables(self, process):
        """Test __iter__ yields RandomVariable instances."""
        rvs = list(process)

        assert len(rvs) == 4
        assert all(isinstance(rv, RandomVariable) for rv in rvs)


class TestEquality:

    def test_equality_same_processes(self):
        """Test equality for identical processes."""
        domain = SampleSpace().from_sequence(size=2)
        time = Time.discrete(length=2)
        outputs = {0: (1, 2, 3), 1: (4, 5, 6)}

        X1 = StochasticProcess(domain=domain, time=time, name="X1").from_dict(outputs)
        X2 = StochasticProcess(domain=domain, time=time, name="X2").from_dict(outputs)

        assert X1 == X2

    def test_inequality_different_data(self):
        """Test inequality for processes with different data."""
        domain = SampleSpace().from_sequence(size=2)
        time = Time.discrete(length=2)
        outputs1 = {0: (1, 2, 3), 1: (4, 5, 6)}
        outputs2 = {0: (1, 2, 3), 1: (4, 5, 7)}

        X1 = StochasticProcess(domain=domain, time=time).from_dict(outputs1)
        X2 = StochasticProcess(domain=domain, time=time).from_dict(outputs2)

        assert X1 != X2

    def test_inequality_with_non_stochastic_process(self):
        """Test inequality comparison with non-StochasticProcess object."""
        domain = SampleSpace().from_sequence(size=2)
        time = Time.discrete(length=2)
        outputs = {0: (1, 2, 3), 1: (4, 5, 6)}
        X = StochasticProcess(domain=domain, time=time).from_dict(outputs)

        assert X != "not a stochastic process"
        assert X != 42


class TestValidationHelpers:

    def test_validate_and_initialize_time_with_length(self):
        """Test _validate_and_initialize_time with length parameter."""
        X = StochasticProcess(is_discrete_time=True)
        X._validate_and_initialize_time(length=4)

        assert X.time is not None
        assert len(X.time) == 5

    def test_validate_and_initialize_time_with_existing_time(self):
        """Test _validate_and_initialize_time with existing time."""
        time = Time.discrete(length=4)
        X = StochasticProcess(time=time)
        X._validate_and_initialize_time()

        assert X.time == time

    def test_validate_and_initialize_time_no_time_or_length_raises(self):
        """Test that _validate_and_initialize_time raises without time or length."""
        X = StochasticProcess(is_discrete_time=True)

        with pytest.raises(ValueError):
            X._validate_and_initialize_time()

    def test_validate_and_initialize_domain_creates_domain(self):
        """Test _validate_and_initialize_domain creates domain."""
        X = StochasticProcess(is_discrete_time=True)
        X._validate_and_initialize_domain(n_trajectories=3)

        assert X.domain is not None
        assert len(X.domain) == 3

    def test_validate_and_initialize_domain_with_existing_domain(self):
        """Test _validate_and_initialize_domain with existing domain."""
        domain = SampleSpace.generate_sequence(size=3)
        X = StochasticProcess(domain=domain, is_discrete_time=True)
        X._validate_and_initialize_domain(n_trajectories=3)

        assert X.domain == domain

    def test_validate_and_initialize_domain_mismatched_size_raises(self):
        """Test that _validate_and_initialize_domain raises with mismatched size."""
        domain = SampleSpace.generate_sequence(size=2)
        X = StochasticProcess(domain=domain, is_discrete_time=True)

        with pytest.raises(ValueError):
            X._validate_and_initialize_domain(n_trajectories=3)


class TestPlotTrajectories:

    def test_plot_trajectories_returns_axes(self):
        """Test that plot_trajectories returns a matplotlib Axes object."""
        T = Time.discrete(length=2)
        X = RandomWalk(p=0.7, time=T).from_enumeration()
        ax = X.plot_trajectories()

        assert isinstance(ax, Axes)

    def test_plot_trajectories_over_discrete_time(self):
        """Test plot_trajectories with discrete time process."""
        T = Time.discrete(length=2)
        X = RandomWalk(p=0.7, time=T).from_enumeration()
        ax = X.plot_trajectories()

        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "state"
        assert ax.get_title() == "Enumerated random walk process 'X'"
        n_lines = len(ax.get_lines())
        assert n_lines == X.n_trajectories

        plt.close()

    def test_plot_trajectories_over_continuous_time(self):
        """Test plot_trajectories with continuous time process."""
        T = Time.continuous(start=1, stop=2, dt=0.13)
        X = BrownianMotion(time=T).from_simulation(n_trajectories=3, random_state=42)
        ax = X.plot_trajectories()

        assert isinstance(ax, Axes)
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "state"
        n_lines = len(ax.get_lines())
        assert n_lines == 3

        plt.close()

    def test_plot_trajectories_with_custom_labels(self):
        """Test plot_trajectories with custom axis labels."""
        T = Time.discrete(length=2)
        X = RandomWalk(p=0.5, time=T).from_enumeration()
        ax = X.plot_trajectories(
            x_label="Custom Time",
            y_label="Custom State",
            title="Custom Title",
        )

        assert ax.get_xlabel() == "Custom Time"
        assert ax.get_ylabel() == "Custom State"
        assert ax.get_title() == "Custom Title"

        plt.close()

    def test_plot_trajectories_with_custom_axes(self):
        """Test plot_trajectories with provided axes object."""
        _, custom_ax = plt.subplots()
        T = Time.discrete(length=2)
        X = RandomWalk(p=0.5, time=T).from_enumeration()
        ax = X.plot_trajectories(ax=custom_ax)

        assert ax is custom_ax
        assert len(ax.get_lines()) == X.n_trajectories

        plt.close()

    def test_plot_trajectories_with_plot_kwargs(self):
        """Test plot_trajectories with custom plot kwargs."""
        T = Time.discrete(length=2)
        X = RandomWalk(p=0.5, time=T).from_enumeration()
        ax = X.plot_trajectories(plot_kwargs={"linewidth": 3, "alpha": 0.5})

        lines = ax.get_lines()
        for line in lines:
            assert line.get_linewidth() == 3
            assert line.get_alpha() == 0.5

        plt.close()

    def test_plot_trajectories_without_data_raises(self):
        """Test that plot_trajectories raises ValueError without data."""
        T = Time.discrete(length=2)
        X = StochasticProcess(time=T)

        with pytest.raises(ValueError):
            X.plot_trajectories()

    def test_plot_trajectories_invalid_axes_raises(self):
        """Test that plot_trajectories raises TypeError with invalid axes."""
        T = Time.discrete(length=2)
        X = RandomWalk(p=0.5, time=T).from_enumeration()

        with pytest.raises(TypeError):
            X.plot_trajectories(ax="not an axes object")


class TestMartingaleMethods:

    def test_enumerated_symmetric_random_walk_is_martingale(self):
        """Test that an enumerated symmetric random walk is a martingale."""
        T = Time.discrete(length=5)
        X = RandomWalk(p=0.5, time=T).from_enumeration()

        assert X.is_martingale()
        assert X.is_submartingale()
        assert X.is_supermartingale()

    def test_enumerated_random_walk_with_positive_drift_is_submartingale(self):
        """Test that an enumerated random walk with positive drift is a submartingale."""
        T = Time.discrete(length=5)
        X = RandomWalk(p=0.7, time=T).from_enumeration()

        assert X.is_submartingale()
        assert not X.is_supermartingale()
        assert not X.is_martingale()

    def test_enumerated_random_walk_with_negative_drift_is_supermartingale(self):
        """Test that an enumerated random walk with negative drift is a supermartingale."""
        T = Time.discrete(length=5)
        X = RandomWalk(p=0.3, time=T).from_enumeration()

        assert X.is_supermartingale()
        assert not X.is_submartingale()
        assert not X.is_martingale()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_simulated_symmetric_random_walk_is_martingale(self):
        """Test that a simulated symmetric random walk is a martingale."""
        T = Time.discrete(length=4)
        X = RandomWalk(p=0.5, time=T).from_simulation(
            n_trajectories=10_000, random_state=42
        )

        assert X.is_martingale(atol=0.5)
        assert X.is_submartingale(atol=0.5)
        assert X.is_supermartingale(atol=0.5)

    def test_martingale_checks_raise_for_non_discrete_state(self):
        """Test that martingale checks raise ValueError for non-discrete-state processes."""
        T = Time.continuous(start=0, stop=1, dt=0.1)
        X = BrownianMotion(time=T).from_simulation(n_trajectories=3, random_state=42)

        with pytest.raises(ValueError):
            X.is_martingale()
        with pytest.raises(ValueError):
            X.is_submartingale()
        with pytest.raises(ValueError):
            X.is_supermartingale()
