import numpy as np
import pandas as pd
import pytest
from scipy.stats import bernoulli, norm

from sigalg.core import SigmaAlgebra, Time
from sigalg.processes import IIDProcess


class TestConstructor:

    def test_constructor_with_distribution(self):
        """Test basic construction with a distribution."""
        dist = bernoulli(p=0.5)
        time = Time.discrete(length=3)
        X = IIDProcess(distribution=dist, time=time, name="X")

        assert X.name == "X"
        assert X.time == time
        assert X.distribution == dist
        assert X.is_discrete_state is True
        assert X.is_discrete_time is True


class TestDataGeneration:

    @pytest.fixture
    def bernoulli(self):
        return bernoulli(p=0.5)

    @pytest.fixture
    def gaussian(self):
        return norm(loc=0.0, scale=1.0)

    @pytest.fixture
    def time_discrete(self):
        return Time.discrete(length=2)

    @pytest.fixture
    def time_continuous(self):
        return Time.continuous(start=0.0, stop=1.0, num_points=5)

    def test_from_enumeration_bernoulli_and_discrete_time(
        self, bernoulli, time_discrete
    ):
        """Test from_enumeration with Bernoulli distribution and discrete time."""
        X = IIDProcess(
            distribution=bernoulli, support=[0, 1], time=time_discrete, name="X"
        ).from_enumeration()

        assert X.n_trajectories == 8
        assert X._is_enumerated is True
        assert X.is_discrete_state is True
        assert X.is_discrete_time is True

    def test_from_simulation_bernoulli_and_discrete_time(
        self, bernoulli, time_discrete
    ):
        """Test from_simulation with Bernoulli distribution and discrete time."""
        X = IIDProcess(
            distribution=bernoulli, time=time_discrete, name="X"
        ).from_simulation(n_trajectories=100, random_state=42)

        assert len(X.data) == 100
        assert X._is_enumerated is False
        assert X.time == time_discrete
        assert X.is_discrete_state is True
        assert X.is_discrete_time is True

    def test_from_enumeration_bernoulli_and_continuous_time(
        self, bernoulli, time_continuous
    ):
        """Test from_enumeration with Bernoulli distribution and continuous time."""
        X = IIDProcess(
            distribution=bernoulli, support=[0, 1], time=time_continuous, name="X"
        ).from_enumeration()

        assert X.n_trajectories == 32
        assert X._is_enumerated is True
        assert X.is_discrete_state is True
        assert X.is_discrete_time is False

    def test_from_simulation_gaussian_and_discrete_time(self, gaussian, time_discrete):
        """Test from_simulation with Gaussian distribution and discrete time."""
        X = IIDProcess(
            distribution=gaussian, time=time_discrete, name="X"
        ).from_simulation(n_trajectories=100, random_state=42)

        assert len(X.data) == 100
        assert X._is_enumerated is False
        assert X.time == time_discrete
        assert X.is_discrete_state is False
        assert X.is_discrete_time is True

    def test_from_enumeration_gaussian_raises(self, gaussian, time_discrete):
        """Test that from_enumeration with Gaussian distribution raises ValueError."""
        X = IIDProcess(distribution=gaussian, time=time_discrete, name="X")

        with pytest.raises(
            ValueError, match="Enumeration is only supported for discrete state spaces"
        ):
            X.from_enumeration()

    def test_from_enumeration_creates_discrete_time_if_not_provided(self, bernoulli):
        """Test from_enumeration creates time index when not provided."""
        X = IIDProcess(
            distribution=bernoulli, support=[0, 1], is_discrete_time=True
        ).from_enumeration(length=1)
        expected_time = Time.discrete(length=1)

        assert X.time == expected_time
        assert X.n_trajectories == 4
        assert X.is_discrete_state is True
        assert X.is_discrete_time is True

    def test_from_simulation_creates_discrete_time_if_not_provided(self, bernoulli):
        """Test from_simulation creates time index when not provided."""
        X = IIDProcess(distribution=bernoulli, is_discrete_time=True).from_simulation(
            n_trajectories=10, length=3, random_state=42
        )
        expected_time = Time.discrete(length=3)

        assert X.time == expected_time
        assert X.time.is_discrete is True

    def test_from_simulation_reproducibility(self, bernoulli, time_discrete):
        """Test that from_simulation with same random_state gives same results."""
        X1 = IIDProcess(distribution=bernoulli, time=time_discrete).from_simulation(
            n_trajectories=20, random_state=42
        )
        X2 = IIDProcess(distribution=bernoulli, time=time_discrete).from_simulation(
            n_trajectories=20, random_state=42
        )

        pd.testing.assert_frame_equal(X1.data, X2.data)


def test_natural_filtration():
    """Test the natural filtration of an IID process."""
    T = Time.discrete(start=1, length=2)
    X = IIDProcess(
        distribution=bernoulli(p=0.7),
        support=[0, 1],
        time=T,
    ).from_enumeration()
    F = X.natural_filtration

    F1_atom_ids = [0, 0, 0, 0, 1, 1, 1, 1]
    F1_sample_id_to_atom_id = dict(zip(X.domain, F1_atom_ids, strict=False))
    expected_F1 = SigmaAlgebra().from_dict(F1_sample_id_to_atom_id)

    F2_atom_ids = [0, 0, 1, 1, 2, 2, 3, 3]
    F2_sample_id_to_atom_id = dict(zip(X.domain, F2_atom_ids, strict=False))
    expected_F2 = SigmaAlgebra().from_dict(F2_sample_id_to_atom_id)

    F3_atom_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    F3_sample_id_to_atom_id = dict(zip(X.domain, F3_atom_ids, strict=False))
    expected_F3 = SigmaAlgebra().from_dict(F3_sample_id_to_atom_id)

    expected_filtration = [expected_F1, expected_F2, expected_F3]
    for actual_algebra, expected_algebra in zip(
        F.sigma_algebras, expected_filtration, strict=False
    ):
        assert actual_algebra == expected_algebra


def test_is_discrete_time_and_state():
    """Test is_discrete_time and is_discrete_state for Bernoulli IID process."""
    dist_discrete = bernoulli(p=0.5)
    dist_continuous = norm(loc=0.0, scale=1.0)
    time_discrete = Time.discrete(length=3)
    time_continuous = Time.continuous(start=0.0, stop=1.0, num_points=5)

    X = IIDProcess(
        distribution=dist_discrete, support=[0, 1], time=time_discrete
    ).from_enumeration()
    Y = IIDProcess(
        distribution=dist_discrete, support=[0, 1], time=time_continuous
    ).from_enumeration()
    Z = IIDProcess(distribution=dist_continuous, time=time_discrete).from_simulation(
        n_trajectories=2, random_state=42
    )
    W = IIDProcess(distribution=dist_continuous, time=time_continuous).from_simulation(
        n_trajectories=2, random_state=42
    )

    assert X.is_discrete_time is True
    assert X.is_discrete_state is True
    assert Y.is_discrete_time is False
    assert Y.is_discrete_state is True
    assert Z.is_discrete_time is True
    assert Z.is_discrete_state is False
    assert W.is_discrete_time is False
    assert W.is_discrete_state is False


def test_time_setter():
    """Test setting the time index of a stochastic process."""
    time1 = Time.discrete(length=3)
    time2 = Time.discrete(length=4)
    X = IIDProcess(
        distribution=bernoulli(p=0.5), support=[0, 1], time=time1
    ).from_enumeration()
    X.time = time2

    assert X.time == time2
    assert X.data is None
    assert X.domain is None


class TestProbabilityMeasure:

    def test_exact_probability_measure_bernoulli(self):
        """Test exact probability measure for enumerated IID Bernoulli process."""
        p = 0.6
        dist = bernoulli(p=p)
        X = IIDProcess(
            distribution=dist, support=[0, 1], is_discrete_time=True
        ).from_enumeration(length=1)
        P = X.probability_measure
        expected_probabilities = pd.Series(
            [0.16, 0.24, 0.24, 0.36], index=X.domain.data, name="probability"
        )

        pd.testing.assert_series_equal(P.data, expected_probabilities)

    def test_empirical_probability_measure_from_simulation(self):
        """Test empirical probability measure for simulated IID process."""
        dist = bernoulli(p=0.5)
        X = IIDProcess(distribution=dist, is_discrete_time=True).from_simulation(
            n_trajectories=100_000,
            length=1,
            random_state=42,
        )
        P_X = X.range.probability_measure

        assert all(np.isclose(P_X.data, 0.25, rtol=0, atol=0.01))


class TestPlotTitle:

    def test_plot_title_includes_distribution_name(self):
        """Test that _plot_title includes distribution name."""
        dist = bernoulli(p=0.5)
        X = IIDProcess(
            distribution=dist, support=[0, 1], name="X", is_discrete_time=True
        ).from_enumeration(length=2)
        title = X._plot_title()

        assert "bernoulli" in title.lower()
