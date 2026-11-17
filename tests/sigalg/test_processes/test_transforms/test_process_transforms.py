from math import inf

import pandas as pd
import pytest
from scipy.stats import bernoulli, expon
from sigalg.core import RandomVariable, Time
from sigalg.finance import BinomialPricingModel
from sigalg.processes import IIDProcess, ProcessTransforms, RandomWalk, StoppingTime

# --------------------- test insert_rv --------------------- #


class TestInsertRv:
    @pytest.fixture
    def X(self):
        T = Time.discrete(start=1, length=2)
        return IIDProcess.generate(
            mode="enum", distribution=bernoulli(p=0.5), support=[0, 1], index=T
        )

    def test_insert_rv_with_random_variable(self, X):
        """Test inserting a random variable at a given time with an explicit rv."""
        X0 = RandomVariable.from_constant(*X.prob_space, constant=0)
        X_insert = ProcessTransforms.insert_rv(process=X, rv=X0, time=0)
        expected_df = pd.DataFrame(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
            ],
            index=X.sample_space.data,
            columns=pd.Index([0, 1, 2, 3], name="t"),
        )

        pd.testing.assert_frame_equal(X_insert.data, expected_df, check_dtype=False)
        assert X_insert.name == "X_insert"

    def test_insert_rv_with_state_matches_insert_rv_with_rv(self, X):
        """Test that inserting a constant state matches inserting an equivalent rv."""
        X0 = RandomVariable.from_constant(*X.prob_space, constant=0)
        X_insert_rv = ProcessTransforms.insert_rv(process=X, rv=X0, time=0)
        X_insert_state = ProcessTransforms.insert_rv(process=X, state=0, time=0)

        pd.testing.assert_frame_equal(X_insert_state.data, X_insert_rv.data)

    def test_custom_name(self, X):
        """Test that a custom name is used when provided."""
        X_insert = ProcessTransforms.insert_rv(
            process=X, state=0, time=0, name="custom"
        )

        assert X_insert.name == "custom"

    def test_invalid_process_raises(self):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.insert_rv(process="not a process", time=0, state=0)

    def test_invalid_time_raises(self, X):
        """Test that a non-real time raises TypeError."""
        with pytest.raises(TypeError, match="time must be a real number."):
            ProcessTransforms.insert_rv(process=X, time="not a real number", state=0)

    def test_invalid_rv_raises(self, X):
        """Test that an invalid rv raises TypeError."""
        with pytest.raises(
            TypeError,
            match="If rv is provided, it must be an instance of RandomVariable.",
        ):
            ProcessTransforms.insert_rv(process=X, time=0, rv="not a random variable")

    def test_invalid_state_raises(self, X):
        """Test that a non-hashable state raises TypeError."""
        with pytest.raises(
            TypeError, match="If state is provided, it must be a hashable object."
        ):
            ProcessTransforms.insert_rv(process=X, time=0, state=[1, 2])

    def test_neither_rv_nor_state_raises(self, X):
        """Test that providing neither rv nor state raises ValueError."""
        with pytest.raises(
            ValueError, match="Exactly one of rv or state must be provided"
        ):
            ProcessTransforms.insert_rv(process=X, time=0)

    def test_both_rv_and_state_raises(self, X):
        """Test that providing both rv and state raises ValueError."""
        X0 = RandomVariable.from_constant(*X.prob_space, constant=0)

        with pytest.raises(
            ValueError, match="Exactly one of rv or state must be provided"
        ):
            ProcessTransforms.insert_rv(process=X, time=0, rv=X0, state=0)

    def test_mismatched_sample_space_raises(self, X):
        """Test that an rv with a different sample space than the process raises ValueError."""
        Y = IIDProcess.generate(
            mode="enum",
            distribution=bernoulli(p=0.5),
            support=[0, 1],
            index=Time.discrete(start=1, length=1),
        )
        Y0 = RandomVariable.from_constant(*Y.prob_space, constant=0)

        with pytest.raises(
            ValueError, match="process and rv must have the same sample_space."
        ):
            ProcessTransforms.insert_rv(process=X, time=0, rv=Y0)


# --------------------- test remove_rv --------------------- #


class TestRemoveRv:
    @pytest.fixture
    def X(self):
        T = Time.discrete(start=1, length=2)
        return RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=T)

    def test_remove_rv_by_time(self, X):
        """Test removing a random variable from a process by time."""
        X_remove = ProcessTransforms.remove_rv(process=X, time=2)
        expected_df = pd.DataFrame(
            [[0, 2], [0, 0], [0, 0], [0, -2]],
            index=X.sample_space.data,
            columns=pd.Index([1, 3], name="t"),
        )

        pd.testing.assert_frame_equal(X_remove.data, expected_df, check_dtype=False)
        assert X_remove.name == "X_remove"

    def test_remove_rv_by_pos_matches_remove_rv_by_time(self, X):
        """Test that removing a random variable by position matches removing by time."""
        X_remove_time = ProcessTransforms.remove_rv(process=X, time=2)
        X_remove_pos = ProcessTransforms.remove_rv(process=X, pos=1)

        pd.testing.assert_frame_equal(X_remove_pos.data, X_remove_time.data)

    def test_custom_name(self, X):
        """Test that a custom name is used when provided."""
        X_remove = ProcessTransforms.remove_rv(process=X, time=2, name="custom")

        assert X_remove.name == "custom"

    def test_invalid_process_raises(self):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.remove_rv(process="not a process", time=2)

    def test_invalid_time_raises(self, X):
        """Test that a non-real time raises TypeError."""
        with pytest.raises(TypeError, match="If provided, time must be a real number."):
            ProcessTransforms.remove_rv(process=X, time="not a real number")

    def test_invalid_pos_raises(self, X):
        """Test that a non-integer pos raises TypeError."""
        with pytest.raises(TypeError, match="If provided, pos must be an integer."):
            ProcessTransforms.remove_rv(process=X, pos="not an integer")

    def test_both_time_and_pos_raises(self, X):
        """Test that providing both time and pos raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both time and pos."):
            ProcessTransforms.remove_rv(process=X, time=2, pos=1)

    def test_neither_time_nor_pos_raises(self, X):
        """Test that providing neither time nor pos raises ValueError."""
        with pytest.raises(
            ValueError, match="Must specify exactly one of time or pos."
        ):
            ProcessTransforms.remove_rv(process=X)

    def test_time_not_in_index_raises(self, X):
        """Test that a time not in the process time index raises ValueError."""
        with pytest.raises(ValueError, match="time must be in the process time index."):
            ProcessTransforms.remove_rv(process=X, time=99)


# --------------------- test drop_last --------------------- #


class TestDropLast:
    @pytest.fixture
    def X(self):
        T = Time.discrete(start=1, length=2)
        return RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=T)

    def test_drop_last_removes_final_time_point(self, X):
        """Test that drop_last removes the final random variable in a process."""
        X_drop = ProcessTransforms.drop_last(process=X)
        expected_df = pd.DataFrame(
            [[0, 1], [0, 1], [0, -1], [0, -1]],
            index=X.sample_space.data,
            columns=pd.Index([1, 2], name="t"),
        )

        pd.testing.assert_frame_equal(X_drop.data, expected_df, check_dtype=False)
        assert X_drop.name == "X_drop"

    def test_custom_name(self, X):
        """Test that a custom name is used when provided."""
        X_drop = ProcessTransforms.drop_last(process=X, name="custom")

        assert X_drop.name == "custom"


# --------------------- test discount --------------------- #


class TestDiscount:
    @pytest.fixture
    def S(self):
        return BinomialPricingModel.generate(
            initial_price=100,
            up_prob=0.7,
            up_factor=1.1,
            down_factor=0.9,
            risk_free_rate=0.05,
            mode="enum",
            length=2,
        )

    def test_discount_with_rate(self, S):
        """Test discounting a process with a specified rate."""
        S_0, u, d, R = 100, 1.1, 0.9, 1.05
        expected_prices = [
            [S_0, S_0 * u, S_0 * u**2],
            [S_0, S_0 * u, S_0 * u * d],
            [S_0, S_0 * d, S_0 * u * d],
            [S_0, S_0 * d, S_0 * d**2],
        ]
        expected_discounted = [
            [price / R**t for t, price in enumerate(row)] for row in expected_prices
        ]
        expected_df = pd.DataFrame(
            expected_discounted,
            index=S.sample_space.data,
            columns=S.time.data,
        )
        S_discount = ProcessTransforms.discount(process=S, rate=0.05)

        pd.testing.assert_frame_equal(S_discount.data, expected_df, check_dtype=False)
        assert S_discount.name == "S_discount"

    def test_discount_with_gross_return_matches_discount_with_rate(self, S):
        """Test that discounting with gross_return matches discounting with the equivalent rate."""
        S_discount_rate = ProcessTransforms.discount(process=S, rate=0.05)
        S_discount_gross_return = ProcessTransforms.discount(
            process=S, gross_return=1.05
        )

        pd.testing.assert_frame_equal(
            S_discount_gross_return.data, S_discount_rate.data
        )

    def test_discounted_process_is_martingale_under_emm(self, S):
        """Test that the discounted process is a martingale under the equivalent martingale measure."""
        Q = S.emm()
        S_discount = ProcessTransforms.discount(process=S, rate=0.05)

        assert S_discount.is_martingale(prob_measure=Q)

    def test_invalid_process_raises(self):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.discount(process="not a process", rate=0.05)

    @pytest.mark.parametrize("rate", [-0.05, "not a real number"])
    def test_invalid_rate_raises(self, S, rate):
        """Test that an invalid rate raises TypeError."""
        with pytest.raises(TypeError, match="rate must be a positive real number."):
            ProcessTransforms.discount(process=S, rate=rate)

    @pytest.mark.parametrize("gross_return", [0.5, "not a real number"])
    def test_invalid_gross_return_raises(self, S, gross_return):
        """Test that an invalid gross_return raises TypeError."""
        with pytest.raises(
            TypeError, match="The gross return must be a real number greater than 1."
        ):
            ProcessTransforms.discount(process=S, gross_return=gross_return)

    def test_neither_rate_nor_gross_return_raises(self, S):
        """Test that providing neither rate nor gross_return raises TypeError."""
        with pytest.raises(
            TypeError, match="One of rate or gross_return must be given."
        ):
            ProcessTransforms.discount(process=S)

    def test_invalid_name_raises(self, S):
        """Test that a non-hashable name raises TypeError."""
        with pytest.raises(TypeError, match="name must be a hashable object or None."):
            ProcessTransforms.discount(process=S, rate=0.05, name=[1, 2])


# --------------------- test increments --------------------- #


class TestIncrements:
    @pytest.fixture
    def X(self):
        T = Time.discrete(length=2)
        return RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)

    def test_forward_increments(self, X):
        """Test computing forward increments of a process."""
        X_increments = ProcessTransforms.increments(process=X, forward=True)
        expected_df = pd.DataFrame(
            [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            index=X.sample_space.data,
            columns=pd.Index([0, 1], name="t"),
        )

        pd.testing.assert_frame_equal(X_increments.data, expected_df, check_dtype=False)
        assert X_increments.name == "X_increments"

    def test_backward_increments(self, X):
        """Test computing backward increments of a process."""
        X_increments = ProcessTransforms.increments(process=X, forward=False)
        expected_df = pd.DataFrame(
            [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            index=X.sample_space.data,
            columns=pd.Index([1, 2], name="t"),
        )

        pd.testing.assert_frame_equal(X_increments.data, expected_df, check_dtype=False)

    def test_custom_name(self, X):
        """Test that a custom name is used when provided."""
        X_increments = ProcessTransforms.increments(process=X, name="custom")

        assert X_increments.name == "custom"

    def test_invalid_process_raises(self):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.increments(process="not a process")


# --------------------- test stopped --------------------- #


class TestStopped:
    @pytest.fixture
    def S(self):
        T = Time.discrete(start=1, stop=10)
        return RandomWalk.generate(
            mode="sim",
            p=0.4,
            initial_state=10,
            index=T,
            n_trajectories=8,
            random_state=42,
            name="S",
        )

    @pytest.fixture
    def tau(self, S):
        return StoppingTime.from_filtration(
            process=S,
            mapping={
                0: inf,
                1: 3,
                2: inf,
                3: 3,
                4: 3,
                5: 7,
                6: 7,
                7: 7,
            },
        )

    def test_stopped_process_holds_value_from_stopping_time_onward(self, S, tau):
        """Test that the stopped process holds its value at the stopping time for all later times."""
        S_stopped = ProcessTransforms.stopped(process=S, stopping_time=tau)
        expected_df = pd.DataFrame(
            [
                [10, 11, 10, 11, 12, 11, 12, 13, 14, 13],
                [10, 9, 8, 8, 8, 8, 8, 8, 8, 8],
                [10, 11, 12, 13, 12, 13, 14, 15, 14, 13],
                [10, 9, 8, 8, 8, 8, 8, 8, 8, 8],
                [10, 9, 8, 8, 8, 8, 8, 8, 8, 8],
                [10, 11, 10, 9, 10, 9, 8, 8, 8, 8],
                [10, 11, 12, 11, 10, 9, 8, 8, 8, 8],
                [10, 11, 12, 11, 10, 9, 8, 8, 8, 8],
            ],
            index=S.sample_space.data,
            columns=S.time.data,
        )

        pd.testing.assert_frame_equal(S_stopped.data, expected_df, check_dtype=False)
        assert S_stopped.name == "S^tau"

    def test_custom_name(self, S, tau):
        """Test that a custom name is used when provided."""
        S_stopped = ProcessTransforms.stopped(
            process=S, stopping_time=tau, name="custom"
        )

        assert S_stopped.name == "custom"

    def test_invalid_process_raises(self, tau):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.stopped(process="not a process", stopping_time=tau)

    def test_invalid_stopping_time_raises(self, S):
        """Test that a non-StoppingTime stopping_time raises TypeError."""
        with pytest.raises(
            TypeError, match="stopping_time must be an instance of StoppingTime."
        ):
            ProcessTransforms.stopped(process=S, stopping_time="not a stopping time")

    def test_mismatched_time_index_raises(self, tau):
        """Test that a process with a different time index than the stopping time raises ValueError."""
        S2 = RandomWalk.generate(
            mode="sim",
            p=0.4,
            initial_state=10,
            index=Time.discrete(start=1, stop=5),
            n_trajectories=3,
            random_state=42,
            name="S2",
        )

        with pytest.raises(
            ValueError,
            match="The time indices of the process and stopping time must match.",
        ):
            ProcessTransforms.stopped(process=S2, stopping_time=tau)


# --------------------- test at --------------------- #


class TestAt:
    @pytest.fixture
    def S(self):
        T = Time.discrete(start=1, stop=10)
        return RandomWalk.generate(
            mode="sim",
            p=0.4,
            initial_state=10,
            index=T,
            n_trajectories=8,
            random_state=42,
            name="S",
        )

    @pytest.fixture
    def tau(self, S):
        return StoppingTime.from_filtration(
            process=S,
            mapping={
                0: inf,
                1: 3,
                2: inf,
                3: 3,
                4: 3,
                5: 7,
                6: 7,
                7: 7,
            },
        )

    def test_at_returns_value_of_process_at_stopping_time(self, S, tau):
        """Test that at returns the value of the process at the stopping time, or 0 if the stopping time is infinite."""
        S_at = ProcessTransforms.at(process=S, stopping_time=tau)
        expected_series = pd.Series(
            [0.0, 8.0, 0.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            index=S.sample_space.data,
            name="S_tau",
        )

        pd.testing.assert_series_equal(
            S_at.data, expected_series, check_dtype=False, check_names=False
        )
        assert S_at.name == "S_tau"

    def test_custom_name(self, S, tau):
        """Test that a custom name is used when provided."""
        S_at = ProcessTransforms.at(process=S, stopping_time=tau, name="custom")

        assert S_at.name == "custom"

    def test_invalid_process_raises(self, tau):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.at(process="not a process", stopping_time=tau)

    def test_invalid_stopping_time_raises(self, S):
        """Test that a non-StoppingTime stopping_time raises TypeError."""
        with pytest.raises(
            TypeError, match="stopping_time must be an instance of StoppingTime."
        ):
            ProcessTransforms.at(process=S, stopping_time="not a stopping time")

    def test_mismatched_time_index_raises(self, tau):
        """Test that a process with a different time index than the stopping time raises ValueError."""
        S2 = RandomWalk.generate(
            mode="sim",
            p=0.4,
            initial_state=10,
            index=Time.discrete(start=1, stop=5),
            n_trajectories=3,
            random_state=42,
            name="S2",
        )

        with pytest.raises(
            ValueError,
            match="The time indices of the process and stopping time must match.",
        ):
            ProcessTransforms.at(process=S2, stopping_time=tau)


# --------------------- test ito_integral --------------------- #


class TestItoIntegral:
    @pytest.fixture
    def X(self):
        T = Time.discrete(length=2)
        return RandomWalk.generate(mode="enum", p=0.6, initial_state=1, index=T)

    def test_ito_integral_with_explicit_integrand_and_integrator(self, X):
        """Test the Ito integral of an explicit integrand with respect to an integrator."""
        integral = ProcessTransforms.ito_integral(integrand=X, integrator=X)
        expected_df = pd.DataFrame(
            [
                [1, 3],
                [1, -1],
                [-1, -1],
                [-1, -1],
            ],
            index=X.sample_space.data,
            columns=X.time[:-1].data,
        )

        pd.testing.assert_frame_equal(integral.data, expected_df, check_dtype=False)


# --------------------- test is_monotonic --------------------- #


class TestIsMonotonic:
    @pytest.fixture
    def X_increasing(self):
        T = Time.discrete(start=1, length=2)
        X = IIDProcess.generate(
            mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T
        )
        return X.cumsum()

    @pytest.fixture
    def X_decreasing(self):
        T = Time.discrete(start=1, length=2)
        X = IIDProcess.generate(
            mode="enum", distribution=bernoulli(p=0.6), support=[-1, 0], index=T
        )
        return X.cumsum()

    @pytest.fixture
    def X_non_monotonic(self):
        T = Time.discrete(start=1, length=2)
        return IIDProcess.generate(
            mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T
        )

    def test_increasing_process_is_monotonic_increasing(self, X_increasing):
        """Test that a non-decreasing process is monotonic increasing."""
        assert ProcessTransforms.is_monotonic(process=X_increasing)

    def test_increasing_process_is_not_monotonic_decreasing(self, X_increasing):
        """Test that a non-decreasing process is not monotonic decreasing."""
        assert not ProcessTransforms.is_monotonic(
            process=X_increasing, increasing=False
        )

    def test_decreasing_process_is_monotonic_decreasing(self, X_decreasing):
        """Test that a non-increasing process is monotonic decreasing."""
        assert ProcessTransforms.is_monotonic(process=X_decreasing, increasing=False)

    def test_decreasing_process_is_not_monotonic_increasing(self, X_decreasing):
        """Test that a non-increasing process is not monotonic increasing."""
        assert not ProcessTransforms.is_monotonic(process=X_decreasing)

    def test_non_monotonic_process_is_neither(self, X_non_monotonic):
        """Test that a non-monotonic process is neither monotonic increasing nor decreasing."""
        assert not ProcessTransforms.is_monotonic(process=X_non_monotonic)
        assert not ProcessTransforms.is_monotonic(
            process=X_non_monotonic, increasing=False
        )

    def test_invalid_process_raises(self):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.is_monotonic(process="not a process")

    def test_invalid_increasing_raises(self, X_increasing):
        """Test that a non-boolean increasing raises TypeError."""
        with pytest.raises(TypeError, match="increasing must be a boolean value."):
            ProcessTransforms.is_monotonic(
                process=X_increasing, increasing="not a boolean"
            )


# --------------------- test to_counting_process --------------------- #


class TestToCountingProcess:
    @pytest.fixture
    def arrival_times(self):
        counts = Time.discrete(start=1, stop=5, variable_name="count")
        interarrival_times = IIDProcess.generate(
            mode="sim",
            distribution=expon(scale=0.5),
            name="interarrival_times",
            index=counts,
            n_trajectories=5,
            random_state=42,
        )
        return interarrival_times.cumsum().with_name("arrival_times")

    @pytest.fixture
    def T(self, arrival_times):
        longest_trajectory = arrival_times.max_value()
        return Time.continuous(start=0.0, stop=longest_trajectory + 0.1, num_points=6)

    def test_to_counting_process_counts_arrivals_up_to_each_time(
        self, arrival_times, T
    ):
        """Test that the counting process counts the number of arrivals up to each time point."""
        poisson = ProcessTransforms.to_counting_process(process=arrival_times, time=T)
        expected_df = pd.DataFrame(
            [
                [0.0, 0.0, 1.0, 1.0, 2.0, 5.0],
                [0.0, 1.0, 2.0, 2.0, 4.0, 5.0],
                [0.0, 2.0, 3.0, 5.0, 5.0, 5.0],
                [0.0, 4.0, 5.0, 5.0, 5.0, 5.0],
                [0.0, 2.0, 2.0, 4.0, 5.0, 5.0],
            ],
            index=arrival_times.sample_space.data,
            columns=T.data,
        )

        pd.testing.assert_frame_equal(poisson.data, expected_df, check_dtype=False)
        assert poisson.name == "arrival_times_counting"

    def test_custom_name(self, arrival_times, T):
        """Test that a custom name is used when provided."""
        poisson = ProcessTransforms.to_counting_process(
            process=arrival_times, time=T, name="custom"
        )

        assert poisson.name == "custom"

    def test_invalid_process_raises(self, T):
        """Test that a non-StochasticProcess process raises TypeError."""
        with pytest.raises(
            TypeError, match="process must be an instance of StochasticProcess."
        ):
            ProcessTransforms.to_counting_process(process="not a process", time=T)

    def test_invalid_time_raises(self, arrival_times):
        """Test that a non-Time time raises TypeError."""
        with pytest.raises(TypeError, match="time must be an instance of Time."):
            ProcessTransforms.to_counting_process(
                process=arrival_times, time="not a time"
            )

    def test_non_monotonic_process_raises(self, T):
        """Test that a non-monotonic process raises ValueError."""
        interarrival_times = IIDProcess.generate(
            mode="sim",
            distribution=expon(scale=0.5),
            name="interarrival_times",
            index=Time.discrete(start=1, stop=5, variable_name="count"),
            n_trajectories=5,
            random_state=42,
        )

        with pytest.raises(
            ValueError,
            match="The input process must be monotonic to convert to a counting process.",
        ):
            ProcessTransforms.to_counting_process(process=interarrival_times, time=T)
