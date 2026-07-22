import numpy as np
import pandas as pd
import pytest

from sigalg.core import ProbabilityMeasure, MeasureSpace, Time
from sigalg.finance import BinomialPricingModel

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters"""
        S = BinomialPricingModel()
        prob_space = MeasureSpace()

        assert S.data is None
        assert S.atom_data is None
        assert S.components is None
        assert S.time is None
        assert S.generated_sig_alg is None
        assert S.prob_space == prob_space
        assert S.sample_space is None
        assert S.sig_alg is None
        assert S.prob_measure is None
        assert S.range is None
        assert S.is_discrete_time is None
        assert S.n_trajectories is None
        assert S.natural_filtration is None
        assert S.last_rv is None
        assert S.random_state is None
        assert S.mode is None
        assert S.length is None
        assert S.initial_price is None
        assert S.risk_free_rate is None
        assert S.risk_free_gross_return is None
        assert S.up_prob is None
        assert S.down_prob is None
        assert S.up_factor is None
        assert S.down_factor is None
        assert S.enum_mode is None
        assert S.sparse_price_array is None


class TestGenerate:
    @pytest.fixture
    def S_dense(self):
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)

        return BinomialPricingModel.generate(
            mode="enum",
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            index=T,
            enum_mode="dense",
        )

    @pytest.fixture
    def S_sparse(self):
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)

        return BinomialPricingModel.generate(
            mode="enum",
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            index=T,
            enum_mode="sparse",
        )

    @pytest.fixture
    def S_sim(self):
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        n_trajectories = 4
        random_state = 42
        T = Time.discrete(length=3)

        return BinomialPricingModel.generate(
            mode="sim",
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            index=T,
            n_trajectories=n_trajectories,
            random_state=random_state,
        )

    def test_trajectories_in_sparse_mode(self, S_sparse):
        """Test that the trajectories are correct in sparse enumeration mode."""
        S_0 = S_sparse.initial_price
        d = S_sparse.down_factor
        u = S_sparse.up_factor
        expected_arr = [
            [S_0, S_0 * u, S_0 * u**2, S_0 * u**3],
            [S_0, S_0 * d, S_0 * u * d, S_0 * u**2 * d],
            [S_0, S_0 * d, S_0 * d**2, S_0 * u * d**2],
            [S_0, S_0 * d, S_0 * d**2, S_0 * d**3],
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S_sparse.sample_space.data,
            columns=S_sparse.time.data,
        )

        pd.testing.assert_frame_equal(S_sparse.data, expected_df, check_dtype=False)

    def test_trajectories_in_dense_mode(self, S_dense):
        """Test the price_process property of a recombining BinomialPricingModel class in dense mode."""
        S_0 = S_dense.initial_price
        d = S_dense.down_factor
        u = S_dense.up_factor
        expected_arr = [
            [S_0, S_0 * u, S_0 * u**2, S_0 * u**3],  # uuu
            [S_0, S_0 * u, S_0 * u**2, S_0 * u**2 * d],  # uud
            [S_0, S_0 * u, S_0 * u * d, S_0 * u**2 * d],  # udu
            [S_0, S_0 * u, S_0 * u * d, S_0 * u * d**2],  # udd
            [S_0, S_0 * d, S_0 * u * d, S_0 * u**2 * d],  # duu
            [S_0, S_0 * d, S_0 * u * d, S_0 * u * d**2],  # dud
            [S_0, S_0 * d, S_0 * d**2, S_0 * u * d**2],  # ddu
            [S_0, S_0 * d, S_0 * d**2, S_0 * d**3],  # ddd
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S_dense.sample_space.data,
            columns=S_dense.time.data,
        )

        pd.testing.assert_frame_equal(S_dense.data, expected_df, check_dtype=False)

    def test_trajectories_in_simulation_mode(self, S_sim):
        """Test that the trajectories are correct in simulation mode."""
        S_0 = S_sim.initial_price
        d = S_sim.down_factor
        u = S_sim.up_factor
        expected_arr = S_0 * np.array(
            [
                [1, d, d * u, d**2 * u],  # dud
                [1, u, u**2, d * u**2],  # uud
                [1, d, d**2, d**2 * u],  # ddu
                [1, u, u**2, d * u**2],  # uud
            ]
        )
        expected_df = pd.DataFrame(
            expected_arr,
            index=S_sim.sample_space.data,
            columns=S_sim.time.data,
        )

        pd.testing.assert_frame_equal(S_sim.data, expected_df, check_dtype=False)


# --------------------- test properties --------------------- #


class TestProbMeasure:
    @pytest.fixture
    def S_dense(self):
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        T = Time.discrete(length=2)

        return BinomialPricingModel.generate(
            mode="enum",
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            index=T,
            enum_mode="dense",
        )

    @pytest.fixture
    def S_sparse(self):
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        T = Time.discrete(length=2)

        return BinomialPricingModel.generate(
            mode="enum",
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            index=T,
            enum_mode="sparse",
        )

    def test_in_dense_enum_mode(self, S_dense):
        """Test that the measure in dense 'enum' mode is correct."""
        p_u = S_dense.up_prob
        p_d = S_dense.down_prob
        expected_mapping = np.array(
            [
                p_u**2,
                p_u * p_d,
                p_d * p_u,
                p_d**2,
            ]
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_dense.sample_space, mapping=expected_mapping
        )

        assert S_dense.prob_measure == expected_measure

    def test_in_sparse_enum_mode(self, S_sparse):
        """Test that the measure in sparse 'enum' mode is correct."""
        p_u = S_sparse.up_prob
        p_d = S_sparse.down_prob
        expected_mapping = np.array(
            [
                p_u**2,
                2 * p_d * p_u,
                p_d**2,
            ]
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_sparse.sample_space, mapping=expected_mapping
        )

        assert S_sparse.prob_measure == expected_measure
