import numpy as np
import pandas as pd
import pytest

from sigalg.core import ProbabilityMeasure, ProbabilitySpace
from sigalg.finance import TrinomialPricingModel

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters"""
        S = TrinomialPricingModel()
        prob_space = ProbabilitySpace()

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
        assert S.middle_factor is None
        assert S.down_factor is None


class TestGenerate:
    @pytest.fixture
    def S_enum(self):
        S_0 = 4
        u = 1.2
        m = 1.1
        d = 0.9
        p_u = 0.6
        p_d = 0.1
        r = 0.01

        return TrinomialPricingModel.generate(
            mode="enum",
            initial_price=S_0,
            up_factor=u,
            middle_factor=m,
            down_factor=d,
            up_prob=p_u,
            down_prob=p_d,
            risk_free_rate=r,
            length=2,
        )

    @pytest.fixture
    def S_sim(self):
        S_0 = 4
        u = 1.2
        m = 1.1
        d = 0.9
        p_u = 0.6
        p_d = 0.1
        r = 0.01
        n_trajectories = 4
        random_state = 42

        return TrinomialPricingModel.generate(
            mode="sim",
            initial_price=S_0,
            up_factor=u,
            middle_factor=m,
            down_factor=d,
            up_prob=p_u,
            down_prob=p_d,
            risk_free_rate=r,
            length=3,
            n_trajectories=n_trajectories,
            random_state=random_state,
        )

    def test_trajectories_in_enum_mode(self, S_enum):
        """Test that the trajectories are correct in enumeration mode."""
        S_0 = S_enum.initial_price
        u = S_enum.up_factor
        m = S_enum.middle_factor
        d = S_enum.down_factor
        expected_arr = S_0 * np.array(
            [
                [1, u, u**2],  # uu
                [1, u, u * m],  # um
                [1, u, u * d],  # ud
                [1, m, m * u],  # mu
                [1, m, m**2],  # mm
                [1, m, m * d],  # md
                [1, d, d * u],  # du
                [1, d, d * m],  # dm
                [1, d, d**2],  # dd
            ]
        )
        expected_df = pd.DataFrame(
            expected_arr,
            index=S_enum.sample_space.data,
            columns=S_enum.time.data,
        )

        pd.testing.assert_frame_equal(S_enum.data, expected_df, check_dtype=False)

    def test_trajectories_in_simulation_mode(self, S_sim):
        """Test that the trajectories are correct in simulation mode."""
        S_0 = S_sim.initial_price
        d = S_sim.down_factor
        m = S_sim.middle_factor
        u = S_sim.up_factor
        expected_arr = S_0 * np.array(
            [
                [1, m, m**2, u * m**2],  # mmu
                [1, d, m * d, u * m * d],  # dmu
                [1, u, u * m, u * m**2],  # umm
                [1, u, u**2, u**3],  # uuu
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
    def S_enum(self):
        S_0 = 4
        u = 1.2
        m = 1.1
        d = 0.9
        p_u = 0.6
        p_d = 0.1
        r = 0.01

        return TrinomialPricingModel.generate(
            mode="enum",
            initial_price=S_0,
            up_factor=u,
            middle_factor=m,
            down_factor=d,
            up_prob=p_u,
            down_prob=p_d,
            risk_free_rate=r,
            length=2,
        )

    def test_in_enum_mode(self, S_enum):
        """Test that the measure in 'enum' mode is correct."""
        p_u = S_enum.up_prob
        p_m = S_enum.middle_prob
        p_d = S_enum.down_prob
        expected_mapping = np.array(
            [
                p_u**2,
                p_u * p_m,
                p_u * p_d,
                p_m * p_u,
                p_m**2,
                p_m * p_d,
                p_d * p_u,
                p_d * p_m,
                p_d**2,
            ]
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_enum.sample_space, mapping=expected_mapping
        )

        assert S_enum.prob_measure == expected_measure
