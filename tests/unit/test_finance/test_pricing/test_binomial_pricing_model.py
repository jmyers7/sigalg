from math import comb

import numpy as np
import pandas as pd
import pytest

from sigalg.core.base.time import Time
from sigalg.core.probability_measures.probability_measure import ProbabilityMeasure
from sigalg.finance import BinomialPricingModel


class TestConstructor:
    def test_basic_construction(self):
        """Test basic construction of the BinomialPricingModel class with all required parameters."""
        S_0 = 100
        u = 1.1
        p = 0.7
        r = 0.01
        T = Time.discrete(length=10)
        S = BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        )

        assert S.initial_price == S_0
        assert S.up_factor == u
        assert S.up_prob == p
        assert S.risk_free_rate == r
        assert S.risk_free_gross_return == 1 + r


class TestPriceProcess:
    @pytest.fixture
    def S(self):
        S_0 = 100
        u = 1.1
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)
        return BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        )

    def test_price_process_in_sparse_mode(self, S):
        """Test the price_process property of the BinomialPricingModel class in sparse mode."""
        S.from_enumeration(enum_mode="sparse")
        S_0 = S.initial_price
        d = S.down_factor
        u = S.up_factor
        expected_arr = [
            [S_0, S_0 * u, S_0 * u**2, S_0 * u**3],
            [S_0, S_0 * d, S_0 * u * d, S_0 * u**2 * d],
            [S_0, S_0 * d, S_0 * d**2, S_0 * u * d**2],
            [S_0, S_0 * d, S_0 * d**2, S_0 * d**3],
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S.domain.data,
            columns=S.time.data,
        )

        pd.testing.assert_frame_equal(S.data, expected_df, check_dtype=False)

    def test_price_process_in_dense_mode(self, S):
        """Test the price_process property of the BinomialPricingModel class in dense mode."""
        S.from_enumeration(enum_mode="dense")
        S_0 = S.initial_price
        d = S.down_factor
        u = S.up_factor
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
            index=S.domain.data,
            columns=S.time.data,
        )

        pd.testing.assert_frame_equal(S.data, expected_df, check_dtype=False)

    def test_discounted_price_process_is_martingale(self, S):
        """Test that the discounted price process is a martingale under the risk-neutral measure."""
        S.from_enumeration(enum_mode="dense")

        assert S.discount(rate=S.risk_free_rate).is_martingale(
            probability_measure=S.risk_neutral_measure
        )

    def test_driving_process(self, S):
        """Test the driving_process property of the BinomialPricingModel class."""
        S.from_enumeration(enum_mode="dense")
        Z = S.driving_process
        u = S.up_factor
        d = S.down_factor

        expected_arr = [
            [u, u, u],  # uuu
            [u, u, d],  # uud
            [u, d, u],  # udu
            [u, d, d],  # udd
            [d, u, u],  # duu
            [d, u, d],  # dud
            [d, d, u],  # ddu
            [d, d, d],  # ddd
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S.domain.data,
            columns=S.time.data[1:],
        )

        pd.testing.assert_frame_equal(Z.data, expected_df, check_dtype=False)


class TestProbabilityMeasures:
    @pytest.fixture
    def S(self):
        S_0 = 100
        u = 1.1
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)
        return BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        )

    def test_risk_neutral_measure_in_sparse_mode(self, S):
        """Test the risk_neutral_measure property on sparse price trajectories."""
        S.from_enumeration(enum_mode="sparse")
        q = S.risk_neutral_prob
        expected_prob = [
            q**3,
            comb(3, 1) * q**2 * (1 - q),
            comb(3, 2) * q * (1 - q) ** 2,
            (1 - q) ** 3,
        ]
        expected_probs_dict = dict(zip(S.domain.data, expected_prob, strict=False))
        expected_measure = ProbabilityMeasure(
            sample_space=S.domain, name="Q"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S.risk_neutral_measure.data.values,
            expected_measure.data.values,
        )

    def test_risk_neutral_measure_in_dense_mode(self, S):
        """Test the risk_neutral_measure property on dense price trajectories."""
        S.from_enumeration(enum_mode="dense")
        q = S.risk_neutral_prob
        expected_prob = [
            q**3,  # uuu
            q**2 * (1 - q),  # uud
            q**2 * (1 - q),  # udu
            q * (1 - q) ** 2,  # udd
            q**2 * (1 - q),  # duu
            q * (1 - q) ** 2,  # dud
            q * (1 - q) ** 2,  # ddu
            (1 - q) ** 3,  # ddd
        ]
        expected_probs_dict = dict(zip(S.domain.data, expected_prob, strict=False))
        expected_measure = ProbabilityMeasure(
            sample_space=S.domain, name="Q"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S.risk_neutral_measure.data.values,
            expected_measure.data.values,
        )

    def test_real_world_measure_in_sparse_mode(self, S):
        """Test the real_world_measure property on sparse price trajectories."""
        S.from_enumeration(enum_mode="sparse")
        p = S.up_prob
        expected_prob = [
            p**3,
            comb(3, 1) * p**2 * (1 - p),
            comb(3, 2) * p * (1 - p) ** 2,
            (1 - p) ** 3,
        ]
        expected_probs_dict = dict(zip(S.domain.data, expected_prob, strict=False))
        expected_measure = ProbabilityMeasure(
            sample_space=S.domain, name="P"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S.probability_measure.data.values,
            expected_measure.data.values,
        )

    def test_real_world_measure_in_dense_mode(self, S):
        """Test the real_world_measure property on dense price trajectories."""
        S.from_enumeration(enum_mode="dense")
        p = S.up_prob
        expected_prob = [
            p**3,  # uuu
            p**2 * (1 - p),  # uud
            p**2 * (1 - p),  # udu
            p * (1 - p) ** 2,  # udd
            p**2 * (1 - p),  # duu
            p * (1 - p) ** 2,  # dud
            p * (1 - p) ** 2,  # ddu
            (1 - p) ** 3,  # ddd
        ]
        expected_probs_dict = dict(zip(S.domain.data, expected_prob, strict=False))
        expected_measure = ProbabilityMeasure(
            sample_space=S.domain, name="P"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S.probability_measure.data.values,
            expected_measure.data.values,
        )
