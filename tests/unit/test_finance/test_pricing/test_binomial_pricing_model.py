from math import comb

import numpy as np
import pandas as pd
import pytest

from sigalg.core.base.time import Time
from sigalg.core.probability_measures.probability_measure import ProbabilityMeasure
from sigalg.finance import BinomialPricingModel


class TestConstructor:
    def test_basic_recombining_construction(self):
        """Test basic construction of a recombining binomial pricing model."""
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
        assert S.down_factor == 1 / u
        assert S.up_prob == p
        assert S.risk_free_rate == r
        assert S.risk_free_gross_return == 1 + r
        assert S.is_recombining is True

    def test_non_recombining_construction(self):
        """Test construction of a non-recombining binomial pricing model."""
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        T = Time.discrete(length=10)
        S = BinomialPricingModel(
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            time=T,
        )

        assert S.initial_price == S_0
        assert S.up_factor == u
        assert S.down_factor == d
        assert S.up_prob == p
        assert S.risk_free_rate == r
        assert S.risk_free_gross_return == 1 + r
        assert S.is_recombining is False


class TestPriceProcess:
    @pytest.fixture
    def S_recombining(self):
        S_0 = 100
        u = 1.1
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)
        return BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        )

    @pytest.fixture
    def S_non_recombining(self):
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)
        return BinomialPricingModel(
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            time=T,
        )

    def test_price_process_in_sparse_mode_recombining(self, S_recombining):
        """Test the price_process property of a recombining BinomialPricingModel class in sparse mode."""
        S_recombining.from_enumeration(enum_mode="sparse")
        S_0 = S_recombining.initial_price
        d = S_recombining.down_factor
        u = S_recombining.up_factor
        expected_arr = [
            [S_0, S_0 * u, S_0 * u**2, S_0 * u**3],
            [S_0, S_0 * d, S_0 * u * d, S_0 * u**2 * d],
            [S_0, S_0 * d, S_0 * d**2, S_0 * u * d**2],
            [S_0, S_0 * d, S_0 * d**2, S_0 * d**3],
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S_recombining.domain.data,
            columns=S_recombining.time.data,
        )

        pd.testing.assert_frame_equal(
            S_recombining.data, expected_df, check_dtype=False
        )

    def test_price_process_in_sparse_mode_non_recombining_raises(
        self, S_non_recombining
    ):
        """Test that accessing the price_process in sparse mode for a non-recombining model raises an error."""
        with pytest.raises(TypeError):
            S_non_recombining.from_enumeration(enum_mode="sparse")

    def test_price_process_in_dense_mode_recombining(self, S_recombining):
        """Test the price_process property of a recombining BinomialPricingModel class in dense mode."""
        S_recombining.from_enumeration(enum_mode="dense")
        S_0 = S_recombining.initial_price
        d = S_recombining.down_factor
        u = S_recombining.up_factor
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
            index=S_recombining.domain.data,
            columns=S_recombining.time.data,
        )

        pd.testing.assert_frame_equal(
            S_recombining.data, expected_df, check_dtype=False
        )

    def test_price_process_in_dense_mode_non_recombining(self, S_non_recombining):
        """Test the price_process property of a non-recombining BinomialPricingModel class in dense mode."""
        S_non_recombining.from_enumeration(enum_mode="dense")
        S_0 = S_non_recombining.initial_price
        d = S_non_recombining.down_factor
        u = S_non_recombining.up_factor
        expected_arr = [
            [S_0, S_0 * u, S_0 * u**2, S_0 * u**3],  # uuu
            [S_0, S_0 * u, S_0 * u**2, S_0 * u**2 * d],  # uud
            [S_0, S_0 * u, S_0 * u * d, S_0 * u**2 * d],  # udu
            [S_0, S_0 * u, S_0 * u * d, S_0 * u * d**2],  # udd
            [S_0, S_0 * d, S_0 * d * u, S_0 * d * u**2],  # duu
            [S_0, S_0 * d, S_0 * d * u, S_0 * d * u * d],  # dud
            [S_0, S_0 * d, S_0 * d**2, S_0 * d**2 * u],  # ddu
            [S_0, S_0 * d, S_0 * d**2, S_0 * d**3],  # ddd
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S_non_recombining.domain.data,
            columns=S_non_recombining.time.data,
        )

        pd.testing.assert_frame_equal(
            S_non_recombining.data, expected_df, check_dtype=False
        )

    def test_discounted_price_process_is_martingale(self, S_recombining):
        """Test that the discounted price process is a martingale under the risk-neutral measure."""
        S_recombining.from_enumeration(enum_mode="dense")

        assert S_recombining.discount(rate=S_recombining.risk_free_rate).is_martingale(
            probability_measure=S_recombining.emms
        )

    def test_driving_process_recombining(self, S_recombining):
        """Test the driving_process property of a recombining BinomialPricingModel class."""
        S_recombining.from_enumeration(enum_mode="dense")
        Z = S_recombining.driving_process
        u = S_recombining.up_factor
        d = S_recombining.down_factor

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
            index=S_recombining.domain.data,
            columns=S_recombining.time.data[1:],
        )

        pd.testing.assert_frame_equal(Z.data, expected_df, check_dtype=False)

    def test_driving_process_non_recombining(self, S_non_recombining):
        """Test the driving_process property of a non-recombining BinomialPricingModel class."""
        S_non_recombining.from_enumeration(enum_mode="dense")
        Z = S_non_recombining.driving_process
        u = S_non_recombining.up_factor
        d = S_non_recombining.down_factor

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
            index=S_non_recombining.domain.data,
            columns=S_non_recombining.time.data[1:],
        )

        pd.testing.assert_frame_equal(Z.data, expected_df, check_dtype=False)


class TestProbabilityMeasures:
    @pytest.fixture
    def S_recombining(self):
        S_0 = 100
        u = 1.1
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)
        return BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        )

    @pytest.fixture
    def S_non_recombining(self):
        S_0 = 100
        u = 1.1
        d = 0.9
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)
        return BinomialPricingModel(
            initial_price=S_0,
            up_factor=u,
            down_factor=d,
            up_prob=p,
            risk_free_rate=r,
            time=T,
        )

    def test_emms_in_sparse_mode_recombining(self, S_recombining):
        """Test the emms property on sparse price trajectories over a recombining tree."""
        S_recombining.from_enumeration(enum_mode="sparse")
        q = S_recombining.risk_neutral_probs[0]
        expected_prob = [
            q**3,
            comb(3, 1) * q**2 * (1 - q),
            comb(3, 2) * q * (1 - q) ** 2,
            (1 - q) ** 3,
        ]
        expected_probs_dict = dict(
            zip(S_recombining.domain.data, expected_prob, strict=False)
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_recombining.domain, name="Q"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S_recombining.emms.data.values,
            expected_measure.data.values,
        )

    def test_emms_in_dense_mode_recombining(self, S_recombining):
        """Test the emms property on dense price trajectories over a recombining tree."""
        S_recombining.from_enumeration(enum_mode="dense")
        q = S_recombining.risk_neutral_probs[0]
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
        expected_probs_dict = dict(
            zip(S_recombining.domain.data, expected_prob, strict=False)
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_recombining.domain, name="Q"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S_recombining.emms.data.values,
            expected_measure.data.values,
        )

    def test_emms_in_dense_mode_non_recombining(
        self, S_non_recombining
    ):
        """Test the emms property on dense price trajectories over a non-recombining tree."""
        S_non_recombining.from_enumeration(enum_mode="dense")
        q = S_non_recombining.risk_neutral_probs[0]
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
        expected_probs_dict = dict(
            zip(S_non_recombining.domain.data, expected_prob, strict=False)
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_non_recombining.domain, name="Q"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S_non_recombining.emms.data.values,
            expected_measure.data.values,
        )

    def test_real_world_measure_in_sparse_mode_recombining(self, S_recombining):
        """Test the real_world_measure property on sparse price trajectories over a recombining tree."""
        S_recombining.from_enumeration(enum_mode="sparse")
        p = S_recombining.up_prob
        expected_prob = [
            p**3,
            comb(3, 1) * p**2 * (1 - p),
            comb(3, 2) * p * (1 - p) ** 2,
            (1 - p) ** 3,
        ]
        expected_probs_dict = dict(
            zip(S_recombining.domain.data, expected_prob, strict=False)
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_recombining.domain, name="P"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S_recombining.probability_measure.data.values,
            expected_measure.data.values,
        )

    def test_real_world_measure_in_dense_mode_non_recombining(self, S_non_recombining):
        """Test the real_world_measure property on dense price trajectories over a non-recombining tree."""
        S_non_recombining.from_enumeration(enum_mode="dense")
        p = S_non_recombining.up_prob
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
        expected_probs_dict = dict(
            zip(S_non_recombining.domain.data, expected_prob, strict=False)
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_non_recombining.domain, name="P"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S_non_recombining.probability_measure.data.values,
            expected_measure.data.values,
        )

    def test_real_world_measure_in_dense_mode_recombining(self, S_recombining):
        """Test the real_world_measure property on dense price trajectories over a recombining tree."""
        S_recombining.from_enumeration(enum_mode="dense")
        p = S_recombining.up_prob
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
        expected_probs_dict = dict(
            zip(S_recombining.domain.data, expected_prob, strict=False)
        )
        expected_measure = ProbabilityMeasure(
            sample_space=S_recombining.domain, name="P"
        ).from_dict(expected_probs_dict)

        np.testing.assert_allclose(
            S_recombining.probability_measure.data.values,
            expected_measure.data.values,
        )
