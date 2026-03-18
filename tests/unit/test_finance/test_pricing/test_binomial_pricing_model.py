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


class TestPriceAndDrivingProcess:
    @pytest.fixture
    def S(self):
        S_0 = 100
        u = 1.1
        p = 0.7
        r = 0.01
        T = Time.discrete(length=3)
        return BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        ).from_enumeration()

    def test_price_process_property(self, S):
        """Test the price_process property of the BinomialPricingModel class."""
        d = 1 / S.up_factor
        s = S.initial_price
        u = S.up_factor
        expected_arr = [
            [s, s * u, s * u**2, s * u**3],
            [s, s * d, s * u * d, s * u**2 * d],
            [s, s * d, s * d**2, s * u * d**2],
            [s, s * d, s * d**2, s * d**3],
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S.domain.data,
            columns=S.time.data,
        )

        pd.testing.assert_frame_equal(S.data, expected_df, check_dtype=False)


class TestRiskNeutralMeasure:
    def test_risk_neutral_measure_property(self):
        """Test the risk_neutral_measure property of the BinomialPricingModel class."""
        S_0 = 100
        u = 1.1
        r = 0.01
        p = 0.7
        T = Time.discrete(length=3)
        S = BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        ).from_enumeration()
        q = S.risk_neutral_prob
        expected_prob = [
            q**3,
            comb(3, 1) * q**2 * (1 - q),
            comb(3, 2) * q * (1 - q) ** 2,
            (1 - q) ** 3,
        ]
        expected_measure = ProbabilityMeasure(
            sample_space=S.domain, name="Q"
        ).from_dict(
            probabilities=dict(zip(S.domain.data, expected_prob, strict=False)),
        )

        np.testing.assert_allclose(
            S.risk_neutral_measure.data.values,
            expected_measure.data.values,
        )
