import pandas as pd
import pytest

from sigalg.core.base.time import Time
from sigalg.core.probability_measures.probability_measure import ProbabilityMeasure
from sigalg.finance import BinomialPricingModel
from sigalg.processes.base.stochastic_process import StochasticProcess


class TestConstructor:
    def test_basic_construction(self):
        """Test basic construction of the BinomialPricingModel class with all required parameters."""
        S_0, u, p, r = 100, 1.1, 0.7, 0.01
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
        S_0, u, p, r = 100, 1.1, 0.7, 0.01
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
            [s, s * d, s * d**2, s * d**3],
            [s, s * d, s * d**2, s * d**2 * u],
            [s, s * d, s * d * u, s * d**2 * u],
            [s, s * d, s * d * u, s * d * u**2],
            [s, s * u, s * u * d, s * d**2 * u],
            [s, s * u, s * u * d, s * d * u**2],
            [s, s * u, s * u**2, s * d * u**2],
            [s, s * u, s * u**2, s * u**3],
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S.domain.data,
            columns=S.time.data,
        )

        pd.testing.assert_frame_equal(S.data, expected_df)

    def test_driving_process_property(self, S):
        """Test the driving_process property of the BinomialPricingModel class."""
        d = 1 / S.up_factor
        u = S.up_factor
        expected_arr = [
            [d, d, d],
            [d, d, u],
            [d, u, d],
            [d, u, u],
            [u, d, d],
            [u, d, u],
            [u, u, d],
            [u, u, u],
        ]
        expected_df = pd.DataFrame(
            expected_arr,
            index=S.domain.data,
            columns=S.time.data[1:],
        )

        assert isinstance(S.driving_process, StochasticProcess)
        pd.testing.assert_frame_equal(S.driving_process.data, expected_df)
        assert S.driving_process.time == S.time[1:]
        assert S.driving_process.name == "driving_process"


class TestRiskNeutralProbability:
    def test_risk_neutral_prob_property(self):
        """Test the risk_neutral_prob property of the BinomialPricingModel class."""
        S_0, u, p, r = 100, 1.1, 0.7, 0.01
        T = Time.discrete(length=3)
        S = BinomialPricingModel(
            initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
        ).from_enumeration()
        R = S.risk_free_gross_return
        d = 1 / u
        q = (R - d) / (u - d)
        expected_prob = [
            (1 - q) ** 3,
            q * (1 - q) ** 2,
            q * (1 - q) ** 2,
            q**2 * (1 - q),
            q * (1 - q) ** 2,
            q**2 * (1 - q),
            q**2 * (1 - q),
            q**3,
        ]
        expected_measure = ProbabilityMeasure(
            sample_space=S.domain, name="Q"
        ).from_dict(
            probabilities=dict(
                zip(S.domain.data, expected_prob, strict=False)
            ),
        )

        pd.testing.assert_series_equal(
            S.risk_neutral_prob.data, expected_measure.data
        )
        assert S.risk_neutral_prob.name == expected_measure.name
