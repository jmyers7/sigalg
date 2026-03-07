import pandas as pd
import pytest

from sigalg.core.base.time import Time
from sigalg.core.probability_measures.probability_measure import ProbabilityMeasure
from sigalg.finance import BinomialPricingModel
from sigalg.processes.base.stochastic_process import StochasticProcess


class TestConstructor:
    def test_basic_construction(self):
        """Test basic construction of the BinomialPricingModel class with all required parameters."""
        s, u, r, length = 100, 1.1, 0.01, 10
        model = BinomialPricingModel(
            initial_price=s, up_factor=u, risk_free_rate=r, length=length
        )

        assert model.initial_price == s
        assert model.up_factor == u
        assert model.risk_free_rate == r
        assert model.length == length
        assert model.risk_free_return == 1 + r


class TestTimeProperty:
    @pytest.fixture
    def model(self):
        s, u, r, length = 100, 1.1, 0.01, 10
        return BinomialPricingModel(
            initial_price=s, up_factor=u, risk_free_rate=r, length=length
        )

    def test_time_property(self, model):
        """Test the time property of the BinomialPricingModel class."""
        expected_time = Time.discrete(length=model.length)

        assert isinstance(model.time, Time)
        assert model.time == expected_time

    def test_time_setter(self, model):
        """Test the time setter of the BinomialPricingModel class."""
        new_time = Time.discrete(length=5)
        model.time = new_time

        assert model.time == new_time
        assert model._price_process is None


class TestPriceAndDrivingProcess:
    @pytest.fixture
    def model(self):
        s, u, r, length = 100, 1.1, 0.01, 3
        return BinomialPricingModel(
            initial_price=s, up_factor=u, risk_free_rate=r, length=length
        )

    def test_price_process_property(self, model):
        """Test the price_process property of the BinomialPricingModel class."""
        price_process = model.price_process

        d = 1 / model.up_factor
        s = model.initial_price
        u = model.up_factor
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
            index=price_process.domain.data,
            columns=price_process.time.data,
        )

        assert isinstance(price_process, StochasticProcess)
        pd.testing.assert_frame_equal(price_process.data, expected_df)
        assert price_process.time == model.time
        assert price_process.name == "price_process"

    def test_driving_process_property(self, model):
        """Test the driving_process property of the BinomialPricingModel class."""
        driving_process = model.driving_process

        d = 1 / model.up_factor
        u = model.up_factor
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
            index=driving_process.domain.data,
            columns=driving_process.time.data,
        )

        assert isinstance(driving_process, StochasticProcess)
        pd.testing.assert_frame_equal(driving_process.data, expected_df)
        assert driving_process.time == model.time[1:]
        assert driving_process.name == "driving_process"


class TestRiskNeutralProbability:
    def test_risk_neutral_prob_property(self):
        """Test the risk_neutral_prob property of the BinomialPricingModel class."""
        s, u, r, length = 100, 1.1, 0.01, 3
        model = BinomialPricingModel(
            initial_price=s, up_factor=u, risk_free_rate=r, length=length
        )
        R = model.risk_free_return
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
            sample_space=model.sample_space, name="risk_neutral"
        ).from_dict(
            probabilities=dict(
                zip(model.sample_space.data, expected_prob, strict=False)
            ),
        )

        pd.testing.assert_series_equal(
            model.risk_neutral_prob.data, expected_measure.data
        )
        assert model.risk_neutral_prob.name == expected_measure.name
