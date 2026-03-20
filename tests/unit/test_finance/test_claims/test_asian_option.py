import numpy as np
import pytest

from sigalg.core import Time
from sigalg.finance import AsianOption, BinomialPricingModel


class TestReplicatingPortfolio:
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

    def test_replicating_portfolio_for_call_in_dense_mode(self, S):
        """Test the replicating_portfolio method for a call option in dense mode."""
        S.from_enumeration(enum_mode="dense")
        R = S.risk_free_gross_return
        K = 100
        call_option = AsianOption(pricing_model=S, strike=K, option_type="call")
        B, N, V, price = call_option.replicating_portfolio()
        expected_S_0 = (
            S.last_rv.expectation(probability_measure=S.risk_neutral_measure) / R**3
        ).item()
        expected_price = (
            call_option.payout.expectation(probability_measure=S.risk_neutral_measure)
            / R**3
        ).item()

        assert V.last_rv == call_option.payout
        assert np.abs(expected_S_0 - S.initial_price) < 1e-8
        assert np.abs(expected_price - price) < 1e-8

        for t in range(3):
            assert B[t] + S[t] * N[t] == V[t]  # test value process is correct
            assert R * B[t] + S[t + 1] * N[t] == V[t + 1]  # test self-financing

        assert B.is_adapted(filtration=S.natural_filtration)
        assert N.is_adapted(filtration=S.natural_filtration)
        assert V.is_adapted(filtration=S.natural_filtration)

        assert V.discount(rate=S.risk_free_rate).is_martingale(
            probability_measure=S.risk_neutral_measure
        )

    def test_replicating_portfolio_for_put_in_dense_mode(self, S):
        """Test the replicating_portfolio method for a put option in dense mode."""
        S.from_enumeration(enum_mode="dense")
        R = S.risk_free_gross_return
        K = 100
        put_option = AsianOption(pricing_model=S, strike=K, option_type="put")
        B, N, V, price = put_option.replicating_portfolio()
        expected_S_0 = (
            S.last_rv.expectation(probability_measure=S.risk_neutral_measure) / R**3
        ).item()
        expected_price = (
            put_option.payout.expectation(probability_measure=S.risk_neutral_measure)
            / R**3
        ).item()

        assert V.last_rv == put_option.payout
        assert np.abs(expected_S_0 - S.initial_price) < 1e-8
        assert np.abs(expected_price - price) < 1e-8

        for t in range(3):
            assert B[t] + S[t] * N[t] == V[t]  # test value process is correct
            assert R * B[t] + S[t + 1] * N[t] == V[t + 1]  # test self-financing

        assert B.is_adapted(filtration=S.natural_filtration)
        assert N.is_adapted(filtration=S.natural_filtration)
        assert V.is_adapted(filtration=S.natural_filtration)

        assert V.discount(rate=S.risk_free_rate).is_martingale(
            probability_measure=S.risk_neutral_measure
        )
