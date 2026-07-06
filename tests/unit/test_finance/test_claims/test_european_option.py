# import numpy as np
# import pytest

# from sigalg.core import Time
# from sigalg.finance import BinomialPricingModel, EuropeanOption


# class TestReplicatingPortfolio:
#     @pytest.fixture
#     def S(self):
#         S_0 = 100
#         u = 1.1
#         p = 0.7
#         r = 0.01
#         T = Time.discrete(length=3)
#         return BinomialPricingModel(
#             initial_price=S_0, up_factor=u, up_prob=p, risk_free_rate=r, time=T
#         )

#     def test_replicating_portfolio_for_call_in_sparse_mode(self, S):
#         """Test the replicating_portfolio method for a call option in sparse mode."""
#         S.from_enumeration(enum_mode="sparse")
#         R = S.risk_free_gross_return
#         K = 100
#         call_option = EuropeanOption(pricing_model=S, strike=K, option_type="call")
#         B, Delta, V, price, tau = S.replicating_portfolio(claim=call_option)
#         expected_S_0 = (
#             S.last_rv.expectation(prob_measure=S.emms) / R**3
#         ).item()
#         expected_price = (
#             call_option.payoff.expectation(prob_measure=S.emms)
#             / R**3
#         ).item()

#         assert V.last_rv == call_option.payoff
#         assert np.abs(expected_S_0 - S.initial_price) < 1e-8
#         assert np.abs(expected_price - price) < 1e-8

#         for t in range(3):
#             assert B[t] + S[t] * Delta[t] == V[t]  # test value process is correct
#             assert R * B[t] + S[t + 1] * Delta[t] == V[t + 1]  # test self-financing

#         Delta_2 = Delta[2].data.values[:-1]
#         V_3 = V[3].data.values
#         S_3 = S[3].data.values

#         for i in range(3):
#             assert np.allclose(
#                 Delta_2[i], (V_3[i] - V_3[i + 1]) / (S_3[i] - S_3[i + 1])
#             )  # test delta hedging formula

#     def test_replicating_portfolio_for_call_in_dense_mode(self, S):
#         """Test the replicating_portfolio method for a call option in dense mode."""
#         S.from_enumeration(enum_mode="dense")
#         R = S.risk_free_gross_return
#         K = 100
#         call_option = EuropeanOption(pricing_model=S, strike=K, option_type="call")
#         B, Delta, V, price, tau = S.replicating_portfolio(claim=call_option)
#         expected_S_0 = (
#             S.last_rv.expectation(prob_measure=S.emms) / R**3
#         ).item()
#         expected_price = (
#             call_option.payoff.expectation(prob_measure=S.emms)
#             / R**3
#         ).item()

#         assert V.last_rv == call_option.payoff
#         assert np.abs(expected_S_0 - S.initial_price) < 1e-8
#         assert np.abs(expected_price - price) < 1e-8

#         for t in range(3):
#             assert B[t] + S[t] * Delta[t] == V[t]  # test value process is correct
#             assert R * B[t] + S[t + 1] * Delta[t] == V[t + 1]  # test self-financing

#         assert B.is_adapted(filtration=S.natural_filtration)
#         assert Delta.is_adapted(filtration=S.natural_filtration)
#         assert V.is_adapted(filtration=S.natural_filtration)

#         assert V.discount(rate=S.risk_free_rate).is_martingale(
#             prob_measure=S.emms
#         )

#     def test_replicating_portfolio_for_put_in_sparse_mode(self, S):
#         """Test the replicating_portfolio method for a put option in sparse mode."""
#         S.from_enumeration(enum_mode="sparse")
#         R = S.risk_free_gross_return
#         K = 100
#         put_option = EuropeanOption(pricing_model=S, strike=K, option_type="put")
#         B, Delta, V, price, tau = S.replicating_portfolio(claim=put_option)
#         expected_S_0 = (
#             S.last_rv.expectation(prob_measure=S.emms) / R**3
#         ).item()
#         expected_price = (
#             put_option.payoff.expectation(prob_measure=S.emms)
#             / R**3
#         ).item()

#         assert V.last_rv == put_option.payoff
#         assert np.abs(expected_S_0 - S.initial_price) < 1e-8
#         assert np.abs(expected_price - price) < 1e-8

#         for t in range(3):
#             assert B[t] + S[t] * Delta[t] == V[t]  # test value process is correct
#             assert R * B[t] + S[t + 1] * Delta[t] == V[t + 1]  # test self-financing

#         Delta_2 = Delta[2].data.values[:-1]
#         V_3 = V[3].data.values
#         S_3 = S[3].data.values

#         for i in range(3):
#             assert np.allclose(
#                 Delta_2[i], (V_3[i] - V_3[i + 1]) / (S_3[i] - S_3[i + 1])
#             )  # test delta hedging formula

#     def test_replicating_portfolio_for_put_in_dense_mode(self, S):
#         """Test the replicating_portfolio method for a put option in dense mode."""
#         S.from_enumeration(enum_mode="dense")
#         R = S.risk_free_gross_return
#         K = 100
#         put_option = EuropeanOption(pricing_model=S, strike=K, option_type="put")
#         B, Delta, V, price, tau = S.replicating_portfolio(claim=put_option)
#         expected_S_0 = (
#             S.last_rv.expectation(prob_measure=S.emms) / R**3
#         ).item()
#         expected_price = (
#             put_option.payoff.expectation(prob_measure=S.emms)
#             / R**3
#         ).item()

#         assert V.last_rv == put_option.payoff
#         assert np.abs(expected_S_0 - S.initial_price) < 1e-8
#         assert np.abs(expected_price - price) < 1e-8

#         for t in range(3):
#             assert B[t] + S[t] * Delta[t] == V[t]  # test value process is correct
#             assert R * B[t] + S[t + 1] * Delta[t] == V[t + 1]  # test self-financing

#         assert B.is_adapted(filtration=S.natural_filtration)
#         assert Delta.is_adapted(filtration=S.natural_filtration)
#         assert V.is_adapted(filtration=S.natural_filtration)

#         assert V.discount(rate=S.risk_free_rate).is_martingale(
#             prob_measure=S.emms
#         )
