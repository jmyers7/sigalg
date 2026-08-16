# from math import inf

# import numpy as np
# import pytest

# from sigalg.core import Time
# from sigalg.finance import AmericanOption, BinomialPricingModel


# class TestReplicatingPortfolio:
#     @pytest.fixture
#     def S_non_recombining(self):
#         S_0 = 4
#         u = 3
#         d = 1 / 2
#         p = 0.7
#         r = 0.25
#         T = Time.discrete(length=3)
#         return BinomialPricingModel(
#             initial_price=S_0,
#             up_factor=u,
#             down_factor=d,
#             up_prob=p,
#             risk_free_rate=r,
#             time=T,
#         )

#     def test_replicating_portfolio_for_put_in_dense_mode_non_recombining(
#         self, S_non_recombining
#     ):
#         """Test that the replicating portfolio for an American put option in a non-recombining tree is correctly computed in dense enumeration mode."""
#         S = S_non_recombining
#         S.from_enumeration(enum_mode="dense")
#         S_0 = S.initial_price
#         R = S.risk_free_gross_return
#         q = S.risk_neutral_probs[0]
#         u = S.up_factor
#         d = S.down_factor
#         K = 5
#         T = S.time[-1]
#         put_option = AmericanOption(pricing_model=S, strike=K, option_type="put")
#         B, Delta, V, price, tau = S.replicating_portfolio(claim=put_option)

#         S_expected = np.array(
#             [
#                 [S_0, S_0 * u, S_0 * u**2, S_0 * u**3],  # uuu
#                 [S_0, S_0 * u, S_0 * u**2, S_0 * u**2 * d],  # uud
#                 [S_0, S_0 * u, S_0 * u * d, S_0 * u**2 * d],  # udu
#                 [S_0, S_0 * u, S_0 * u * d, S_0 * u * d**2],  # udd
#                 [S_0, S_0 * d, S_0 * d * u, S_0 * d * u**2],  # duu
#                 [S_0, S_0 * d, S_0 * d * u, S_0 * d * u * d],  # dud
#                 [S_0, S_0 * d, S_0 * d**2, S_0 * d**2 * u],  # ddu
#                 [S_0, S_0 * d, S_0 * d**2, S_0 * d**3],  # ddd
#             ]
#         )

#         V_expected = [
#             [
#                 (q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R,
#                 (1 - q) ** 2 * 2 / R**2,
#                 0,
#                 0,
#             ],
#             [
#                 (q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R,
#                 (1 - q) ** 2 * 2 / R**2,
#                 0,
#                 0,
#             ],
#             [
#                 (q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R,
#                 (1 - q) ** 2 * 2 / R**2,
#                 (1 - q) * 2 / R,
#                 0,
#             ],
#             [
#                 (q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R,
#                 (1 - q) ** 2 * 2 / R**2,
#                 (1 - q) * 2 / R,
#                 2,
#             ],
#             [(q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R, 3, (1 - q) * 2 / R, 0],
#             [(q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R, 3, (1 - q) * 2 / R, 2],
#             [(q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R, 3, 4, 2],
#             [(q * (1 - q) ** 2 * 2 / R**2 + (1 - q) * 3) / R, 3, 4, 4.5],
#         ]

#         tau_expected = np.array([inf, inf, inf, 3, 1, 1, 1, 1])

#         assert np.allclose(S.data.values, S_expected)
#         assert np.allclose(V.data.values, V_expected)
#         assert np.allclose(tau.data.values, tau_expected)

#         for t in range(T):
#             assert R * B[t] + S[t + 1] * Delta[t] == V[t + 1]
