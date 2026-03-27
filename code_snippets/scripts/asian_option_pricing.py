# """Price an Asian call option using a binomial pricing model."""

# import matplotlib.pyplot as plt

# from sigalg.core import Time
# from sigalg.finance import AsianOption, BinomialPricingModel

# S_0 = 100  # (1)!
# u = 1.1  # (2)!
# p = 0.7  # (3)!
# r = 0.01  # (4)!
# T = 3  # (5)!
# time = Time.discrete(length=T)

# S = BinomialPricingModel(  # (6)!
#     initial_price=S_0,
#     up_factor=u,
#     up_prob=p,
#     risk_free_rate=r,
#     time=time,
# )

# S.from_enumeration(enum_mode="dense")  # (7)!
# print("Underlying prices:\n", S)

# # S.plot_trajectories(  # (8)!
# #     y_label="price", title="Underlying prices"
# # )
# # plt.show()

# K = 100
# asian_call = AsianOption(pricing_model=S, strike=K, option_type="call")

# print("\nAsian call option exercise values:\n", asian_call.payoff)

# B, N, V, price = S.replicating_portfolio(claim=asian_call)

# print("\nReplicating portfolio bank holdings:\n", B)
# print("\nReplicating portfolio underlying asset holdings:\n", N)
# print("\nReplicating portfolio values:\n", V)

# V.discount(rate=S.risk_free_rate).plot_trajectories(
#     y_label="value", title="Replicating Portfolio Value"
# )
# plt.show()
