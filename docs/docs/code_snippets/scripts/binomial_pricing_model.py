from sigalg.finance import BinomialPricingModel, european_option

S_0 = 100  # (1)!
u = 1.1  # (2)!
r = 0.01  # (3)!
model = BinomialPricingModel(
    initial_price=S_0, up_factor=u, risk_free_rate=r, length=3
)  # (4)!

S = model.price_process  # (5)!
print("Price process of the stock:\n", S)

call = european_option(price=S[3], strike=100, option_type="call")  # (6)!
B, N, V, price = model.replicating_portfolio(claim=call)  # (7)!

print("\nThe bank account balance:\n", B)  # (8)!
print("\nNumber of units of stock held:\n", N)  # (9)!
print("\nValue of the replicating portfolio:\n", V)  # (10)!
print("\nFair price of the European call option:\n", price)  # (11)!

F = S.natural_filtration  # (12)!
# print("\nIs the B process predictable:", B.is_predictable(F))
# print("\nIs the N process predictable:", N.is_predictable(F))
