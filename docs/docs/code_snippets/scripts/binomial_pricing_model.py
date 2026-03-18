from sigalg.core import Time
from sigalg.finance import BinomialPricingModel, european_option

S_0 = 100  # (1)!
u = 1.1  # (2)!
p = 0.7
r = 0.01  # (3)!
T = 3  # (4)!
time = Time.discrete(length=T)

S = BinomialPricingModel(  # (5)!
    initial_price=S_0,
    up_factor=u,
    up_prob=p,
    risk_free_rate=r,
    time=time,
).from_enumeration()

K = 100  # (6)!
call_option = european_option(price=S[T], strike=K)

B, N, V, price = S.replicating_portfolio(claim=call_option)  # (7)!

print("The price process of the underlying asset:\n", S)
print("\nThe European call option:\n", call_option)
print("\nThe bank account balance:\n", B)
print("\nNumber of units of underlying held:\n", N)
print("\nValue of the replicating portfolio:\n", V)
print("\nThe price of the call option is:", price)
