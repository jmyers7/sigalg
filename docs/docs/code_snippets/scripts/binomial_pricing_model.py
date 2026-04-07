"""Model the price process of a risky asset using a binomial tree."""

import matplotlib.pyplot as plt

from sigalg.core import Time
from sigalg.finance import BinomialPricingModel

S_0 = 100  # (1)!
u = 1.1  # (2)!
p = 0.7  # (3)!
r = 0.01  # (4)!
T = 3  # (5)!
time = Time.discrete(length=T)

S = BinomialPricingModel(  # (6)!
    initial_price=S_0,
    up_factor=u,
    up_prob=p,
    risk_free_rate=r,
    time=time,
)

S.from_enumeration(enum_mode="dense")  # (7)!
print("Binomial Pricing Model (Dense Enumeration):\n", S)

S.plot_trajectories(  # (8)!
    y_label="price", title="Binomial Pricing Model: Dense Enumeration"
)
plt.show()

print(  # (9)!
    "\nReal-World Probability Measure (Dense Enumeration):\n", S.probability_measure
)
print("\nRisk Neutral Measure (Dense Enumeration):\n", S.emms)  # (10)!

S_discounted = S.discount(rate=S.risk_free_rate)  # (11)!
print(
    "\nAre the discounted prices a martingale under the risk-neutral measure? ",
    S_discounted.is_martingale(probability_measure=S.emms),
)

S.from_enumeration(enum_mode="sparse")  # (12)!
print("\nBinomial Pricing Model (Sparse Enumeration):\n", S)

S.plot_trajectories(  # (13)!
    y_label="price", title="Binomial Pricing Model: Sparse Enumeration"
)
plt.show()

print(  # (14)!
    "\nReal-World Probability Measure (Sparse Enumeration):\n", S.probability_measure
)
print("\nRisk Neutral Measure (Sparse Enumeration):\n", S.emms)  # (15)!
