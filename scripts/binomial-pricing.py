"""Demo risk-neutral pricing of a European call option with binomial trees."""

import numpy as np
import sigalg as sa

S_0, r, u, d, T = 100, 0.05, 1.1, 0.9, 3

S = sa.BinomialPricingModel.generate(
    initial_price=S_0,
    up_factor=u,
    down_factor=d,
    risk_free_rate=r,
    length=T,
    mode="enum",
)

K = 100
euro_call = sa.Claim.european_option(model=S, strike=K, type="call", name="euro_call")

Delta, B, V = S.replicating_portfolio(claim=euro_call)
X = euro_call.price_process()

X.last_rv == V.last_rv  # (1)!
all(np.allclose(V[t + 1], (1 + r) * B[t] + S[t + 1] * Delta[t]) for t in range(T))

Q = S.emm()  # (2)!
X.discount(r).is_martingale(prob_measure=Q)
