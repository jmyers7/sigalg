"""Demo risk-neutral pricing of an American put option with trinomial trees."""

import numpy as np
import sigalg as sa

S_0, u, m, d, r, T = 100, 1.1, 1.0, 0.9, 0.05, 4

S = sa.TrinomialPricingModel.generate(
    initial_price=S_0,
    up_factor=u,
    middle_factor=m,
    down_factor=d,
    risk_free_rate=r,
    mode="enum",
    length=T,
)


K = 90
american_put = sa.Claim.american_option(
    strike=K,
    model=S,
    option_type="put",
    name="american_put",
)

theta = 0.5  # (1)!
Q = S.emm(theta=theta)
S.discount(r).is_martingale(prob_measure=Q)

X = american_put.price_process(theta=theta)  # (2)!
X.discount(r).is_martingale(prob_measure=Q, kind="super")

tau = american_put.exercise_time(theta=theta)  # (3)!

price = X[0]  # (4)!
discounted_expectation = american_put.discount(r).at(tau).expectation(measure=Q)
np.allclose(price, discounted_expectation)
