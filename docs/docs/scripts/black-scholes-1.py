"""Demonstrate Black-Scholes pricing and hedging with SigAlg."""

import matplotlib.pyplot as plt
import sigalg as sa

sa.style.set_style("black")

S_0, r, mu, sigma = 50, 0.01, 0.02, 0.1
T = sa.Time.continuous(start=0, stop=5, dt=0.1)

S = sa.BlackScholes.generate(
    initial_price=S_0,
    risk_free_rate=r,
    expected_return=mu,
    volatility=sigma,
    n_trajectories=10,
    index=T,
    random_state=42,
)

Phi = sa.Claim.european_option(strike=52, model=S, type="put")
X = Phi.price_process()  # (1)!
delta = Phi.delta()  # (2)!

_, axes = plt.subplots(ncols=2, sharex=True, figsize=(10, 4))

S.plot_trajectories(axes[0], colors=[sa.style.purple], label="$S_t$")
X.plot_trajectories(axes[0], colors=[sa.style.teal], label="$X_t$")
delta.plot_trajectories(axes[1], colors=[sa.style.blue, sa.style.lime])

axes[0].set_title(r"Prices $S_t$ and $X_t$ of underlying asset and claim")
axes[0].legend()
axes[0].set_xlabel("$t$")
axes[1].set_title(r"$\Delta$-hedging process")
axes[1].set_xlabel("$t$")

plt.tight_layout()
plt.show()
