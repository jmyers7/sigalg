"""Demonstrate Delta-hedging a short position on call options with SigAlg."""

import matplotlib.pyplot as plt
import numpy as np
import sigalg as sa

sa.style.set_style("black")

dt = 0.01
T = sa.Time.continuous(start=0, stop=1, dt=dt)
S_0, r, mu, sigma = 49, 0.05, 0.04, 0.1
K, n = 50, 100  # (1)!
premium = 3 * n  # (2)!
R = np.exp(r * dt)


fig, axes = plt.subplots(ncols=3, nrows=4, sharex=True, sharey="row", figsize=(15, 10))

for j, seed in enumerate([5, 25, 42]):  # (3)!
    col_ax = axes[:, j]

    S = sa.BlackScholes.generate(
        initial_price=S_0,
        risk_free_rate=r,
        expected_return=mu,
        volatility=sigma,
        n_trajectories=1,
        index=T,
        random_state=seed,
    )

    Phi = sa.Claim.european_option(strike=K, model=S, option_type="call")
    Delta, B, V = S.hedging_portfolio(Phi)  # (4)!

    S.plot_trajectories(col_ax[0])
    col_ax[0].axhline(K, color="red", label="Strike")

    (n * Delta).plot_trajectories(col_ax[1], colors=[sa.style.blue])

    (n * V.discount(gross_return=R) + premium).plot_trajectories(
        col_ax[2],
        colors=[sa.style.teal],  # (5)!
    )
    col_ax[2].axhline(
        n * Phi.discount(gross_return=R).last_rv.item(),
        color="red",
        label="Discounted payoff",
    )

    (n * B).plot_trajectories(col_ax[3], colors=[sa.style.amber])

    if j == 0:
        col_ax[0].set_ylabel("$S_t$")
        col_ax[1].set_ylabel(r"$\Delta_t$")
        col_ax[2].set_ylabel(r"$\tilde{V}_t+\text{premium}$")
        col_ax[3].set_ylabel("$B_t$")

        col_ax[0].legend()
        col_ax[2].legend()

    col_ax[3].set_xlabel("$t$")

axes[0, 0].set_title("Out of the money")
axes[0, 1].set_title("In the money")
axes[0, 2].set_title("Late dive out of the money")

plt.tight_layout()
plt.show()
