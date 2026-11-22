"""Demonstrate the informal differential 'multiplication' rules in stochastic calculus."""
# (1)!

import matplotlib.pyplot as plt
import numpy as np
import sigalg as sa

sa.style.set_style()
rng = np.random.default_rng(42)

fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(8.5, 3))

quad_var = sa.ProcessTransforms.quad_var
cross_var = sa.ProcessTransforms.cross_var

T = sa.Time.continuous(start=0, stop=1, dt=0.001)
W = sa.BrownianMotion.generate(n_trajectories=5, index=T, name="W", random_state=rng)
t = sa.StochasticProcess.from_time(process=W)

quad_var(W).plot_trajectories(axes[0], colors=[sa.style.purple])
cross_var(t, W).plot_trajectories(axes[1], colors=[sa.style.teal])
quad_var(t).plot_trajectories(axes[2], colors=[sa.style.blue])

axes[0].set_title(r"$[W]_t = t \quad \Rightarrow \quad dW_t^2=0$")
axes[1].set_title(r"$[t,W]_t = 0 \quad \Rightarrow \quad dt \, dW_t=0$")
axes[2].set_title(r"$[t]_t= 0 \quad \Rightarrow \quad dt^2=0$")

fig.suptitle(
    r"Quadratic/cross variations of a Brownian motion $W_t$ and a 'time' process $t$"
)
plt.tight_layout()
plt.show()
