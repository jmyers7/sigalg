"""Test the quadratic variation method."""
# (1)!

import matplotlib.pyplot as plt
import sigalg as sa

sa.style.set_style()

T = sa.Time.continuous(start=0, stop=1, dt=0.001)
W = sa.BrownianMotion.generate(
    n_trajectories=5,
    index=T,
    name="W",
    random_state=42,
)
t = sa.StochasticProcess.from_time(process=W)

_, ax = plt.subplots()
W.quad_var().plot_trajectories(ax, colors=[sa.style.purple])
t.plot_trajectories(ax, colors=[sa.style.amber])
plt.show()
