"""Visualize the time-evolution of the transition probabilities of a random walk via a ridgeline plot, demonstrating a diffusion with positive drift."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sigalg.core import Time
from sigalg.processes import RandomWalk

yellow = "#FFC300"
blue = "#3399FF"
purple = "#AA77CC"

T = Time.discrete(length=200)
X = RandomWalk(  # (1)!
    p=0.7,
    name="X",
    time=T,
).from_simulation(
    n_trajectories=10_000,
    random_state=42,
)

_, ax = plt.subplots(figsize=(7, 5))

n_plots = 8  # (2)!
time_step = 25
times = [time_step * k for k in range(1, n_plots + 1)]

cmap = LinearSegmentedColormap.from_list(  # (3)!
    "conditional_cmap", [yellow, purple, blue]
)
colors = [cmap(i / (n_plots - 1)) for i in range(n_plots)]

for color, t in zip(colors, times, strict=False):
    probabilities = X[t - 1].range.probability_measure.data  # (4)!
    probabilities.index = X[t - 1].range.sample_space.data.values  # (5)!

    ax.bar(  # (6)!
        x=probabilities.index,
        height=-probabilities.values * 175,  # (7)!
        width=0.8,
        color=color,
        bottom=t,
    )

ax.invert_yaxis()
ax.set_xlabel("state")
ax.set_ylabel("time")
ax.set_title(
    "Time-evolution of the probability distribution\nof a random walk",
)
plt.tight_layout()
plt.show()
