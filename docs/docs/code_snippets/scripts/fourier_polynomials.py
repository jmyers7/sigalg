"""Fit a trigonometric polynomial to data using the L2 orthogonal projection operator."""

import matplotlib.pyplot as plt
import numpy as np

from sigalg.core import Index, RandomVector
from sigalg.l2 import L2

arr = np.load("data/fourier_data.npy")

component_names = Index().from_list(["X", "Y"])
Z = RandomVector(
    name="Z",
    index=component_names,
).from_numpy(array=arr)

Omega = Z.domain
P = Z.probability_measure

X, Y = Z.components

H = L2(sample_space=Omega, probability_measure=P)

_, u, _ = H.proj(
    rv=Y,
    subspace=[np.cos(n * X) for n in range(1, 5)],
)

x = np.linspace(X.data.min(), X.data.max(), 100)
y = sum(u[n - 1] * np.cos(n * x) for n in range(1, 5))

plt.scatter(X.data, Y.data, color="blue", label="Data")
plt.plot(x, y, color="red", label="Trigonometric Fit")
plt.legend()
plt.tight_layout()
plt.show()
