"""Fit a cubic polynomial to data using the L2 orthogonal projection operator."""

import matplotlib.pyplot as plt
import numpy as np

from sigalg.core import Index, RandomVector
from sigalg.l2 import L2

arr = np.load("data/regression_data.npy")  # (1)!

component_names = Index().from_list(["X", "Y"])
Z = RandomVector(  # (2)!
    name="Z",
    index=component_names,
).from_numpy(array=arr)

Omega = Z.domain  # (3)!
P = Z.probability_measure

X, Y = Z.components  # (4)!

H = L2(sample_space=Omega, probability_measure=P)  # (5)!

_, u, _ = H.proj(  # (6)!
    rv=Y,
    subspace=[X**0, X, X**2, X**3],
)

x = np.linspace(X.data.min(), X.data.max(), 100)
y = u[0] + u[1] * x + u[2] * x**2 + u[3] * x**3  # (7)!

plt.scatter(X.data, Y.data, color="blue", label="Data")  # (8)!
plt.plot(x, y, color="red", label="Polynomial Fit")
plt.legend()
plt.tight_layout()
plt.show()
