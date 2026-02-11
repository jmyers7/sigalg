"""Create a random walk from scratch."""

import pandas as pd
from scipy.stats import bernoulli

from sigalg.core import Time
from sigalg.processes import IIDProcess

# Create an IID Bernoulli process B with p=0.7
T = Time.discrete(start=1, stop=4)
B = IIDProcess(
    distribution=bernoulli(p=0.7),
    support=[0, 1],
    time=T,
).from_enumeration()

# Create a process Y with Y_t = -1 or Y_t = 1
Y = 2 * B - 1

# Create a random walk process X
X = Y.cumsum(name="X")

# Get the natural filtration F_t = σ(X_1,X_2,...,X_t)
F = X.natural_filtration

# Compute the conditional expectation E(X_4 | F_3)
expectation = X[4].expectation(sigma_algebra=F[3])

# Print the trajectories of X and the expectation together
print(pd.concat([X.data, expectation.data], axis=1))

# A random walk with positive drift is a submartingale
print("\nIs X a submartingale?", X.is_submartingale())
