"""Demo SigAlg API by creating a random walk from scratch."""

import pandas as pd
from scipy.stats import bernoulli

from sigalg.core import Time
from sigalg.processes import IIDProcess

T = Time.discrete(start=1, stop=4)  # (1)!
B = IIDProcess(
    distribution=bernoulli(p=0.7),
    support=[0, 1],
    time=T,
).from_enumeration()  # (2)!

Y = 2 * B - 1  # (3)!
X = Y.cumsum(name="X")  # (4)!
X.insert_rv(state=0, time=0, in_place=True)  # (5)!

F = X.natural_filtration  # (6)!
expectation = X[4].expectation(sigma_algebra=F[3])  # (7)!

print(pd.concat([X.data, expectation.data], axis=1))  # (8)!
print("\nIs X a submartingale?", X.is_submartingale())  # (9)!
