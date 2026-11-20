"""Demo a Markov chain in SigAlg."""

from math import prod

import numpy as np
import sigalg as sa

K = [[0.4, 0.6], [0.7, 0.3]]
pi = [0.2, 0.8]
T = 10

X = sa.MarkovChain.generate(kernel=K, initial_distribution=pi, length=T, mode="enum")

P = X.measure  # (1)!
np.allclose(
    P >> X, (P >> X[0]) * prod(P.conditional(X[t]) >> X[t + 1] for t in range(T))
)
