"""Demonstrate the Fundamental Theorem of Calculus with Itô's correction."""
# (1)!

import numpy as np
import sigalg as sa

T = sa.Time.continuous(start=0, stop=4, dt=0.001)
W = sa.BrownianMotion.generate(
    n_trajectories=5,
    index=T,
    name="W",
    random_state=42,
)

ito_integral = sa.ProcessTransforms.ito_integral
quad_var = sa.ProcessTransforms.quad_var


def f(X: sa.StochasticProcess) -> sa.StochasticProcess:
    return X**2


def f_prime(X: sa.StochasticProcess) -> sa.StochasticProcess:
    return 2 * X


increment = ito_integral(integrator=f(W))
classical_term = ito_integral(f_prime(W), W)
ito_correction = ito_integral(integrator=quad_var(W))

np.allclose(increment, classical_term + ito_correction)
