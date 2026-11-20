"""Demo polynomial regression as orthogonal projection in an L^2-space."""

import matplotlib.pyplot as plt
import numpy as np
import sigalg as sa

sa.style.set_style()

n = 20
Omega = sa.SampleSpace.from_sequence(size=20)
P = sa.ProbabilityMeasure.uniform(Omega, name="P")
H = sa.L2(Omega, measure=P)  # (1)!

X = sa.RandomVariable(  # (2)!
    domain=Omega,
    measure=P,
    mapping=np.linspace(-2, 2, n),
)
epsilon = sa.RandomVariable.from_rand(
    domain=Omega,
    measure=P,
    distribution="normal",
    name="epsilon",
    random_state=42,
)
Y = (2 + 3 * X - X**2 + epsilon).with_name("Y")

one = sa.RandomVariable.from_constant(  # (3)!
    domain=Omega,
    measure=P,
    constant=1,
    name="one",
)
polynomial_basis = [one, X, X**2, X**3]
Y_fit, coeffs_poly, dim_poly = H.proj(Y, polynomial_basis)

x_smooth = np.linspace(-2, 2, 100)  # (4)!
y_smooth = (
    coeffs_poly[0]
    + coeffs_poly[1] * x_smooth
    + coeffs_poly[2] * x_smooth**2
    + coeffs_poly[3] * x_smooth**3
)

_, ax = plt.subplots()
ax.scatter(X, Y, color=sa.style.amber, s=50)
ax.plot(x_smooth, y_smooth)
plt.tight_layout()
plt.show()
