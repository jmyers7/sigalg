import matplotlib.pyplot as plt
import sigalg as sa

sa.style.set_style("black")

mu = 0.7
sigma = 0.5
T = sa.Time.continuous(start=0, stop=2, dt=0.001)

X = sa.BrownianMotion.generate(
    drift=mu,
    diffusion=sigma,
    initial_state=1,
    n_trajectories=5,
    index=T,
    random_state=42,
)

W = X.driver
t = sa.StochasticProcess.from_time(process=X)

f_t = X**2
f_x = 2 * t * X
f_tt = 0.0
f_xx = 2 * t

ito_integral = sa.ProcessTransforms.ito_integral

increment = ito_integral(integrator=t * X**2)
int_dt = ito_integral(f_t + mu * f_x + (1 / 2) * sigma**2 * f_xx, t)
int_dw = ito_integral(sigma * f_x, W)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 4))

increment.plot_trajectories(
    axes[0], colors=[sa.style.purple, sa.style.teal], plot_kwargs={"linewidth": 1.5}
)
(int_dt + int_dw).plot_trajectories(
    axes[1], colors=[sa.style.purple, sa.style.teal], plot_kwargs={"linewidth": 1.5}
)

axes[0].set_title(r"$f(t, X_t) - f(0, X_0)$")
axes[1].set_title(
    r"$\int_0^t \left( f_t(u, X_u)+\mu f_x(u,X_u) + \frac{1}{2}\sigma^2 f_{xx}(u,X_u) \right) \, du + \int_0^t \sigma f_x(u, X_u) \, dW_u$"
)
axes[0].set_xlabel(r"$t$")
axes[1].set_xlabel(r"$t$")

fig.suptitle(r"Itô's lemma for $f(t,X_t) = tX_t^2$ and $dX_t = 0.7\, dt + 0.5 \, dW_t$")
plt.tight_layout()
plt.show()
