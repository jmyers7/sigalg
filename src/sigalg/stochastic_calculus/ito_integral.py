"""Module for computing the Ito integral of a stochastic process with respect to Brownian motion."""

from ..core.base.time import Time
from ..processes.base.stochastic_process import StochasticProcess
from ..processes.types.brownian_motion import BrownianMotion


def ito_integral(process: StochasticProcess, random_state: int | None = None) -> Time:
    """Compute the Ito integral of a stochastic process with respect to Brownian motion."""
    time = process.time
    dt = time[1] - time[0]
    brownian_time = Time(name="brownian_time").from_list(
        time.data.to_list() + [time.data[-1] + dt], is_discrete=False
    )
    W = BrownianMotion(time=brownian_time, name="W").from_simulation(
        n_trajectories=process.data.shape[0], random_state=random_state
    )
    name = f"int({process.name} dW)" if process.name is not None else "Ito integral"
    return (process * W.increments(forward=True)).sum().with_name(name)
