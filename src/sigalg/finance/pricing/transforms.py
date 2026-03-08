from numbers import Real

import numpy as np

from sigalg.processes import StochasticProcess


# TODO: Expand docstring
# TODO: Write unit tests
def discount(rate: Real, process: StochasticProcess) -> StochasticProcess:
    """Discount a price process using a given rate."""
    if not isinstance(rate, Real) or rate <= 0:
        raise ValueError("Rate must be a positive real number.")
    if not isinstance(process, StochasticProcess):
        raise TypeError("Process must be an instance of StochasticProcess.")

    num_rows = len(process.data)
    discount_factors = np.repeat(
        [[1 / (1 + rate) ** t for t in range(len(process))]], num_rows, axis=0
    )

    discounted_data = process.data * discount_factors
    name = f"{process.name}_discounted" if process.name else "discounted_process"

    output = StochasticProcess(
        time=process.time,
        domain=process.domain,
        name=name,
        is_discrete_state=True,
    ).from_pandas(discounted_data)
    output._is_enumerated = True
    output._probability_measure = process._probability_measure
    return output
