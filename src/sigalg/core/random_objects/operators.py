from numbers import Real

from ..sigma_algebras import SigmaAlgebra
from .random_variable import RandomVariable


def _validate_numeric_random_variable(rv: RandomVariable) -> None:
    """Validate that all values of the random variable are numeric (but not boolean)."""
    for value in rv.values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "Cannot compute expectation of a random variable with non-numeric values. "
            )


def unconditional_expectation(rv: RandomVariable) -> Real:
    if rv.probability_space is None:
        raise ValueError(
            "RandomVariable must have a probability_space to compute expectation."
        )
    _validate_numeric_random_variable(rv)
    probabilities = rv.probability_space.probability_measure.values
    rv_values = rv.values
    aligned_probs = probabilities.reindex(rv_values.index)
    return (rv_values * aligned_probs).sum()


def expectation(
    rv: RandomVariable, sigma_algebra: SigmaAlgebra | None = None
) -> RandomVariable | Real:
    if sigma_algebra is None:
        return unconditional_expectation(rv)
    else:
        _validate_numeric_random_variable(rv)
        events = sigma_algebra.to_events_as_probability_spaces()
        atom_id_to_expectation = {}
        for idx, event in events.items():
            atom_id_to_expectation[idx] = unconditional_expectation(rv(event))
        outputs = {}
        for sample_id in rv.domain.values:
            atom_id = sigma_algebra.sample_id_to_atom_id[sample_id]
            outputs[sample_id] = atom_id_to_expectation[atom_id]
        return RandomVariable(
            domain=rv.domain,
            probability_space=rv.probability_space,
            outputs=outputs,
            name=f"E({rv.name}|{sigma_algebra.name})",
        )
