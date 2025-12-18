from numbers import Real

from ...core.random_objects.random_variable import RandomVariable
from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra


def _validate_numeric_random_variable(rv: RandomVariable) -> None:
    if not isinstance(rv, RandomVariable):
        raise TypeError("rv must be a RandomVariable.")
    for value in rv.values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "Cannot compute expectation of a random variable with non-numeric values. "
            )


def _unconditional_expectation(rv: RandomVariable) -> Real:
    if not isinstance(rv, RandomVariable):
        raise TypeError("rv must be a RandomVariable.")
    if rv.probability_space is None:
        raise ValueError("rv must have a probability_space to compute expectation.")
    _validate_numeric_random_variable(rv)
    probabilities = rv.probability_space.probability_measure.values
    rv_values = rv.values
    aligned_probs = probabilities.reindex(rv_values.index)
    return (rv_values * aligned_probs).sum()


def expectation(
    rv: RandomVariable, sigma_algebra: SigmaAlgebra | None = None
) -> RandomVariable | Real:
    if not isinstance(rv, RandomVariable):
        raise TypeError("rv must be a RandomVariable.")
    if sigma_algebra is not None and not isinstance(sigma_algebra, SigmaAlgebra):
        raise TypeError("sigma_algebra must be a SigmaAlgebra or None.")
    if rv.probability_space is None:
        raise ValueError("rv must have a probability_space to compute expectation.")

    if sigma_algebra is None:
        return _unconditional_expectation(rv)
    else:
        if sigma_algebra.sample_space != rv.domain:
            raise ValueError(
                "SigmaAlgebra sample_space must match RandomVariable domain."
            )
        _validate_numeric_random_variable(rv)

        atom_id_to_probability_space = {
            atom_id: rv.probability_space.get_event_as_probability_space(sample_ids)
            for atom_id, sample_ids in sigma_algebra.atom_id_to_sample_ids.items()
        }
        events = atom_id_to_probability_space

        atom_id_to_expectation = {}
        for idx, event in events.items():
            atom_id_to_expectation[idx] = _unconditional_expectation(rv(event))
        outputs = {}
        for sample_id in rv.domain.data:
            atom_id = sigma_algebra.sample_id_to_atom_id[sample_id]
            outputs[sample_id] = atom_id_to_expectation[atom_id]
        return RandomVariable.on_probability_space(
            probability_space=rv.probability_space,
            outputs=outputs,
            name=f"E({rv.name}|{sigma_algebra.name})",
        )
