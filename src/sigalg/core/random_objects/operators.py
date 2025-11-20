from ..sigma_algebras import SigmaAlgebra
from .random_variable import RandomVariable


def unconditional_expectation(rv: RandomVariable) -> float:
    return (rv.values * rv.domain.probability_measure.values).sum()


def expectation(
    rv: RandomVariable, sigma_algebra: SigmaAlgebra | None = None
) -> RandomVariable | float:
    if sigma_algebra is None:
        return unconditional_expectation(rv)
    else:
        events = sigma_algebra.to_events_as_probability_spaces()
        atom_id_to_expectation = {}
        for idx, event in events.items():
            atom_id_to_expectation[idx] = unconditional_expectation(rv(event))
        outputs = {}
        for sample_id in rv.domain.values:
            atom_id = sigma_algebra.sample_id_to_atom_id[sample_id]
            outputs[sample_id] = atom_id_to_expectation[atom_id]
        return RandomVariable(
            domain=rv.domain, outputs=outputs, name=f"E[{rv.name}|{sigma_algebra.name}]"
        )
