"""Create a ProbabilitySpace from an instance of SampleSpace, SigmaAlgebra, and ProbabilityMeasure."""

from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra

Omega = SampleSpace().from_sequence(size=4)  # (1)!

atom_ids = {
    0: 0,
    1: 1,
    2: 0,
    3: 1,
}
F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)  # (2)!

probabilities = {
    0: 0.1,
    1: 0.2,
    2: 0.4,
    3: 0.3,
}
P = ProbabilityMeasure(sample_space=Omega).from_dict(probabilities)  # (3)!

prob_space = Omega.make_probability_space(  # (4)!
    sigma_algebra=F,
    probability_measure=P,
)

print(prob_space)
