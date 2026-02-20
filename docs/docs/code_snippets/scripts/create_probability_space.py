"""Create a probability space from a sample space, a sigma-algebra, and a probability measure."""

from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra

Omega = SampleSpace().from_sequence(size=4)  # (1)!

atom_ids = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
}
F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)  # (2)!

probabilities = {
    0: 0.1,
    1: 0.2,
    2: 0.4,
    3: 0.3,
}
P = ProbabilityMeasure(sample_space=Omega).from_dict(probabilities)  # (3)!

prob_space = ProbabilitySpace(  # (4)!
    sample_space=Omega,
    sigma_algebra=F,
    probability_measure=P,
)

print(prob_space.sample_space)  # (5)!
print("\n", prob_space.sigma_algebra)
print("\n", prob_space.probability_measure)

new_atom_ids = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
}
G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(new_atom_ids)  # (6)!

new_probabilities = {
    0: 0.1,
    1: 0.6,
    2: 0.2,
    3: 0.1,
}
Q = ProbabilityMeasure(sample_space=Omega, name="Q").from_dict(  # (7)!
    new_probabilities
)

prob_space.sigma_algebra = G  # (8)!
prob_space.probability_measure = Q

print("\n", prob_space.sigma_algebra)
print("\n", prob_space.probability_measure)
