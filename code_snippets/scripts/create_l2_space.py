"""Create an L2-space from a sample space, sigma-algebra, and probability measure. Check if random variables are in the L2-space."""

from sigalg.core import ProbabilityMeasure, RandomVariable, SampleSpace, SigmaAlgebra
from sigalg.l2 import L2

Omega = SampleSpace().from_sequence(size=4)  # (1)!

atom_ids = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
}
F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)  # (2)!

probabilities = {
    0: 0.2,
    1: 0.1,
    2: 0.4,
    3: 0.3,
}
P = ProbabilityMeasure(sample_space=Omega).from_dict(probabilities)  # (3)!

H = L2(sample_space=Omega, sigma_algebra=F, probability_measure=P)  # (4)!

outputs_X = {
    0: 3,
    1: 3,
    2: 5,
    3: 5,
}
outputs_Y = {
    0: 1,
    1: 3,
    2: 4,
    3: 2,
}
X = RandomVariable(domain=Omega).from_dict(outputs_X)
Y = RandomVariable(domain=Omega, name="Y").from_dict(outputs_Y)  # (5)!

print("Is X in H?", X in H)  # (6)!
print("\nIs Y in H?", Y in H)  # (7)!
