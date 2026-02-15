from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra

# Create a sample space Omega = [0, 1, 2, 3]
Omega = SampleSpace().from_sequence(size=4)

# Define a sigma-algebra
atom_ids = {
    0: 0,
    1: 1,
    2: 0,
    3: 1,
}
F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)

# Define a probability measure
probabilities = {
    0: 0.1,
    1: 0.2,
    2: 0.4,
    3: 0.3,
}
P = ProbabilityMeasure(sample_space=Omega).from_dict(probabilities)

# Create a probability space
prob_space = Omega.make_probability_space(sigma_algebra=F, probability_measure=P)

print(prob_space)
