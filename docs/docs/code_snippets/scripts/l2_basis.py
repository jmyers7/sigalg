from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
from sigalg.l2 import L2

Omega = SampleSpace().from_sequence(size=3)  # (1)!

F = SigmaAlgebra(sample_space=Omega).from_dict(  # (2)!
    {
        0: 0,
        1: 0,
        2: 1,
    }
)

P = ProbabilityMeasure(sample_space=Omega).from_dict(  # (3)!
    {
        0: 0.2,
        1: 0.5,
        2: 0.3,
    }
)

H = L2(  # (4)!
    sample_space=Omega,
    sigma_algebra=F,
    probability_measure=P,
)

e_0, e_1 = H.basis.values()  # (5)!

print("Basis vectors with measure P:")
print("\n", e_0)
print("\n", e_1)

Q = ProbabilityMeasure(sample_space=Omega).from_dict(  # (6)!
    {
        0: 0.7,
        1: 0.3,
        2: 0.0,
    }
)

H.probability_measure = Q  # (7)!

print("\nNumber of basis vectors with new measure Q:", len(H.basis))  # (8)!
print("\nNew basis vector:", list(H.basis.values())[0])
