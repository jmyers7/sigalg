"""Probability spaces inherit methods from SampleSpace, SigmaAlgebra, and ProbabilityMeasure."""

from sigalg.core import ProbabilitySpace, SampleSpace

Omega = SampleSpace().from_sequence(size=4)  # (1)!

prob_space = ProbabilitySpace(sample_space=Omega)  # (2)!

A = prob_space.get_event([0, 2])  # (3)!

print("Is A measurable?", prob_space.is_measurable(A))  # (4)!

print("P(A) =", prob_space.P(A))  # (5)!
