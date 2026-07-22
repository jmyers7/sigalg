import numpy as np

from sigalg.core import (
    ProbabilityMeasure,
    RadonNikodym,
    RandomVariable,
    SampleSpace,
    SigmaAlgebra,
)


class TestMathematicalInvariants:
    def test_change_of_variables(self):
        """Test the change-of-variables formula for Radon-Nikodym derivatives."""
        Omega = SampleSpace.from_sequence(size=10)
        F = SigmaAlgebra.from_rand(
            num_atoms=3,
            sample_space=Omega,
            random_state=42,
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.8,
                2: 0.0,
            },
        )
        Q = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.9,
                1: 0.1,
                2: 0.0,
            },
            name="Q",
        )
        dQ_dP = RadonNikodym.from_measures(prob_measure=Q, wrt=P)
        X = RandomVariable.from_randnorm(
            sample_space=Omega,
            sig_alg=F,
            prob_measure=P,
            random_state=42,
        )

        assert np.allclose(
            X.integrate(prob_measure=Q), (X * dQ_dP).integrate(prob_measure=P)
        )

    def test_radon_nikodym_derivatives_and_conditional_measures(self):
        """Test the relationship between Radon-Nikodym derivatives and conditional measures."""
        Omega = SampleSpace.from_sequence(size=10)
        F = SigmaAlgebra.from_rand(
            num_atoms=3,
            sample_space=Omega,
            random_state=42,
        )
        P = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.8,
                2: 0.0,
            },
        )
        Q = ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.9,
                1: 0.1,
                2: 0.0,
            },
            name="Q",
        )
        dQ_dP = RadonNikodym.from_measures(prob_measure=Q, wrt=P)
        for U in F.to_atoms:
            assert dQ_dP.integrate(event=U) == Q(U)

    def test_conditional_distribution_radon_nikodym_formula(self):
        """Test the formula for the Radon-Nikodym derivative of a conditional distribution."""
        Omega = SampleSpace.from_sequence(size=50)
        F = SigmaAlgebra.from_rand(
            num_atoms=23,
            sample_space=Omega,
            random_state=42,
            variable_names=["A_i"],
        )
        G = SigmaAlgebra.from_rand(
            num_atoms=12,
            super=F,
            random_state=42,
            name="G",
            variable_names=["B_i"],
        )
        P = ProbabilityMeasure.from_rand(
            sig_alg=F,
            num_null_atoms=4,
            random_state=42,
        )

        for i, B in G.atom_id_to_event.items():
            Q = P.given(G, name="Q")(B_i=i)
            dQ_dP = RadonNikodym.from_measures(Q, P)
            if P(B) > 0:
                assert P.equal_almost_surely(dQ_dP, B.indicator / P(B))
