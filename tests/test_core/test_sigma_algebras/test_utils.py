import pytest

import sigalg as sa


class TestIsSubAlgebra:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["s0", "s1", "s2", "s3"])

    def test_trivial_is_sub_algebra_of_any(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space=sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        other = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        assert sa.is_sub_algebra(sub=trivial, super=other)

    def test_any_is_sub_algebra_of_power_set(self, sample_space):
        power_set = sa.SigmaAlgebra.power_set(sample_space=sample_space)
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        other = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        assert sa.is_sub_algebra(sub=other, super=power_set)

    def test_trivial_is_sub_algebra_of_power_set(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space=sample_space)
        assert sa.is_sub_algebra(sub=trivial, super=power_set)

    def test_power_set_not_sub_algebra_of_trivial(self, sample_space):
        trivial = sa.SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space=sample_space)
        assert not sa.is_sub_algebra(sub=power_set, super=trivial)

    def test_sigma_algebra_is_sub_algebra_of_itself(self, sample_space):
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma_alg = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        assert sa.is_sub_algebra(sub=sigma_alg, super=sigma_alg)

    def test_coarser_is_sub_algebra_of_finer(self, sample_space):
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=fine_atom_ids
        )
        assert sa.is_sub_algebra(sub=coarse, super=fine)

    def test_finer_not_sub_algebra_of_coarser(self, sample_space):
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=fine_atom_ids
        )
        assert not sa.is_sub_algebra(sub=fine, super=coarse)

    def test_incomparable_sigma_algebras(self, sample_space):
        sigma1_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma1 = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=sigma1_atom_ids
        )
        sigma2_atom_ids = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        sigma2 = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=sigma2_atom_ids
        )
        assert not sa.is_sub_algebra(sub=sigma1, super=sigma2)
        assert not sa.is_sub_algebra(sub=sigma2, super=sigma1)

    def test_three_level_sub_algebra_chain(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3", "s4", "s5"])
        coarsest = sa.SigmaAlgebra.trivial(sample_space=sample_space)
        middle_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1, "s4": 1, "s5": 1}
        middle = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=middle_atom_ids
        )
        finest_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 2, "s4": 2, "s5": 3}
        finest = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=finest_atom_ids
        )
        assert sa.is_sub_algebra(sub=coarsest, super=middle)
        assert sa.is_sub_algebra(sub=middle, super=finest)
        assert sa.is_sub_algebra(sub=coarsest, super=finest)
        assert not sa.is_sub_algebra(sub=middle, super=coarsest)
        assert not sa.is_sub_algebra(sub=finest, super=middle)
        assert not sa.is_sub_algebra(sub=finest, super=coarsest)

    def test_single_element_sample_space(self):
        sample_space = sa.SampleSpace(["s0"])
        sigma1 = sa.SigmaAlgebra.trivial(sample_space=sample_space)
        sigma2 = sa.SigmaAlgebra.power_set(sample_space=sample_space)
        assert sa.is_sub_algebra(sub=sigma1, super=sigma2)
        assert sa.is_sub_algebra(sub=sigma2, super=sigma1)

    def test_two_element_sample_space(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        trivial = sa.SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = sa.SigmaAlgebra.power_set(sample_space=sample_space)
        assert sa.is_sub_algebra(sub=trivial, super=power_set)
        assert not sa.is_sub_algebra(sub=power_set, super=trivial)

    def test_generated_by_random_variable(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        X_outputs = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        X = sa.RandomVariable(domain=sample_space, outputs=X_outputs, name="X")
        sigma_X = X.sigma_algebra
        Y_outputs = {"s0": 0, "s1": 1, "s2": 2, "s3": 3}
        Y = sa.RandomVariable(domain=sample_space, outputs=Y_outputs, name="Y")
        sigma_Y = Y.sigma_algebra
        assert sa.is_sub_algebra(sub=sigma_X, super=sigma_Y)
        assert not sa.is_sub_algebra(sub=sigma_Y, super=sigma_X)

    def test_sub_algebra_is_transitive(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        A = sa.SigmaAlgebra.trivial(sample_space=sample_space)
        B_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        B = sa.SigmaAlgebra(sample_space=sample_space, sample_id_to_atom_id=B_atom_ids)
        C = sa.SigmaAlgebra.power_set(sample_space=sample_space)
        assert sa.is_sub_algebra(sub=A, super=B)
        assert sa.is_sub_algebra(sub=B, super=C)
        assert sa.is_sub_algebra(sub=A, super=C)

    def test_sub_algebra_with_probability_spaces(self):
        """Test sub-algebra relation with sigma algebras on probability spaces."""
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_space = sa.ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = sa.SigmaAlgebra(
            probability_space=prob_space, sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = sa.SigmaAlgebra(
            probability_space=prob_space, sample_id_to_atom_id=fine_atom_ids
        )
        assert sa.is_sub_algebra(sub=coarse, super=fine)
        assert not sa.is_sub_algebra(sub=fine, super=coarse)
