import pytest
import sigalg as sa


class TestSigmaAlgebraInitialization:
    """Tests for SigmaAlgebra initialization"""

    def test_with_non_sample_space(self):
        """Should raise error when sample_space is not SampleSpace"""
        with pytest.raises(ValueError):
            sa.SigmaAlgebra(sample_space="not_a_sample_space", atom_ids="power_set")

    def test_with_invalid_atom_ids(self):
        """Should raise error when atom_ids is invalid"""
        data = [[1], [2], [3]]
        sample_space = sa.SampleSpace(data=data)

        with pytest.raises(ValueError):
            sa.SigmaAlgebra(sample_space, atom_ids=123)  # Not a list or valid string

    def test_init_power_set(self):
        """Should create SigmaAlgebra with power set"""
        data = [[1], [2], [3]]
        sample_space = sa.SampleSpace(data=data)

        sigma_algebra = sa.SigmaAlgebra(sample_space, "power_set")

        assert sigma_algebra.atoms == sample_space.list_of_sample_points

    def test_init_trivial(self):
        """Should create SigmaAlgebra with trivial algebra"""
        data = [[1], [2], [3], [4]]
        sample_space = sa.SampleSpace(data=data)

        sigma_algebra = sa.SigmaAlgebra(sample_space, "trivial")

        assert sigma_algebra.atoms[0] == sample_space.as_event()

    def test_init_custom_atoms(self):
        """Should create SigmaAlgebra with custom atoms"""
        data = [[1], [2], [3], [4]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [0, 0, 1, 2]
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)

        assert sigma_algebra.atoms[0] == sample_space[["omega1", "omega2"]]
        assert sigma_algebra.atoms[1] == sample_space[["omega3"]]
        assert sigma_algebra.atoms[2] == sample_space[["omega4"]]

    def test_init_invalid_atom_ids_length(self):
        """Should raise error for invalid atom_ids length"""
        data = [[1], [2], [3]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [0, 1]  # Invalid length

        with pytest.raises(ValueError):
            sa.SigmaAlgebra(sample_space, atom_ids)

    def test_init_invalid_atom_ids_type(self):
        """Should raise error for invalid atom_ids type"""
        data = [[1], [2], [3]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = "invalid_type"  # Not a list

        with pytest.raises(ValueError):
            sa.SigmaAlgebra(sample_space, atom_ids)

    def test_init_atom_id_index(self):
        """Should match atom_ids index with sample space index"""
        data = [[1], [2], [3]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [0, 1, 0]
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)

        assert sigma_algebra._atom_ids.index.equals(sample_space._df.index)


class TestSigmaAlgebraAtomsProperty:
    """Tests for SigmaAlgebra atoms property"""

    def test_atoms_property(self):
        """Should return correct atoms via atoms property"""
        data = [[1], [2], [3], [4]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [0, 0, 1, 2]
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)

        atoms = sigma_algebra.atoms

        assert len(atoms) == 3
        assert atoms[0] == sample_space[["omega1", "omega2"]]
        assert atoms[1] == sample_space[["omega3"]]
        assert atoms[2] == sample_space[["omega4"]]

    def test_atoms_with_ids(self):
        """Should return atoms as dict with atom IDs as keys"""
        data = [[1], [2], [3], [4], [5]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [2, 2, 5, 3, 5]
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)

        atoms = sigma_algebra.atoms_with_ids

        assert len(atoms) == 3
        assert atoms[2] == sample_space[["omega1", "omega2"]]
        assert atoms[3] == sample_space[["omega4"]]
        assert atoms[5] == sample_space[["omega3", "omega5"]]

    def test_atoms_are_events(self):
        """Atoms returned should be Event objects"""
        data = [[1], [2], [3]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [0, 1, 0]
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)

        atoms = sigma_algebra.atoms

        assert all(isinstance(atom, sa.Event) for atom in atoms)


class TestClassMethods:
    """Tests for SigmaAlgebra atom_id_of method"""

    def test_atom_id_of(self):
        """Should return correct atom ID for given sample point index"""
        data = [[1], [2], [3], [4]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [0, 0, 1, 2]
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)

        assert sigma_algebra.atom_id_of("omega1") == 0
        assert sigma_algebra.atom_id_of("omega2") == 0
        assert sigma_algebra.atom_id_of("omega3") == 1
        assert sigma_algebra.atom_id_of("omega4") == 2
    
    def test_len(self):
        """Should return correct number of atoms via len()"""
        data = [[1], [2], [3], [4], [5]]
        sample_space = sa.SampleSpace(data=data)

        atom_ids = [0, 1, 0, 2, 1]
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)

        assert len(sigma_algebra) == 3  # Three unique atoms: 0, 1, 2
