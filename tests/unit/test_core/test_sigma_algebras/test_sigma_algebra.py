from collections.abc import Hashable

import pandas as pd
import pytest

from sigalg.core import Event, SampleSpace, SigmaAlgebra


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_construction_with_integer_atom_ids(self, sample_space):
        """Test constructing a SigmaAlgebra with integer atom IDs."""
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert sigma_algebra.sample_space == sample_space
        assert sigma_algebra.sample_id_to_atom_id == atom_ids

    def test_construction_with_string_atom_ids(self, sample_space):
        """Test constructing a SigmaAlgebra with string atom IDs."""
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B", "omega3": "B"}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert sigma_algebra.sample_id_to_atom_id == atom_ids

    def test_construction_with_tuple_atom_ids(self, sample_space):
        """Test constructing a SigmaAlgebra with tuple atom IDs."""
        atom_ids = {
            "omega0": (0, 0),
            "omega1": (0, 1),
            "omega2": (1, 0),
            "omega3": (1, 1),
        }
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert sigma_algebra.sample_id_to_atom_id == atom_ids

    def test_construction_with_mixed_hashable_atom_ids(self, sample_space):
        """Test constructing a SigmaAlgebra with mixed hashable atom IDs."""
        atom_ids = {"omega0": 0, "omega1": "special", "omega2": 0, "omega3": (1, 2)}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        assert sigma.sample_id_to_atom_id == atom_ids


class TestNumAtoms:

    def test_num_atoms(self):
        """Test that num_atoms property returns correct number of atoms."""
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_algebra = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert sigma_algebra.num_atoms == 2


class TestAtomIds:

    def test_atom_ids_property(self):
        """Test that atom_ids property returns correct list of unique atom IDs."""
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_algebra = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert set(sigma_algebra.atom_ids) == {0, 1}


class TestAtomIdToSampleIds:

    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_returns_dict(self, sigma_algebra):
        """Test that atom_id_to_sample_ids returns a dictionary."""
        result = sigma_algebra.atom_id_to_sample_ids
        assert isinstance(result, dict)

    def test_has_correct_number_of_atoms(self, sigma_algebra):
        """Test that the dictionary has the correct number of atoms."""
        result = sigma_algebra.atom_id_to_sample_ids
        assert len(result) == 2

    def test_keys_are_atom_ids(self, sigma_algebra):
        """Test that the keys of the dictionary are the correct atom IDs."""
        result = sigma_algebra.atom_id_to_sample_ids
        assert set(result.keys()) == {0, 1}

    def test_values_are_lists(self, sigma_algebra):
        """Test that the values of the dictionary are lists of sample IDs."""
        result = sigma_algebra.atom_id_to_sample_ids
        for sample_list in result.values():
            assert isinstance(sample_list, list)

    def test_atoms_have_correct_samples(self, sigma_algebra):
        """Test that each atom ID maps to the correct sample IDs."""
        result = sigma_algebra.atom_id_to_sample_ids
        assert set(result[0]) == {"omega0", "omega1"}
        assert set(result[1]) == {"omega2", "omega3"}

    def test_with_string_atom_ids(self):
        """Test atom_id_to_sample_ids with string atom IDs."""
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        result = sigma.atom_id_to_sample_ids
        assert set(result.keys()) == {"A", "B"}
        assert set(result["A"]) == {"omega0", "omega1"}
        assert set(result["B"]) == {"omega2"}

    def test_with_tuple_atom_ids(self):
        """Test atom_id_to_sample_ids with tuple atom IDs."""
        space = SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": (0, 0), "omega1": (1, 1)}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        result = sigma.atom_id_to_sample_ids
        assert (0, 0) in result
        assert (1, 1) in result


class TestAtomIdToEvent:

    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_returns_dict(self, sigma_algebra):
        """Test that atom_id_to_event returns a dictionary."""
        result = sigma_algebra.atom_id_to_event
        assert isinstance(result, dict)

    def test_has_correct_number_of_atoms(self, sigma_algebra):
        """Test that the dictionary has the correct number of atoms."""
        result = sigma_algebra.atom_id_to_event
        assert len(result) == 2

    def test_keys_are_atom_ids(self, sigma_algebra):
        """Test that the keys of the dictionary are the correct atom IDs."""
        result = sigma_algebra.atom_id_to_event
        assert set(result.keys()) == {0, 1}

    def test_values_are_events(self, sigma_algebra):
        """Test that the values of the dictionary are Event instances."""
        result = sigma_algebra.atom_id_to_event
        for event in result.values():
            assert isinstance(event, Event)

    def test_atoms_have_correct_indices(self, sigma_algebra):
        """Test that each atom ID maps to the correct Event."""
        result = sigma_algebra.atom_id_to_event
        atom_0 = result[0]
        atom_1 = result[1]
        assert set(atom_0.data) == {"omega0", "omega1"}
        assert set(atom_1.data) == {"omega2", "omega3"}

    def test_event_names_are_atom_ids(self, sigma_algebra):
        """Test that each Event has the correct name corresponding to its atom ID."""
        result = sigma_algebra.atom_id_to_event
        assert result[0].name == 0
        assert result[1].name == 1


class TestAtomIdToCardinality:

    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_returns_dict(self, sigma_algebra):
        """Test that atom_id_to_cardinality returns a dictionary."""
        result = sigma_algebra.atom_id_to_cardinality
        assert isinstance(result, dict)

    def test_has_correct_keys(self, sigma_algebra):
        """Test that the keys of the dictionary are the correct atom IDs."""
        result = sigma_algebra.atom_id_to_cardinality
        assert set(result.keys()) == {0, 1}

    def test_values_are_integers(self, sigma_algebra):
        """Test that the values of the dictionary are integers."""
        result = sigma_algebra.atom_id_to_cardinality
        for cardinality in result.values():
            assert isinstance(cardinality, int)

    def test_correct_cardinalities(self, sigma_algebra):
        """Test that each atom ID maps to the correct cardinality."""
        result = sigma_algebra.atom_id_to_cardinality
        assert result[0] == 2
        assert result[1] == 2

    def test_with_uneven_partition(self):
        """Test atom_id_to_cardinality with uneven atom sizes."""
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0, "omega3": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        result = sigma.atom_id_to_cardinality
        assert result[0] == 3
        assert result[1] == 1


class TestToAtoms:

    def test_to_atoms(self):
        """Test that to_atoms method returns correct list of Events."""
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_algebra = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        atoms = sigma_algebra.to_atoms()
        assert isinstance(atoms, list)
        assert len(atoms) == 2
        for atom in atoms:
            assert isinstance(atom, Event)
        atom_samples = [set(atom.data) for atom in atoms]
        assert {frozenset(s) for s in atom_samples} == {
            frozenset({"omega0", "omega1"}),
            frozenset({"omega2", "omega3"}),
        }


class TestIsMeasurable:

    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 1, "omega2": 1, "omega3": 2}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_atom_is_measurable(self, sigma_algebra):
        """Test that an atom is measurable."""
        event = Event(
            sample_space=sigma_algebra.sample_space, indices=["omega1", "omega2"]
        )
        assert sigma_algebra.is_measurable(event)

    def test_union_of_atoms_is_measurable(self, sigma_algebra):
        """Test that the union of atoms is measurable."""
        event = Event(
            sample_space=sigma_algebra.sample_space,
            indices=["omega0", "omega1", "omega2"],
        )
        assert sigma_algebra.is_measurable(event)

    def test_partial_atom_is_not_measurable(self, sigma_algebra):
        """Test that a partial atom is not measurable."""
        event = Event(sample_space=sigma_algebra.sample_space, indices=["omega1"])
        assert not sigma_algebra.is_measurable(event)

    def test_empty_event_is_measurable(self, sigma_algebra):
        """Test that the empty event is measurable."""
        event = Event(sample_space=sigma_algebra.sample_space, indices=[])
        assert sigma_algebra.is_measurable(event)

    def test_full_space_is_measurable(self, sigma_algebra):
        """Test that the full sample space is measurable."""
        event = Event(
            sample_space=sigma_algebra.sample_space,
            indices=list(sigma_algebra.sample_space.data),
        )
        assert sigma_algebra.is_measurable(event)


class TestGetAtomContaining:

    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_get_atom_containing(self, sigma_algebra):
        """Test that get_atom_containing returns the correct atom."""
        atom = sigma_algebra.get_atom_containing("omega0")
        assert isinstance(atom, Event)
        assert set(atom.data) == {"omega0", "omega1"}

    def test_get_atom_containing_returns_correct_atom(self, sigma_algebra):
        """Test that get_atom_containing returns the correct atom for a given sample ID."""
        atom = sigma_algebra.get_atom_containing("omega2")
        assert set(atom.data) == {"omega2", "omega3"}

    def test_get_atom_containing_each_sample_point(self, sigma_algebra):
        """Test that get_atom_containing works for each sample point."""
        atom0 = sigma_algebra.get_atom_containing("omega0")
        atom1 = sigma_algebra.get_atom_containing("omega1")
        atom2 = sigma_algebra.get_atom_containing("omega2")
        atom3 = sigma_algebra.get_atom_containing("omega3")
        assert atom0 == atom1
        assert atom2 == atom3
        assert atom0 != atom2


class TestFromPandas:

    def test_from_pandas_with_custom_name(self):
        """Test that from_pandas creates a SigmaAlgebra correctly from a pd.Series with custom name."""
        data = pd.Series(
            data={"omega0": 0, "omega1": 1, "omega2": 1},
            name="atoms",
        )
        sigma_algebra = SigmaAlgebra.from_pandas(data=data, name="G")
        assert isinstance(sigma_algebra, SigmaAlgebra)
        assert sigma_algebra.name == "G"
        assert sigma_algebra.sample_space.name == "Omega"
        assert sigma_algebra.sample_id_to_atom_id == {
            "omega0": 0,
            "omega1": 1,
            "omega2": 1,
        }
        assert sigma_algebra.data.name == "atoms"
        pd.testing.assert_series_equal(sigma_algebra.data, data)

    def test_from_pandas_default_name(self):
        """Test that from_pandas creates a SigmaAlgebra correctly from a pd.Series with default name."""
        data = pd.Series(data={"omega0": 0, "omega1": 1, "omega2": 1})
        sigma_algebra = SigmaAlgebra.from_pandas(data=data)
        assert isinstance(sigma_algebra, SigmaAlgebra)
        assert sigma_algebra.name == "F"
        assert sigma_algebra.sample_space.name == "Omega"
        assert sigma_algebra.sample_id_to_atom_id == {
            "omega0": 0,
            "omega1": 1,
            "omega2": 1,
        }
        assert sigma_algebra.data.name is None
        pd.testing.assert_series_equal(sigma_algebra.data, data)


class TestPowerSet:

    def test_power_set(self):
        """Test that power_set method creates the correct SigmaAlgebra."""
        sample_space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma_algebra = SigmaAlgebra.power_set(sample_space)
        assert sigma_algebra.sample_id_to_atom_id == {
            "omega0": 0,
            "omega1": 1,
            "omega2": 2,
        }


class TestTrivial:

    def test_trivial_creation(self):
        """Test that trivial method creates the correct SigmaAlgebra."""
        sample_space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma_algebra = SigmaAlgebra.trivial(sample_space)
        assert sigma_algebra.sample_id_to_atom_id == {
            "omega0": 0,
            "omega1": 0,
            "omega2": 0,
        }


class TestIteration:

    @pytest.fixture
    def sigma_algebra(self):
        sample_space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    def test_iteration_yields_tuples(self, sigma_algebra):
        """Test that iterating over the SigmaAlgebra yields tuples of (atom_id, Event)."""
        for atom_id, event in sigma_algebra:
            assert isinstance(atom_id, Hashable)
            assert isinstance(event, Event)

    def test_iteration_covers_all_atoms(self, sigma_algebra):
        """Test that iteration covers all atom IDs."""
        atom_ids_seen = set()
        for atom_id, _ in sigma_algebra:
            atom_ids_seen.add(atom_id)
        assert atom_ids_seen == {0, 1}

    def test_can_convert_to_dict(self, sigma_algebra):
        """Test that the SigmaAlgebra can be converted to a dictionary."""
        atoms_dict = dict(sigma_algebra)
        assert len(atoms_dict) == 2
        assert all(isinstance(event, Event) for event in atoms_dict.values())

    def test_iteration_matches_atom_id_to_event(self, sigma_algebra):
        """Test that iteration matches the atom_id_to_event property."""
        from_iter = dict(sigma_algebra)
        from_property = sigma_algebra.atom_id_to_event
        assert set(from_iter.keys()) == set(from_property.keys())
        for atom_id in from_iter:
            assert set(from_iter[atom_id].data) == set(from_property[atom_id].data)

    def test_iteration_with_string_atom_ids(self):
        """Test iteration works with string atom IDs."""
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        atom_ids_seen = []
        for atom_id, _ in sigma:
            atom_ids_seen.append(atom_id)
        assert set(atom_ids_seen) == {"A", "B"}

    def test_iteration_order_is_consistent(self, sigma_algebra):
        """Test that iteration order is consistent across multiple iterations."""
        keys1 = [atom_id for atom_id, _ in sigma_algebra]
        keys2 = [atom_id for atom_id, _ in sigma_algebra]
        assert keys1 == keys2


class TestEquality:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2"])

    def test_equality_same_components(self, sample_space):
        """Test that two SigmaAlgebras with the same components are equal."""
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_algebra1 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        sigma_algebra2 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert sigma_algebra1 == sigma_algebra2

    def test_equality_different_atom_ids(self, sample_space):
        """Test that two SigmaAlgebras with different atom IDs are not equal."""
        atom_ids1 = {"omega0": 0, "omega1": 0, "omega2": 1}
        atom_ids2 = {"omega0": 0, "omega1": 1, "omega2": 1}
        sigma_algebra1 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids1, sample_space=sample_space
        )
        sigma_algebra2 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids2, sample_space=sample_space
        )
        assert sigma_algebra1 != sigma_algebra2

    def test_equality_different_sample_spaces(self):
        """Test that two SigmaAlgebras with different sample spaces are not equal."""
        sample_space1 = SampleSpace(["omega0", "omega1"])
        sample_space2 = SampleSpace(["a", "b"])
        atom_ids1 = {"omega0": 0, "omega1": 0}
        atom_ids2 = {"a": 0, "b": 0}
        sigma_algebra1 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids1, sample_space=sample_space1
        )
        sigma_algebra2 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids2, sample_space=sample_space2
        )
        assert sigma_algebra1 != sigma_algebra2

    def test_equality_with_non_sigma_algebra(self, sample_space):
        """Test that a SigmaAlgebra is not equal to a non-SigmaAlgebra object."""
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        assert sigma_algebra != "not a sigma algebra"
        assert sigma_algebra != 123
        assert sigma_algebra != sample_space


class TestOrderRelations:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    def test_le_trivial_and_power_set(self, sample_space):
        """Test that trivial SigmaAlgebra is less than or equal to power set SigmaAlgebra."""
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial <= power_set
        assert not power_set <= trivial

    def test_le_reflexive(self, sample_space):
        """Test that a SigmaAlgebra is less than or equal to itself."""
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma_algebra = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        assert sigma_algebra <= sigma_algebra

    def test_le_coarser_and_finer(self, sample_space):
        """Test that a coarser SigmaAlgebra is less than or equal to a finer SigmaAlgebra."""
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=fine_atom_ids
        )
        assert coarse <= fine
        assert not fine <= coarse

    def test_le_transitive(self):
        """Test that the less than or equal relation is transitive."""
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        A = SigmaAlgebra.trivial(sample_space=sample_space)
        B_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        B = SigmaAlgebra(sample_space=sample_space, sample_id_to_atom_id=B_atom_ids)
        C = SigmaAlgebra.power_set(sample_space=sample_space)
        assert A <= B
        assert B <= C
        assert A <= C

    def test_lt_proper_sub_algebra(self, sample_space):
        """Test that trivial SigmaAlgebra is less than power set SigmaAlgebra."""
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial < power_set
        assert not power_set < trivial

    def test_lt_not_proper_when_equal(self, sample_space):
        """Test that a SigmaAlgebra is not less than itself."""
        sigma_algebra = SigmaAlgebra.trivial(sample_space=sample_space)
        assert not sigma_algebra < sigma_algebra

    def test_ge_power_set_and_trivial(self, sample_space):
        """Test that power set SigmaAlgebra is greater than or equal to trivial SigmaAlgebra."""
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert power_set >= trivial
        assert not trivial >= power_set

    def test_ge_reflexive(self, sample_space):
        """Test that a SigmaAlgebra is greater than or equal to itself."""
        sigma_algebra = SigmaAlgebra.trivial(sample_space=sample_space)
        assert sigma_algebra >= sigma_algebra

    def test_ge_finer_and_coarser(self, sample_space):
        """Test that a finer SigmaAlgebra is greater than or equal to a coarser SigmaAlgebra."""
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=fine_atom_ids
        )
        assert fine >= coarse
        assert not coarse >= fine

    def test_gt_proper_super_algebra(self, sample_space):
        """Test that power set SigmaAlgebra is greater than trivial SigmaAlgebra."""
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert power_set > trivial
        assert not trivial > power_set

    def test_gt_not_proper_when_equal(self, sample_space):
        """Test that a SigmaAlgebra is not greater than itself."""
        sigma_algebra = SigmaAlgebra.trivial(sample_space=sample_space)
        assert not sigma_algebra > sigma_algebra

    def test_incomparable_sigma_algebras(self, sample_space):
        """Test that two incomparable SigmaAlgebras are neither less than nor greater than each other."""
        sigma1_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma1 = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=sigma1_atom_ids
        )
        sigma2_atom_ids = {"s0": 0, "s1": 1, "s2": 0, "s3": 1}
        sigma2 = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=sigma2_atom_ids
        )
        assert not sigma1 <= sigma2
        assert not sigma2 <= sigma1
        assert not sigma1 >= sigma2
        assert not sigma2 >= sigma1
        assert not sigma1 < sigma2
        assert not sigma2 < sigma1
        assert not sigma1 > sigma2
        assert not sigma2 > sigma1

    def test_three_level_chain(self):
        """Test a chain of three SigmaAlgebras with proper ordering."""
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        middle_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        middle = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=middle_atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial <= middle <= power_set
        assert trivial < middle < power_set
        assert power_set >= middle >= trivial
        assert power_set > middle > trivial

    def test_antisymmetry(self, sample_space):
        """Test that if two SigmaAlgebras are less than or equal to each other, they are equal."""
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma_algebra1 = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        sigma_algebra2 = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        assert sigma_algebra1 <= sigma_algebra2
        assert sigma_algebra2 <= sigma_algebra1
        assert sigma_algebra1 == sigma_algebra2


class TestValidation:
    pass
