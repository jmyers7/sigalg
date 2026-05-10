from collections.abc import Hashable

import pandas as pd
import pytest
from pydantic import ValidationError

from sigalg.core import Event, Lattice, SampleSpace, SigmaAlgebra

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters."""
        F = SigmaAlgebra()

        assert F.name == "F"
        assert F.sample_space is None
        assert F.data is None
        assert F.sample_id_to_atom_id is None
        assert F.atom_space is None
        assert F.num_atoms is None
        assert F.atom_ids is None
        assert F.atom_id_to_sample_ids is None
        assert F.atom_id_to_event is None
        assert F.atom_id_to_cardinality is None
        assert F.is_power_set is None

    def test_constructor_with_custom_parameters(self):
        """Test the constructor with a custom parameters."""
        Omega = SampleSpace().from_sequence(size=3)
        G = SigmaAlgebra(sample_space=Omega, name="G")

        assert G.name == "G"
        assert G.sample_space == Omega
        assert G.data is None
        assert G.sample_id_to_atom_id is None
        assert G.atom_space is None
        assert G.num_atoms is None
        assert G.atom_ids is None
        assert G.atom_id_to_sample_ids is None
        assert G.atom_id_to_event is None
        assert G.atom_id_to_cardinality is None
        assert G.is_power_set is None


class TestFromDict:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    def test_constructor_integer_atom_ids_default_name(self, Omega):
        """Test constructor with integer atom IDs and default name."""
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

        assert F.sample_space == Omega
        assert F.sample_id_to_atom_id == atom_ids
        assert F.name == "F"

    def test_constructor_string_atom_ids_custom_name(self, Omega):
        """Test constructor with string atom IDs and custom name."""
        atom_ids = {0: "A", 1: "A", 2: "B", 3: "B"}
        G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
            sample_id_to_atom_id=atom_ids
        )

        assert G.sample_space == Omega
        assert G.sample_id_to_atom_id == atom_ids
        assert G.name == "G"

    def test_constructor_tuple_atom_ids(self, Omega):
        """Test constructor with tuple atom IDs."""
        atom_ids = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1),
        }
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

        assert F.sample_space == Omega
        assert F.sample_id_to_atom_id == atom_ids

    def test_constructor_mixed_hashable_atom_ids(self, Omega):
        """Test constructor with mixed hashable atom IDs."""
        atom_ids = {0: 0, 1: "special", 2: 0, 3: (1, 2)}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

        assert F.sample_space == Omega
        assert F.sample_id_to_atom_id == atom_ids

    def test_invalid_missing_sample_id_raises(self):
        """Test that missing sample ID raises ValidationError."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: 0, 1: 0, 5: 1}
        with pytest.raises(
            ValidationError,
            match="mapping must contain an entry for every sample index",
        ):
            SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    def test_invalid_incomplete_mapping_raises(self):
        """Test that incomplete mapping raises ValidationError."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: 0, 1: 0}
        with pytest.raises(
            ValidationError,
            match="mapping must contain an entry for every sample index",
        ):
            SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    def test_invalid_unhashable_atom_id_raises(self):
        """Test that unhashable atom ID raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: [1, 2], 1: 0, 2: 1}
        with pytest.raises(
            TypeError, match="All values in the mapping must be Hashable"
        ):
            SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)


class TestFromPandas:
    def test_from_pandas(self):
        """Test from_pandas method."""
        data = pd.Series(data={0: 0, 1: 1, 2: 1})
        G = SigmaAlgebra(name="G").from_pandas(data=data)
        expected_sample_space = SampleSpace().from_sequence(size=3)
        expected_data = pd.Series(
            data={0: 0, 1: 1, 2: 1},
            index=pd.Index([0, 1, 2]),
            name="atom ID",
        )

        assert G.name == "G"
        assert G.sample_space == expected_sample_space
        pd.testing.assert_series_equal(G.data, expected_data)

    def test_invalid_input_raises(self):
        """Test that invalid input raises an exception."""
        with pytest.raises(TypeError):
            SigmaAlgebra.from_pandas(data=["not", "a", "series"])


class TestPowerSet:
    def test_power_set_three_samples_default_name(self):
        """Test that power_set method creates the correct SigmaAlgebra for three samples."""
        Omega = SampleSpace().from_sequence(size=3)
        power_set = SigmaAlgebra.power_set(Omega)

        assert power_set.name == "power_set"
        assert power_set.num_atoms == 3
        assert power_set.sample_space == Omega
        for idx, sample_id in enumerate(Omega.data):
            assert power_set.sample_id_to_atom_id[sample_id] == idx

    def test_power_set_four_samples_custom_name(self):
        """Test that power_set method creates the correct SigmaAlgebra for four samples with custom name."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra.power_set(Omega, name="F")

        assert F.name == "F"
        assert F.num_atoms == 4
        assert F.sample_space == Omega
        for idx, sample_id in enumerate(Omega.data):
            assert F.sample_id_to_atom_id[sample_id] == idx

    def test_power_set_single_sample_point(self):
        """Test that power_set method creates the correct SigmaAlgebra for single sample point."""
        Omega = SampleSpace().from_sequence(size=1)
        power_set = SigmaAlgebra.power_set(Omega)

        assert power_set.name == "power_set"
        assert power_set.num_atoms == 1
        assert power_set.sample_space == Omega
        for idx, sample_id in enumerate(Omega.data):
            assert power_set.sample_id_to_atom_id[sample_id] == idx


class TestTrivial:
    def test_trivial_creation_three_samples_default_name(self):
        """Test that trivial method creates the correct SigmaAlgebra for three samples."""
        Omega = SampleSpace().from_sequence(size=3)
        trivial = SigmaAlgebra.trivial(Omega)

        assert trivial.name == "trivial"
        assert trivial.num_atoms == 1
        assert trivial.sample_space == Omega
        unique_atom_ids = set(trivial.sample_id_to_atom_id.values())
        assert len(unique_atom_ids) == 1

    def test_trivial_creation_four_samples_custom_name(self):
        """Test that trivial method creates the correct SigmaAlgebra for four samples with custom name."""
        Omega = SampleSpace().from_sequence(size=4)
        F = SigmaAlgebra.trivial(Omega, name="F")

        assert F.name == "F"
        assert F.num_atoms == 1
        assert F.sample_space == Omega
        unique_atom_ids = set(F.sample_id_to_atom_id.values())
        assert len(unique_atom_ids) == 1

    def test_trivial_creation_single_sample_point(self):
        """Test that trivial method creates the correct SigmaAlgebra for single sample point."""
        Omega = SampleSpace().from_sequence(size=1)
        trivial = SigmaAlgebra.trivial(Omega)

        assert trivial.name == "trivial"
        assert trivial.num_atoms == 1
        assert trivial.sample_space == Omega
        unique_atom_ids = set(trivial.sample_id_to_atom_id.values())
        assert len(unique_atom_ids) == 1


# --------------------- test properties --------------------- #


class TestSampleSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def atom_ids(self):
        return {0: 0, 1: 0, 2: 1, 3: 1}

    @pytest.fixture
    def data(self):
        return pd.Series([0, 0, 1, 1])

    def test_sample_space_getter_on_sigma_algebra_with_data(self, Omega, atom_ids):
        """Test sample_space property getter on SigmaAlgebra with data."""
        F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)

        assert F.sample_space == Omega

    def test_sample_space_getter_on_sigma_algebra_without_data(self, Omega):
        """Test sample_space property getter on SigmaAlgebra without data."""
        F = SigmaAlgebra(sample_space=Omega)

        assert F.sample_space == Omega

    def test_sample_space_setter_on_empty_sigma_algebra(self, Omega):
        """Test sample_space property setter on empty SigmaAlgebra."""
        F = SigmaAlgebra()
        F.sample_space = Omega

        assert F.sample_space == Omega

    def test_sample_space_setter_on_sigma_algebra_from_dict(self, Omega, atom_ids):
        """Test sample_space property setter on SigmaAlgebra created from dict."""
        F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)
        Omega_new = SampleSpace().from_list(["a", "b", "c", "d"])
        data_new = pd.Series([0, 0, 1, 1], index=Omega_new.data, name="atom ID")
        atom_IDs_new = dict(zip(Omega_new.data, atom_ids.values()))
        F.sample_space = Omega_new

        assert F.sample_space == Omega_new
        pd.testing.assert_series_equal(F.data, data_new)
        assert F.sample_id_to_atom_id == atom_IDs_new

    def test_sample_space_setter_on_sigma_algebra_from_pandas(self, Omega, data):
        """Test sample_space property setter on SigmaAlgebra created from pandas."""
        F = SigmaAlgebra(sample_space=Omega).from_pandas(data)
        Omega_new = SampleSpace().from_list(["a", "b", "c", "d"])
        data_new = pd.Series([0, 0, 1, 1], index=Omega_new.data, name="atom ID")
        atom_IDs_new = dict(zip(Omega_new.data, data.values))
        F.sample_space = Omega_new

        assert F.sample_space == Omega_new
        pd.testing.assert_series_equal(F.data, data_new)
        assert F.sample_id_to_atom_id == atom_IDs_new


class TestSampleIdToAtomId:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def sample_id_to_atom_id(self):
        return {0: 0, 1: 0, 2: 1, 3: 1}

    def test_sample_id_to_atom_id_and_from_dict(self, Omega, sample_id_to_atom_id):
        """Test sample_id_to_atom_id dictionary is the same passed into the from_dict constructor."""
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=sample_id_to_atom_id
        )

        assert F.sample_id_to_atom_id == sample_id_to_atom_id

    def test_sample_id_to_atom_id_and_from_pandas(self, sample_id_to_atom_id):
        """Test sample_id_to_atom_id dictionary and compatibility with from_pandas constructor."""
        data = pd.Series([0, 0, 1, 1])
        F = SigmaAlgebra().from_pandas(data=data)

        assert F.sample_id_to_atom_id == sample_id_to_atom_id


class TestData:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def data(self):
        return pd.Series([0, 0, 1, 1], index=pd.Index([0, 1, 2, 3], name="ID"))

    def test_data_and_from_pandas_with_no_sample_space(self, data):
        """Test data attribute is the same passed into the from_pandas constructor with added names and no sample space."""
        F = SigmaAlgebra().from_pandas(data=data)
        data.name = "atom ID"

        pd.testing.assert_series_equal(F.data, data)

    def test_data_and_from_pandas_with_sample_space(self, Omega, data):
        """Test data attribute is the same passed into the from_pandas constructor with added names and sample space."""
        F = SigmaAlgebra(sample_space=Omega).from_pandas(data=data)
        data.name = "atom ID"

        pd.testing.assert_series_equal(F.data, data)

    def test_data_from_dict(self, Omega, data):
        """Test data attribute and from_dict compatibility"""
        sample_id_to_atom_id = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=sample_id_to_atom_id
        )
        data.name = "atom ID"
        data.index.name = "sample"

        pd.testing.assert_series_equal(F.data, data)


class TestAtomSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    def test_atom_space_with_ascending_integer_atom_ids(self, Omega):
        """Test atom_space attribute with ascending integer atom identifiers."""
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            }
        )
        expected_atom_space = SampleSpace(
            name="atom_space", data_name="atom ID"
        ).from_list([0, 1])

        assert F.atom_space == expected_atom_space

    def test_atom_space_with_non_ascending_integer_atom_ids(self, Omega):
        """Test atom_space attribute with non-ascending integer atom identifiers."""
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 1,
                1: 1,
                2: 0,
                3: 0,
            }
        )
        expected_atom_space = SampleSpace(
            name="atom_space", data_name="atom ID"
        ).from_list([1, 0])

        assert F.atom_space == expected_atom_space

    def test_atom_space_with_string_atom_ids(self, Omega):
        """Test atom_space attribute with string atom identifiers."""
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: "a",
                1: "d",
                2: "a",
                3: "c",
            }
        )
        expected_atom_space = SampleSpace(
            name="atom_space", data_name="atom ID"
        ).from_list(["a", "d", "c"])

        assert F.atom_space == expected_atom_space


class TestNumAtoms:
    def test_num_atoms_two_atoms(self):
        """Test that num_atoms returns 2 for two-atom sigma algebra."""
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        expected_num_atoms = 2

        assert F.num_atoms == expected_num_atoms

    def test_num_atoms_trivial_one_atom(self):
        """Test that num_atoms returns 1 for trivial sigma algebra."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: 0, 1: 0, 2: 0}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        expected_num_atoms = 1

        assert F.num_atoms == expected_num_atoms

    def test_num_atoms_power_set_three_atoms(self):
        """Test that num_atoms returns 3 for three-atom power set."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: 0, 1: 1, 2: 2}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        expected_num_atoms = 3

        assert F.num_atoms == expected_num_atoms

    def test_num_atoms_single_sample_point(self):
        """Test that num_atoms returns 1 for single sample point."""
        Omega = SampleSpace().from_sequence(size=1)
        atom_ids = {0: 0}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        expected_num_atoms = 1

        assert F.num_atoms == expected_num_atoms


class TestAtomIds:
    def test_atom_ids_integer_atom_ids(self):
        """Test atom_ids returns correct set of integer IDs."""
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        expected_atom_ids = [0, 1]
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

        assert F.atom_ids == expected_atom_ids

    def test_atom_ids_string_atom_ids(self):
        """Test atom_ids returns correct set of string IDs."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: "A", 1: "A", 2: "B"}
        expected_atom_ids = ["A", "B"]
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

        assert F.atom_ids == expected_atom_ids

    def test_atom_ids_tuple_atom_ids(self):
        """Test atom_ids returns correct set of tuple IDs."""
        Omega = SampleSpace().from_sequence(size=2)
        atom_ids = {0: (0, 0), 1: (1, 1)}
        expected_atom_ids = [(0, 0), (1, 1)]
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

        assert F.atom_ids == expected_atom_ids

    def test_atom_ids_single_atom(self):
        """Test atom_ids returns single atom ID."""
        Omega = SampleSpace().from_sequence(size=1)
        atom_ids = {0: 0}
        expected_atom_ids = [0]
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

        assert F.atom_ids == expected_atom_ids


class TestAtomIdDictionaries:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="Omega", data_name="sample").from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        return SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    def test_atom_id_to_sample_ids(self, F):
        """Test that atom_id_to_sample_ids returns correct mapping."""
        atom_id_to_sample_ids = F.atom_id_to_sample_ids
        expected = {
            0: [0, 1],
            1: [2, 3],
        }

        assert atom_id_to_sample_ids == expected

    def test_atom_id_to_event(self, F, Omega):
        """Test that atom_id_to_event returns correct mapping."""
        atom_id_to_event = F.atom_id_to_event
        expected = {
            0: Event(sig_alg=F, name=0).from_list([0, 1]),
            1: Event(sig_alg=F, name=1).from_list([2, 3]),
        }

        assert atom_id_to_event == expected

    def test_atom_id_to_cardinality(self, F):
        """Test that atom_id_to_cardinality returns correct mapping."""
        atom_id_to_cardinality = F.atom_id_to_cardinality
        expected = {
            0: 2,
            1: 2,
        }

        assert atom_id_to_cardinality == expected


class TestToAtoms:
    def test_to_atoms_two_equal_atoms(self):
        """Test that to_atoms returns correct list of Events for two equal atoms."""
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        atoms = F.to_atoms
        expected = [
            Event(sig_alg=F, name=0).from_list([0, 1]),
            Event(sig_alg=F, name=1).from_list([2, 3]),
        ]
        assert atoms == expected

    def test_to_atoms_trivial_single_atom(self):
        """Test that to_atoms returns correct list for trivial single atom."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: 0, 1: 0, 2: 0}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        atoms = F.to_atoms
        expected = [
            Event(sig_alg=F, name=0).from_list([0, 1, 2]),
        ]
        assert atoms == expected

    def test_to_atoms_power_set_three_atoms(self):
        """Test that to_atoms returns correct list for power set with three atoms."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: 0, 1: 1, 2: 2}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        atoms = F.to_atoms
        expected = [
            Event(sig_alg=F, name=0).from_list([0]),
            Event(sig_alg=F, name=1).from_list([1]),
            Event(sig_alg=F, name=2).from_list([2]),
        ]
        assert atoms == expected

    def test_to_atoms_uneven_partition(self):
        """Test that to_atoms returns correct list for uneven partition."""
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {0: 0, 1: 0, 2: 0, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        atoms = F.to_atoms
        expected = [
            Event(sig_alg=F, name=0).from_list([0, 1, 2]),
            Event(sig_alg=F, name=1).from_list([3]),
        ]
        assert atoms == expected


# --------------------- test atom and event methods --------------------- #


class TestGetEvent:
    @pytest.fixture
    def F(self):
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
        }
        return SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    def test_get_event_measurable_single_atom(self, F):
        """Test get_event with indices forming a single atom."""
        A = F.get_event([0, 1], name="A")

        assert isinstance(A, Event)
        assert A.name == "A"
        assert A.indices == [0, 1]
        assert A.sig_alg == F

    def test_get_event_measurable_second_atom(self, F):
        """Test get_event with indices forming the second atom."""
        B = F.get_event([2, 3], name="B")

        assert isinstance(B, Event)
        assert B.name == "B"
        assert B.indices == [2, 3]
        assert B.sig_alg == F

    def test_get_event_measurable_union_of_atoms(self, F):
        """Test get_event with indices forming a union of multiple atoms."""
        C = F.get_event([0, 1, 2, 3], name="C")

        assert isinstance(C, Event)
        assert C.name == "C"
        assert C.indices == [0, 1, 2, 3]
        assert C.sig_alg == F

    def test_get_event_measurable_empty_event(self, F):
        """Test get_event with empty indices."""
        empty = F.get_event([], name="empty")

        assert isinstance(empty, Event)
        assert empty.name == "empty"
        assert len(empty) == 0
        assert empty.sig_alg == F

    def test_get_event_custom_name(self, F):
        """Test get_event with custom name parameter."""
        event = F.get_event([0, 1], name="CustomEvent")

        assert event.name == "CustomEvent"

    def test_get_event_default_name(self, F):
        """Test get_event with default name."""
        event = F.get_event([2, 3])

        assert event.name == "A"

    def test_get_event_single_point_from_atom(self, F):
        """Test get_event with single point from a two-point atom."""
        with pytest.raises(ValueError, match="do not form a measurable event"):
            F.get_event([0], name="invalid")

    def test_invalid_indices_not_in_sample_space(self, F):
        """Test that indices not in sample space raise ValueError."""
        with pytest.raises(
            ValueError, match="The event is not a subset of the sample space"
        ):
            F.get_event([0, 1, 5], name="invalid")

    def test_invalid_non_measurable_event(self, F):
        """Test that non-measurable event raises ValueError."""
        with pytest.raises(ValueError, match="do not form a measurable event"):
            F.get_event([0, 2], name="invalid")

    def test_invalid_partial_atoms(self, F):
        """Test that partial atoms are not measurable."""
        with pytest.raises(ValueError, match="do not form a measurable event"):
            F.get_event([1, 3], name="invalid")

    def test_get_event_power_set(self):
        """Test get_event on power set sigma-algebra allows any subset."""
        Omega = SampleSpace().from_sequence(size=3)
        F_power = SigmaAlgebra.power_set(Omega)
        A = F_power.get_event([0], name="A")
        B = F_power.get_event([0, 2], name="B")

        assert set(A.indices) == {0}
        assert set(B.indices) == {0, 2}


class TestGetAtomContaining:
    @pytest.fixture
    def F(self):
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
        }
        return SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    def test_get_atom_containing_first_atom_point(self, F):
        """Test that get_atom_containing returns the correct atom for first atom point."""
        atom = F.get_atom_containing(0)
        expected_atom = Event(sig_alg=F, name=0).from_list([0, 1])

        assert atom == expected_atom

    def test_get_atom_containing_first_atom_point_second(self, F):
        """Test that get_atom_containing returns the correct atom for first atom second point."""
        atom = F.get_atom_containing(1)
        expected_atom = Event(sig_alg=F, name=0).from_list([0, 1])

        assert atom == expected_atom

    def test_get_atom_containing_second_atom_point(self, F):
        """Test that get_atom_containing returns the correct atom for second atom point."""
        atom = F.get_atom_containing(2)
        expected_atom = Event(sig_alg=F, name=1).from_list([2, 3])

        assert atom == expected_atom

    def test_get_atom_containing_second_atom_point_second(self, F):
        """Test that get_atom_containing returns the correct atom for second atom second point."""
        atom = F.get_atom_containing(3)
        expected_atom = Event(sig_alg=F, name=1).from_list([2, 3])

        assert atom == expected_atom

    def test_invalid_sample_id_not_in_sample_space(self, F):
        """Test that invalid sample ID not in sample space raises ValueError."""
        with pytest.raises(ValueError):
            F.get_atom_containing(5)

    def test_invalid_sample_id_non_existent(self, F):
        """Test that non-existent sample ID raises ValueError."""
        with pytest.raises(ValueError):
            F.get_atom_containing("invalid")


# --------------------- test measurability methods --------------------- #


class TestIsMeasurable:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 1,
                2: 1,
                3: 2,
            }
        )

    def test_is_measurable_measurable_event(self, F):
        """Test is_measurable method with a measurable event."""
        A = Event(sig_alg=SigmaAlgebra.power_set(F.sample_space)).from_list([0, 1, 2])
        assert F.is_measurable(event=A)

    def test_is_measurable_nonmeasurable_event(self, Omega, F):
        """Test is_measurable method with a non-measurable event."""
        power_set = SigmaAlgebra.power_set(Omega)
        A = Event(sig_alg=power_set).from_list([2, 3])
        assert not F.is_measurable(event=A)

    def test_is_measurable_with_list_of_indices(self, F):
        """Test is_measurable method with a list of indices."""
        assert F.is_measurable(event_list=[0, 1, 2])

    def test_is_measurable_with_nonmeasurable_list_of_indices(self, F):
        """Test is_measurable method with a non-measurable list of indices."""
        assert not F.is_measurable(event_list=[2, 3])

    def test_is_measurable_empty_event(self, F):
        """Test is_measurable method with an empty event."""
        A = Event(sig_alg=SigmaAlgebra.power_set(F.sample_space)).from_list([])
        assert F.is_measurable(event=A)

    def test_is_measurable_full_space(self, F):
        """Test is_measurable method with the full sample space."""
        A = Event(sig_alg=SigmaAlgebra.power_set(F.sample_space)).from_list(
            [0, 1, 2, 3]
        )
        assert F.is_measurable(event=A)

    def test_invalid_input_wrong_type_string(self, F):
        """Test that invalid input of wrong type string raises TypeError."""
        with pytest.raises(TypeError):
            F.is_measurable("not an event")

    def test_invalid_input_wrong_type_int(self, F):
        """Test that invalid input of wrong type int raises TypeError."""
        with pytest.raises(TypeError):
            F.is_measurable(123)

    def test_invalid_input_wrong_type_list(self, F):
        """Test that invalid input of wrong type list raises TypeError."""
        with pytest.raises(TypeError):
            F.is_measurable([0, 1])

    def test_event_with_different_sample_space_raises(self, F):
        """Test that an event with a different sample space raises ValueError."""
        different_Omega = SampleSpace().from_list(["a", "b", "c"])
        A = Event(sig_alg=SigmaAlgebra.power_set(different_Omega)).from_list(["a"])
        with pytest.raises(ValueError):
            F.is_measurable(event=A)


# --------------------- test sequence methods --------------------- #


class TestIteration:
    @pytest.fixture
    def F(self):
        Omega = SampleSpace().from_sequence(size=4)
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        return SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)

    def test_iteration_yields_tuples(self, F):
        """Test that iterating over the SigmaAlgebra yields tuples of (atom_id, Event)."""
        for atom_id, event in F:
            assert isinstance(atom_id, Hashable)
            assert isinstance(event, Event)

    def test_iteration_covers_all_atoms(self, F):
        """Test that iteration covers all atom IDs."""
        atom_ids_seen = set()
        for atom_id, _ in F:
            atom_ids_seen.add(atom_id)
        assert atom_ids_seen == {0, 1}

    def test_can_convert_to_dict(self, F):
        """Test that the SigmaAlgebra can be converted to a dictionary."""
        atoms_dict = dict(F)
        assert len(atoms_dict) == 2
        assert all(isinstance(event, Event) for event in atoms_dict.values())

    def test_iteration_matches_atom_id_to_event(self, F):
        """Test that iteration matches the atom_id_to_event property."""
        from_iter = dict(F)
        from_property = F.atom_id_to_event
        assert set(from_iter.keys()) == set(from_property.keys())
        for atom_id in from_iter:
            assert set(from_iter[atom_id].data) == set(from_property[atom_id].data)

    def test_iteration_with_string_atom_ids(self):
        """Test iteration works with string atom IDs."""
        Omega = SampleSpace().from_sequence(size=3)
        atom_ids = {0: "A", 1: "A", 2: "B"}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        atom_ids_seen = []
        for atom_id, _ in F:
            atom_ids_seen.append(atom_id)
        assert set(atom_ids_seen) == {"A", "B"}

    def test_iteration_order_is_consistent(self, F):
        """Test that iteration order is consistent across multiple iterations."""
        keys1 = [atom_id for atom_id, _ in F]
        keys2 = [atom_id for atom_id, _ in F]
        assert keys1 == keys2


class TestEquality:
    def test_non_equality_different_atom(self):
        """Test the __eq__ method for inequality with different atoms."""
        Omega = SampleSpace().from_sequence(size=3)
        F1 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        F2 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 1, 2: 1}
        )
        assert F1 != F2

    def test_non_equality_different_sample_spaces(self):
        """Test the __eq__ method for inequality with different sample spaces."""
        Omega1 = SampleSpace().from_sequence(size=2)
        Omega2 = SampleSpace().from_list(["a", "b"])
        F1 = SigmaAlgebra(sample_space=Omega1).from_dict(
            sample_id_to_atom_id={0: 0, 1: 0}
        )
        F2 = SigmaAlgebra(sample_space=Omega2).from_dict(
            sample_id_to_atom_id={"a": 0, "b": 0}
        )
        assert F1 != F2

    def test_non_equality_wrong_type_string(self):
        """Test the __eq__ method for inequality with wrong type string."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        other = "not a sigma algebra"
        assert F != other

    def test_non_equality_wrong_type_int(self):
        """Test the __eq__ method for inequality with wrong type int."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        other = 123
        assert F != other

    def test_non_equality_wrong_type_sample_space(self):
        """Test the __eq__ method for inequality with wrong type sample space."""
        Omega = SampleSpace().from_sequence(size=3)
        F = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        assert F != Omega

    def test_equality_same_components_different_names(self):
        """Test the __eq__ method for equality with same components but different names."""
        Omega = SampleSpace().from_sequence(size=3)
        F1 = SigmaAlgebra(sample_space=Omega, name="F1").from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        F2 = SigmaAlgebra(sample_space=Omega, name="F2").from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        assert F1 == F2

    def test_equality_identical_components(self):
        """Test the __eq__ method for equality with identical components."""
        Omega = SampleSpace().from_sequence(size=3)
        F1 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        F2 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id={0: 0, 1: 0, 2: 1}
        )
        assert F1 == F2


# --------------------- test lattice methods --------------------- #


class TestJoin:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F1(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="F1").from_dict(
            sample_id_to_atom_id={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
            }
        )

    @pytest.fixture
    def F2(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="F2").from_dict(
            sample_id_to_atom_id={
                0: 0,
                1: 1,
                2: 1,
                3: 1,
            }
        )

    @pytest.fixture
    def F3(self, Omega):
        return SigmaAlgebra(sample_space=Omega, name="F3").from_dict(
            sample_id_to_atom_id={
                0: 1,
                1: 0,
                2: 0,
                3: 1,
            }
        )

    def test_join_sigma_algebra_method(self, Omega, F1, F2):
        """Test the join method of SigmaAlgebra using the | operator."""
        expected_join = SigmaAlgebra(sample_space=Omega, name="join").from_dict(
            sample_id_to_atom_id={
                0: 0,
                1: 1,
                2: 2,
                3: 2,
            }
        )
        actual_join = F1 | F2

        assert actual_join == expected_join

    def test_join_function(self, Omega, F1, F2, F3):
        """Test the join function with multiple SigmaAlgebra instances."""
        expected_join = SigmaAlgebra(sample_space=Omega, name="join").from_dict(
            sample_id_to_atom_id={
                0: 0,
                1: 1,
                2: 2,
                3: 3,
            }
        )
        actual_join = Lattice.join([F1, F2, F3])

        assert actual_join == expected_join

    def test_join_function_with_one_algebra(self, F1):
        """Test the join function with a single SigmaAlgebra instance."""
        actual_join = Lattice.join([F1])

        assert actual_join == F1

    def test_join_with_string_instead_of_list_raises_error(self):
        """Test that join raises TypeError when given string input."""
        with pytest.raises(TypeError, match="Expected a list of sigma-algebras"):
            Lattice.join("not a list")

    def test_join_with_dict_instead_of_list_raises_error(self):
        """Test that join raises TypeError when given dict input."""
        with pytest.raises(TypeError, match="Expected a list of sigma-algebras"):
            Lattice.join({"key": "value"})

    def test_join_with_int_instead_of_list_raises_error(self):
        """Test that join raises TypeError when given int input."""
        with pytest.raises(TypeError, match="Expected a list of sigma-algebras"):
            Lattice.join(123)

    def test_join_with_empty_list_raises_error(self):
        """Test that join raises ValueError when given empty list."""
        with pytest.raises(ValueError, match="empty list"):
            Lattice.join([])

    def test_join_with_non_sigma_algebra_element_raises_error(self, F1):
        """Test that join raises TypeError when list contains non-SigmaAlgebra elements."""
        with pytest.raises(
            TypeError, match="All elements of the list must be a SigmaAlgebra"
        ):
            Lattice.join([F1, "not a sigma algebra"])

    def test_join_with_different_sample_spaces_raises_error(self, F1):
        """Test that join raises ValueError when sigma algebras have different sample spaces."""
        different_Omega = SampleSpace().from_list(["a", "b", "c"])
        F_different = SigmaAlgebra.trivial(different_Omega)
        with pytest.raises(ValueError, match="same sample space"):
            Lattice.join([F1, F_different])


class TestOrderRelations:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    def test_le_trivial_and_power_set(self, Omega):
        """Test that trivial SigmaAlgebra is less than or equal to power set SigmaAlgebra."""
        trivial = SigmaAlgebra.trivial(sample_space=Omega)
        power_set = SigmaAlgebra.power_set(sample_space=Omega)
        assert trivial <= power_set
        assert not power_set <= trivial

    def test_le_reflexive(self, Omega):
        """Test that a SigmaAlgebra is less than or equal to itself."""
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        assert F <= F

    def test_le_coarser_and_finer(self, Omega):
        """Test that a coarser SigmaAlgebra is less than or equal to a finer SigmaAlgebra."""
        coarse_atom_ids = {0: 0, 1: 0, 2: 0, 3: 1}
        F_coarse = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {0: 0, 1: 0, 2: 1, 3: 2}
        F_fine = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=fine_atom_ids
        )
        assert F_coarse <= F_fine
        assert not F_fine <= F_coarse

    def test_le_transitive(self):
        """Test that the less than or equal relation is transitive."""
        Omega = SampleSpace().from_sequence(size=4)
        F1 = SigmaAlgebra.trivial(sample_space=Omega)
        B_atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F2 = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=B_atom_ids)
        F3 = SigmaAlgebra.power_set(sample_space=Omega)
        assert F1 <= F2
        assert F2 <= F3
        assert F1 <= F3

    def test_lt_proper_sub_algebra(self, Omega):
        """Test that trivial SigmaAlgebra is less than power set SigmaAlgebra."""
        trivial = SigmaAlgebra.trivial(sample_space=Omega)
        power_set = SigmaAlgebra.power_set(sample_space=Omega)
        assert trivial < power_set
        assert not power_set < trivial

    def test_lt_not_proper_when_equal(self, Omega):
        """Test that a SigmaAlgebra is not less than itself."""
        F = SigmaAlgebra.trivial(sample_space=Omega)
        assert not F < F

    def test_ge_power_set_and_trivial(self, Omega):
        """Test that power set SigmaAlgebra is greater than or equal to trivial SigmaAlgebra."""
        power_set = SigmaAlgebra.power_set(sample_space=Omega)
        trivial = SigmaAlgebra.trivial(sample_space=Omega)
        assert power_set >= trivial
        assert not trivial >= power_set

    def test_ge_reflexive(self, Omega):
        """Test that a SigmaAlgebra is greater than or equal to itself."""
        F = SigmaAlgebra.trivial(sample_space=Omega)
        assert F >= F

    def test_ge_finer_and_coarser(self, Omega):
        """Test that a finer SigmaAlgebra is greater than or equal to a coarser SigmaAlgebra."""
        coarse_atom_ids = {0: 0, 1: 0, 2: 0, 3: 1}
        F_coarse = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {0: 0, 1: 0, 2: 1, 3: 2}
        F_fine = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=fine_atom_ids
        )
        assert F_fine >= F_coarse
        assert not F_coarse >= F_fine

    def test_gt_proper_super_algebra(self, Omega):
        """Test that power set SigmaAlgebra is greater than trivial SigmaAlgebra."""
        trivial = SigmaAlgebra.trivial(sample_space=Omega)
        power_set = SigmaAlgebra.power_set(sample_space=Omega)
        assert power_set > trivial
        assert not trivial > power_set

    def test_gt_not_proper_when_equal(self, Omega):
        """Test that a SigmaAlgebra is not greater than itself."""
        F = SigmaAlgebra.trivial(sample_space=Omega)
        assert not F > F

    def test_incomparable_sigma_algebras(self, Omega):
        """Test that two incomparable SigmaAlgebras are neither less than nor greater than each other."""
        sigma1_atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=sigma1_atom_ids
        )
        sigma2_atom_ids = {0: 0, 1: 1, 2: 0, 3: 1}
        F2 = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=sigma2_atom_ids
        )
        assert not F1 <= F2
        assert not F2 <= F1
        assert not F1 >= F2
        assert not F2 >= F1
        assert not F1 < F2
        assert not F2 < F1
        assert not F1 > F2
        assert not F2 > F1

    def test_three_level_chain(self):
        """Test a chain of three SigmaAlgebras with proper ordering."""
        Omega = SampleSpace().from_sequence(size=4)
        trivial = SigmaAlgebra.trivial(sample_space=Omega)
        middle_atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        middle = SigmaAlgebra(sample_space=Omega).from_dict(
            sample_id_to_atom_id=middle_atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space=Omega)
        assert trivial <= middle <= power_set
        assert trivial < middle < power_set
        assert power_set >= middle >= trivial
        assert power_set > middle > trivial

    def test_antisymmetry(self, Omega):
        """Test that if two SigmaAlgebras are less than or equal to each other, they are equal."""
        atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        F1 = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        F2 = SigmaAlgebra(sample_space=Omega).from_dict(sample_id_to_atom_id=atom_ids)
        assert F1 <= F2
        assert F2 <= F1
        assert F1 == F2
