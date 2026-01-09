from collections.abc import Hashable

import pandas as pd
import pytest

from sigalg.core import Event, SampleSpace, SigmaAlgebra, join


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )

    def test_constructor_integer_atom_ids_default_name(self, sample_space):
        """Test constructor with integer atom IDs and default name."""
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        name = "F"
        sigma_algebra = SigmaAlgebra(sample_space=sample_space, name=name).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_name = "F"

        assert sigma_algebra.sample_space == sample_space
        assert sigma_algebra.sample_id_to_atom_id == atom_ids
        assert sigma_algebra.name == expected_name

    def test_constructor_string_atom_ids_custom_name(self, sample_space):
        """Test constructor with string atom IDs and custom name."""
        atom_ids = {"omega_0": "A", "omega_1": "A", "omega_2": "B", "omega_3": "B"}
        name = "CustomSigma"
        sigma_algebra = SigmaAlgebra(sample_space=sample_space, name=name).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_name = "CustomSigma"

        assert sigma_algebra.sample_space == sample_space
        assert sigma_algebra.sample_id_to_atom_id == atom_ids
        assert sigma_algebra.name == expected_name

    def test_constructor_tuple_atom_ids(self, sample_space):
        """Test constructor with tuple atom IDs."""
        atom_ids = {
            "omega_0": (0, 0),
            "omega_1": (0, 1),
            "omega_2": (1, 0),
            "omega_3": (1, 1),
        }
        name = None
        sigma_algebra = SigmaAlgebra(sample_space=sample_space, name=name).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_name = None

        assert sigma_algebra.sample_space == sample_space
        assert sigma_algebra.sample_id_to_atom_id == atom_ids
        assert sigma_algebra.name == expected_name

    def test_constructor_mixed_hashable_atom_ids(self, sample_space):
        """Test constructor with mixed hashable atom IDs."""
        atom_ids = {"omega_0": 0, "omega_1": "special", "omega_2": 0, "omega_3": (1, 2)}
        name = "Mixed"
        sigma_algebra = SigmaAlgebra(sample_space=sample_space, name=name).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_name = "Mixed"

        assert sigma_algebra.sample_space == sample_space
        assert sigma_algebra.sample_id_to_atom_id == atom_ids
        assert sigma_algebra.name == expected_name

    def test_invalid_missing_sample_id_raises(self):
        """Test that missing sample ID raises exception."""
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_5": 1}
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        with pytest.raises((TypeError, ValueError)):
            SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id=atom_ids
            )

    def test_invalid_incomplete_mapping_raises(self):
        """Test that incomplete mapping raises exception."""
        atom_ids = {"omega_0": 0, "omega_1": 0}
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        with pytest.raises((TypeError, ValueError)):
            SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id=atom_ids
            )

    def test_invalid_unhashable_atom_id_raises(self):
        """Test that unhashable atom ID raises exception."""
        atom_ids = {"omega_0": [1, 2], "omega_1": 0, "omega_2": 1}
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        with pytest.raises((TypeError, ValueError)):
            SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id=atom_ids
            )


class TestNumAtoms:

    def test_num_atoms_two_atoms(self):
        """Test that num_atoms returns 2 for two-atom sigma algebra."""
        space = SampleSpace.generate_sequence(size=4, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_num_atoms = 2

        assert sigma_algebra.num_atoms == expected_num_atoms

    def test_num_atoms_trivial_one_atom(self):
        """Test that num_atoms returns 1 for trivial sigma algebra."""
        space = SampleSpace.generate_sequence(size=3, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 0}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_num_atoms = 1

        assert sigma_algebra.num_atoms == expected_num_atoms

    def test_num_atoms_power_set_three_atoms(self):
        """Test that num_atoms returns 3 for three-atom power set."""
        space = SampleSpace.generate_sequence(size=3, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": 0, "omega_1": 1, "omega_2": 2}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_num_atoms = 3

        assert sigma_algebra.num_atoms == expected_num_atoms

    def test_num_atoms_single_sample_point(self):
        """Test that num_atoms returns 1 for single sample point."""
        space = SampleSpace.generate_sequence(size=1, initial_index=0, prefix="omega")
        atom_ids = {"omega": 0}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        expected_num_atoms = 1

        assert sigma_algebra.num_atoms == expected_num_atoms


class TestAtomIds:

    def test_atom_ids_integer_atom_ids(self):
        """Test atom_ids returns correct set of integer IDs."""
        space = SampleSpace.generate_sequence(size=4, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        expected_atom_ids = {0, 1}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

        assert set(sigma_algebra.atom_ids) == expected_atom_ids

    def test_atom_ids_string_atom_ids(self):
        """Test atom_ids returns correct set of string IDs."""
        space = SampleSpace.generate_sequence(size=3, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": "A", "omega_1": "A", "omega_2": "B"}
        expected_atom_ids = {"A", "B"}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

        assert set(sigma_algebra.atom_ids) == expected_atom_ids

    def test_atom_ids_tuple_atom_ids(self):
        """Test atom_ids returns correct set of tuple IDs."""
        space = SampleSpace.generate_sequence(size=2, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": (0, 0), "omega_1": (1, 1)}
        expected_atom_ids = {(0, 0), (1, 1)}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

        assert set(sigma_algebra.atom_ids) == expected_atom_ids

    def test_atom_ids_single_atom(self):
        """Test atom_ids returns single atom ID."""
        space = SampleSpace.generate_sequence(size=1, initial_index=0, prefix="omega")
        atom_ids = {"omega": 0}
        expected_atom_ids = {0}
        sigma_algebra = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

        assert set(sigma_algebra.atom_ids) == expected_atom_ids


class TestAtomIdDictionaries:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega", name="Omega", data_name="sample"
        )

    @pytest.fixture
    def sigma_algebra(self, sample_space):
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        return SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

    def test_atom_id_to_sample_ids(self, sigma_algebra):
        """Test that atom_id_to_sample_ids returns correct mapping."""
        atom_id_to_sample_ids = sigma_algebra.atom_id_to_sample_ids
        expected = {
            0: ["omega_0", "omega_1"],
            1: ["omega_2", "omega_3"],
        }

        assert atom_id_to_sample_ids == expected

    def test_atom_id_to_event(self, sigma_algebra, sample_space):
        """Test that atom_id_to_event returns correct mapping."""
        atom_id_to_event = sigma_algebra.atom_id_to_event
        expected = {
            0: Event(sample_space=sample_space, name=0).from_list(
                ["omega_0", "omega_1"]
            ),
            1: Event(sample_space=sample_space, name=1).from_list(
                ["omega_2", "omega_3"]
            ),
        }

        assert atom_id_to_event == expected

    def test_atom_id_to_cardinality(self, sigma_algebra):
        """Test that atom_id_to_cardinality returns correct mapping."""
        atom_id_to_cardinality = sigma_algebra.atom_id_to_cardinality
        expected = {
            0: 2,
            1: 2,
        }

        assert atom_id_to_cardinality == expected


class TestToAtoms:

    def test_to_atoms_two_equal_atoms(self):
        """Test that to_atoms method returns correct list of Events for two equal atoms."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        atoms = sigma_algebra.to_atoms()
        expected = [
            Event(sample_space=sample_space).from_list(["omega_0", "omega_1"]),
            Event(sample_space=sample_space).from_list(["omega_2", "omega_3"]),
        ]
        assert atoms == expected

    def test_to_atoms_trivial_single_atom(self):
        """Test that to_atoms method returns correct list for trivial single atom."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 0}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        atoms = sigma_algebra.to_atoms()
        expected = [
            Event(sample_space=sample_space).from_list(
                ["omega_0", "omega_1", "omega_2"]
            ),
        ]
        assert atoms == expected

    def test_to_atoms_power_set_three_atoms(self):
        """Test that to_atoms method returns correct list for power set with three atoms."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        atom_ids = {"omega_0": 0, "omega_1": 1, "omega_2": 2}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        atoms = sigma_algebra.to_atoms()
        expected = [
            Event(sample_space=sample_space).from_list(["omega_0"]),
            Event(sample_space=sample_space).from_list(["omega_1"]),
            Event(sample_space=sample_space).from_list(["omega_2"]),
        ]
        assert atoms == expected

    def test_to_atoms_uneven_partition(self):
        """Test that to_atoms method returns correct list for uneven partition."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 0, "omega_3": 1}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        atoms = sigma_algebra.to_atoms()
        expected = [
            Event(sample_space=sample_space).from_list(
                ["omega_0", "omega_1", "omega_2"]
            ),
            Event(sample_space=sample_space).from_list(["omega_3"]),
        ]
        assert atoms == expected


class TestIsMeasurable:

    @pytest.fixture
    def sigma_algebra(self):
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        atom_ids = {"omega_0": 0, "omega_1": 1, "omega_2": 1, "omega_3": 2}
        return SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

    def test_is_measurable_measurable_event(self, sigma_algebra):
        """Test is_measurable method with a measurable event."""
        event = Event(sample_space=sigma_algebra.sample_space).from_list(
            ["omega_0", "omega_1", "omega_2"]
        )
        assert sigma_algebra.is_measurable(event)

    def test_is_measurable_nonmeasurable_event(self, sigma_algebra):
        """Test is_measurable method with a non-measurable event."""
        event = Event(sample_space=sigma_algebra.sample_space).from_list(
            ["omega_2", "omega_3"]
        )
        assert not sigma_algebra.is_measurable(event)

    def test_is_measurable_empty_event(self, sigma_algebra):
        """Test is_measurable method with an empty event."""
        event = Event(sample_space=sigma_algebra.sample_space).from_list([])
        assert sigma_algebra.is_measurable(event)

    def test_is_measurable_full_space(self, sigma_algebra):
        """Test is_measurable method with the full sample space."""
        event = Event(sample_space=sigma_algebra.sample_space).from_list(
            ["omega_0", "omega_1", "omega_2", "omega_3"]
        )
        assert sigma_algebra.is_measurable(event)

    def test_invalid_input_wrong_type_string(self, sigma_algebra):
        """Test that invalid input of wrong type string raises TypeError."""
        with pytest.raises(TypeError):
            sigma_algebra.is_measurable("not an event")

    def test_invalid_input_wrong_type_int(self, sigma_algebra):
        """Test that invalid input of wrong type int raises TypeError."""
        with pytest.raises(TypeError):
            sigma_algebra.is_measurable(123)

    def test_invalid_input_wrong_type_list(self, sigma_algebra):
        """Test that invalid input of wrong type list raises TypeError."""
        with pytest.raises(TypeError):
            sigma_algebra.is_measurable(["omega_0", "omega_1"])

    def test_event_with_different_sample_space_raises(self, sigma_algebra):
        """Test that an event with a different sample space raises ValueError."""
        different_space = SampleSpace().from_list(["a", "b", "c"])
        event = Event(sample_space=different_space).from_list(["a"])
        with pytest.raises(ValueError):
            sigma_algebra.is_measurable(event)


class TestGetAtomContaining:

    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace.generate_sequence(size=4, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        return SigmaAlgebra(sample_space=space).from_dict(sample_id_to_atom_id=atom_ids)

    def test_get_atom_containing_first_atom_point(self, sigma_algebra):
        """Test that get_atom_containing returns the correct atom for first atom point."""
        atom = sigma_algebra.get_atom_containing("omega_0")
        assert set(atom.indices) == {"omega_0", "omega_1"}

    def test_get_atom_containing_first_atom_point_second(self, sigma_algebra):
        """Test that get_atom_containing returns the correct atom for first atom second point."""
        atom = sigma_algebra.get_atom_containing("omega_1")
        assert set(atom.indices) == {"omega_0", "omega_1"}

    def test_get_atom_containing_second_atom_point(self, sigma_algebra):
        """Test that get_atom_containing returns the correct atom for second atom point."""
        atom = sigma_algebra.get_atom_containing("omega_2")
        assert set(atom.indices) == {"omega_2", "omega_3"}

    def test_get_atom_containing_second_atom_point_second(self, sigma_algebra):
        """Test that get_atom_containing returns the correct atom for second atom second point."""
        atom = sigma_algebra.get_atom_containing("omega_3")
        assert set(atom.indices) == {"omega_2", "omega_3"}

    def test_invalid_sample_id_not_in_sample_space(self, sigma_algebra):
        """Test that invalid sample ID not in sample space raises ValueError."""
        with pytest.raises(ValueError):
            sigma_algebra.get_atom_containing("omega_5")

    def test_invalid_sample_id_non_existent(self, sigma_algebra):
        """Test that non-existent sample ID raises ValueError."""
        with pytest.raises(ValueError):
            sigma_algebra.get_atom_containing("invalid")


class TestFromPandas:

    def test_from_pandas_custom_names(self):
        """Test that from_pandas creates a SigmaAlgebra correctly with custom names."""
        series_data = {"omega_0": 0, "omega_1": 1, "omega_2": 1}
        data = pd.Series(data=series_data, name="atoms")
        sigma_algebra = SigmaAlgebra(name="G").from_pandas(data=data)
        data.name = "atom ID"

        assert isinstance(sigma_algebra, SigmaAlgebra)
        assert sigma_algebra.name == "G"
        assert sigma_algebra.sample_space.name == "Omega"
        assert sigma_algebra.sample_id_to_atom_id == series_data
        assert sigma_algebra.data.name == "atom ID"
        pd.testing.assert_series_equal(sigma_algebra.data, data)

    def test_from_pandas_default_names(self):
        """Test that from_pandas creates a SigmaAlgebra correctly with default names."""
        series_data = {"omega_0": 0, "omega_1": 1, "omega_2": 1}
        data = pd.Series(data=series_data, name=None)
        sigma_algebra = SigmaAlgebra().from_pandas(data=data)
        data.name = "atom ID"

        assert isinstance(sigma_algebra, SigmaAlgebra)
        assert sigma_algebra.name == "F"
        assert sigma_algebra.sample_space.name == "Omega"
        assert sigma_algebra.sample_id_to_atom_id == series_data
        assert sigma_algebra.data.name == "atom ID"
        pd.testing.assert_series_equal(sigma_algebra.data, data)

    def test_from_pandas_string_atom_ids(self):
        """Test that from_pandas works with string atom IDs."""
        series_data = {"s_0": "A", "s_1": "A", "s_2": "B"}
        data = pd.Series(data=series_data, name="partitions")
        sigma_algebra = SigmaAlgebra(name="CustomF").from_pandas(data=data)
        data.name = "atom ID"

        assert isinstance(sigma_algebra, SigmaAlgebra)
        assert sigma_algebra.name == "CustomF"
        assert sigma_algebra.sample_space.name == "Omega"
        assert sigma_algebra.sample_id_to_atom_id == series_data
        assert sigma_algebra.data.name == "atom ID"
        pd.testing.assert_series_equal(sigma_algebra.data, data)

    def test_invalid_input_list_instead_of_series(self):
        """Test that invalid input list raises TypeError."""
        with pytest.raises(TypeError):
            SigmaAlgebra.from_pandas(data=["not", "a", "series"])

    def test_invalid_input_dict_instead_of_series(self):
        """Test that invalid input dict raises TypeError."""
        with pytest.raises(TypeError):
            SigmaAlgebra.from_pandas(data={"key": "value"})

    def test_invalid_input_string_instead_of_series(self):
        """Test that invalid input string raises TypeError."""
        with pytest.raises(TypeError):
            SigmaAlgebra.from_pandas(data="string")


class TestJoin:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace.generate_sequence(size=4)

    @pytest.fixture
    def F1(self, sample_space):
        return SigmaAlgebra(sample_space=sample_space, name="F1").from_dict(
            sample_id_to_atom_id={
                "omega_0": 0,
                "omega_1": 0,
                "omega_2": 1,
                "omega_3": 1,
            }
        )

    @pytest.fixture
    def F2(self, sample_space):
        return SigmaAlgebra(sample_space=sample_space, name="F2").from_dict(
            sample_id_to_atom_id={
                "omega_0": 0,
                "omega_1": 1,
                "omega_2": 1,
                "omega_3": 1,
            }
        )

    @pytest.fixture
    def F3(self, sample_space):
        return SigmaAlgebra(sample_space=sample_space, name="F3").from_dict(
            sample_id_to_atom_id={
                "omega_0": 1,
                "omega_1": 0,
                "omega_2": 0,
                "omega_3": 1,
            }
        )

    def test_join_sigma_algebra_method(self, sample_space, F1, F2):
        """Test the join method of SigmaAlgebra using the | operator."""
        expected_join = SigmaAlgebra(sample_space=sample_space, name="join").from_dict(
            sample_id_to_atom_id={
                "omega_0": 0,
                "omega_1": 1,
                "omega_2": 2,
                "omega_3": 2,
            }
        )
        actual_join = F1 | F2

        assert actual_join == expected_join

    def test_join_function(self, sample_space, F1, F2, F3):
        """Test the join function with multiple SigmaAlgebra instances."""
        expected_join = SigmaAlgebra(sample_space=sample_space, name="join").from_dict(
            sample_id_to_atom_id={
                "omega_0": 0,
                "omega_1": 1,
                "omega_2": 2,
                "omega_3": 3,
            }
        )
        actual_join = join([F1, F2, F3])

        assert actual_join == expected_join

    def test_join_function_with_one_algebra(self, F1):
        """Test the join function with a single SigmaAlgebra instance."""
        actual_join = join([F1])

        assert actual_join == F1

    def test_join_with_string_instead_of_list_raises_error(self):
        """Test that join raises TypeError when given string input."""
        with pytest.raises(
            TypeError, match="Expected a list of SigmaAlgebra instances"
        ):
            join("not a list")

    def test_join_with_dict_instead_of_list_raises_error(self):
        """Test that join raises TypeError when given dict input."""
        with pytest.raises(
            TypeError, match="Expected a list of SigmaAlgebra instances"
        ):
            join({"key": "value"})

    def test_join_with_int_instead_of_list_raises_error(self):
        """Test that join raises TypeError when given int input."""
        with pytest.raises(
            TypeError, match="Expected a list of SigmaAlgebra instances"
        ):
            join(123)

    def test_join_with_empty_list_raises_error(self):
        """Test that join raises ValueError when given empty list."""
        with pytest.raises(ValueError, match="empty list"):
            join([])

    def test_join_with_non_sigma_algebra_element_raises_error(self, F1):
        """Test that join raises TypeError when list contains non-SigmaAlgebra elements."""
        with pytest.raises(
            TypeError, match="All elements of the list must be SigmaAlgebra instances"
        ):
            join([F1, "not a sigma algebra"])

    def test_join_with_different_sample_spaces_raises_error(self, F1):
        """Test that join raises ValueError when sigma algebras have different sample spaces."""
        different_space = SampleSpace().from_list(["a", "b", "c"])
        F_different = SigmaAlgebra.trivial(different_space)
        with pytest.raises(ValueError, match="same sample space"):
            join([F1, F_different])


class TestPowerSet:

    def test_power_set_three_samples_default_name(self):
        """Test that power_set method creates the correct SigmaAlgebra for three samples."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        sigma_algebra = SigmaAlgebra.power_set(sample_space)

        assert sigma_algebra.name == "power_set"
        assert sigma_algebra.num_atoms == 3
        assert sigma_algebra.sample_space == sample_space
        for idx, sample_id in enumerate(sample_space.data):
            assert sigma_algebra.sample_id_to_atom_id[sample_id] == idx

    def test_power_set_four_samples_custom_name(self):
        """Test that power_set method creates the correct SigmaAlgebra for four samples with custom name."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="s"
        )
        sigma_algebra = SigmaAlgebra.power_set(sample_space, name="PowerSet")

        assert sigma_algebra.name == "PowerSet"
        assert sigma_algebra.num_atoms == 4
        assert sigma_algebra.sample_space == sample_space
        for idx, sample_id in enumerate(sample_space.data):
            assert sigma_algebra.sample_id_to_atom_id[sample_id] == idx

    def test_power_set_single_sample_point(self):
        """Test that power_set method creates the correct SigmaAlgebra for single sample point."""
        sample_space = SampleSpace.generate_sequence(
            size=1, initial_index=0, prefix="omega"
        )
        sigma_algebra = SigmaAlgebra.power_set(sample_space, name="SinglePoint")

        assert sigma_algebra.name == "SinglePoint"
        assert sigma_algebra.num_atoms == 1
        assert sigma_algebra.sample_space == sample_space
        for idx, sample_id in enumerate(sample_space.data):
            assert sigma_algebra.sample_id_to_atom_id[sample_id] == idx


class TestTrivial:

    def test_trivial_creation_three_samples_default_name(self):
        """Test that trivial method creates the correct SigmaAlgebra for three samples."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        sigma_algebra = SigmaAlgebra.trivial(sample_space)

        assert sigma_algebra.name == "trivial"
        assert sigma_algebra.num_atoms == 1
        assert sigma_algebra.sample_space == sample_space
        unique_atom_ids = set(sigma_algebra.sample_id_to_atom_id.values())
        assert len(unique_atom_ids) == 1

    def test_trivial_creation_four_samples_custom_name(self):
        """Test that trivial method creates the correct SigmaAlgebra for four samples with custom name."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="s"
        )
        sigma_algebra = SigmaAlgebra.trivial(sample_space, name="TrivialSigma")

        assert sigma_algebra.name == "TrivialSigma"
        assert sigma_algebra.num_atoms == 1
        assert sigma_algebra.sample_space == sample_space
        unique_atom_ids = set(sigma_algebra.sample_id_to_atom_id.values())
        assert len(unique_atom_ids) == 1

    def test_trivial_creation_single_sample_point(self):
        """Test that trivial method creates the correct SigmaAlgebra for single sample point."""
        sample_space = SampleSpace.generate_sequence(
            size=1, initial_index=0, prefix="omega"
        )
        sigma_algebra = SigmaAlgebra.trivial(sample_space, name="Trivial")

        assert sigma_algebra.name == "Trivial"
        assert sigma_algebra.num_atoms == 1
        assert sigma_algebra.sample_space == sample_space
        unique_atom_ids = set(sigma_algebra.sample_id_to_atom_id.values())
        assert len(unique_atom_ids) == 1


class TestIteration:

    @pytest.fixture
    def sigma_algebra(self):
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        atom_ids = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        return SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

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
        space = SampleSpace.generate_sequence(size=3, initial_index=0, prefix="omega")
        atom_ids = {"omega_0": "A", "omega_1": "A", "omega_2": "B"}
        sigma = SigmaAlgebra(sample_space=space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
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

    def test_non_equality_different_atom_ids(self):
        """Test the __eq__ method for inequality with different atom IDs."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        other = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 1, "omega_2": 1}
        )
        assert given != other

    def test_non_equality_different_sample_spaces(self):
        """Test the __eq__ method for inequality with different sample spaces."""
        sample_space1 = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        sample_space2 = SampleSpace().from_list(["a", "b"])
        given = SigmaAlgebra(sample_space=sample_space1).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0}
        )
        other = SigmaAlgebra(sample_space=sample_space2).from_dict(
            sample_id_to_atom_id={"a": 0, "b": 0}
        )
        assert given != other

    def test_non_equality_wrong_type_string(self):
        """Test the __eq__ method for inequality with wrong type string."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        other = "not a sigma algebra"
        assert given != other

    def test_non_equality_wrong_type_int(self):
        """Test the __eq__ method for inequality with wrong type int."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        other = 123
        assert given != other

    def test_non_equality_wrong_type_sample_space(self):
        """Test the __eq__ method for inequality with wrong type sample space."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        other = sample_space
        assert given != other

    def test_equality_same_components_different_names(self):
        """Test the __eq__ method for equality with same components but different names."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given = SigmaAlgebra(sample_space=sample_space, name="F1").from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        other = SigmaAlgebra(sample_space=sample_space, name="F2").from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        assert given == other

    def test_equality_identical_components(self):
        """Test the __eq__ method for equality with identical components."""
        sample_space = SampleSpace.generate_sequence(
            size=3, initial_index=0, prefix="omega"
        )
        given = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        other = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1}
        )
        assert given == other


class TestOrderRelations:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace.generate_sequence(size=4, initial_index=0, prefix="s")

    def test_le_trivial_and_power_set(self, sample_space):
        """Test that trivial SigmaAlgebra is less than or equal to power set SigmaAlgebra."""
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial <= power_set
        assert not power_set <= trivial

    def test_le_reflexive(self, sample_space):
        """Test that a SigmaAlgebra is less than or equal to itself."""
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        assert sigma_algebra <= sigma_algebra

    def test_le_coarser_and_finer(self, sample_space):
        """Test that a coarser SigmaAlgebra is less than or equal to a finer SigmaAlgebra."""
        coarse_atom_ids = {"s_0": 0, "s_1": 0, "s_2": 0, "s_3": 1}
        coarse = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 2}
        fine = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=fine_atom_ids
        )
        assert coarse <= fine
        assert not fine <= coarse

    def test_le_transitive(self):
        """Test that the less than or equal relation is transitive."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="s"
        )
        A = SigmaAlgebra.trivial(sample_space=sample_space)
        B_atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        B = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=B_atom_ids
        )
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
        coarse_atom_ids = {"s_0": 0, "s_1": 0, "s_2": 0, "s_3": 1}
        coarse = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=coarse_atom_ids
        )
        fine_atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 2}
        fine = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=fine_atom_ids
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
        sigma1_atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        sigma1 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=sigma1_atom_ids
        )
        sigma2_atom_ids = {"s_0": 0, "s_1": 1, "s_2": 0, "s_3": 1}
        sigma2 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=sigma2_atom_ids
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
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="s"
        )
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        middle_atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        middle = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=middle_atom_ids
        )
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial <= middle <= power_set
        assert trivial < middle < power_set
        assert power_set >= middle >= trivial
        assert power_set > middle > trivial

    def test_antisymmetry(self, sample_space):
        """Test that if two SigmaAlgebras are less than or equal to each other, they are equal."""
        atom_ids = {"s_0": 0, "s_1": 0, "s_2": 1, "s_3": 1}
        sigma_algebra1 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        sigma_algebra2 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        assert sigma_algebra1 <= sigma_algebra2
        assert sigma_algebra2 <= sigma_algebra1
        assert sigma_algebra1 == sigma_algebra2
