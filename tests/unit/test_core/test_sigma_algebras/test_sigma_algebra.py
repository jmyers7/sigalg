from collections.abc import Hashable

import pandas as pd
import pytest

from sigalg.core import Event, SampleSpace, SigmaAlgebra


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.mark.parametrize(
        "atom_ids,name",
        [
            pytest.param(
                {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1},
                "F",
                id="integer_atom_ids_default_name",
            ),
            pytest.param(
                {"omega0": "A", "omega1": "A", "omega2": "B", "omega3": "B"},
                "CustomSigma",
                id="string_atom_ids_custom_name",
            ),
            pytest.param(
                {
                    "omega0": (0, 0),
                    "omega1": (0, 1),
                    "omega2": (1, 0),
                    "omega3": (1, 1),
                },
                None,
                id="tuple_atom_ids",
            ),
            pytest.param(
                {"omega0": 0, "omega1": "special", "omega2": 0, "omega3": (1, 2)},
                "Mixed",
                id="mixed_hashable_atom_ids",
            ),
        ],
    )
    def test_constructor(self, sample_space, atom_ids, name):
        """Test constructor with various atom ID types and names."""
        if name is not None:
            sigma_algebra = SigmaAlgebra(
                sample_id_to_atom_id=atom_ids, sample_space=sample_space, name=name
            )
            expected_name = name
        else:
            sigma_algebra = SigmaAlgebra(
                sample_id_to_atom_id=atom_ids, sample_space=sample_space
            )
            expected_name = "F"

        assert sigma_algebra.sample_space == sample_space
        assert sigma_algebra.sample_id_to_atom_id == atom_ids
        assert sigma_algebra.name == expected_name

    @pytest.mark.parametrize(
        "atom_ids,sample_space_indices",
        [
            pytest.param(
                {"omega0": 0, "omega1": 0, "omega5": 1},
                ["omega0", "omega1", "omega2"],
                id="missing_sample_id",
            ),
            pytest.param(
                {"omega0": 0, "omega1": 0},
                ["omega0", "omega1", "omega2"],
                id="incomplete_mapping",
            ),
            pytest.param(
                {"omega0": [1, 2], "omega1": 0, "omega2": 1},
                ["omega0", "omega1", "omega2"],
                id="unhashable_atom_id",
            ),
        ],
    )
    def test_invalid_inputs_raise(self, atom_ids, sample_space_indices):
        """Test that invalid inputs raise appropriate exceptions."""
        sample_space = SampleSpace(sample_space_indices)
        with pytest.raises((TypeError, ValueError)):
            SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)


class TestNumAtoms:

    @pytest.mark.parametrize(
        "indices,atom_ids,expected_num_atoms",
        [
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1},
                2,
                id="two_atoms",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 0, "omega1": 0, "omega2": 0},
                1,
                id="trivial_one_atom",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 0, "omega1": 1, "omega2": 2},
                3,
                id="power_set_three_atoms",
            ),
            pytest.param(
                ["omega0"],
                {"omega0": 0},
                1,
                id="single_sample_point",
            ),
        ],
    )
    def test_num_atoms(self, indices, atom_ids, expected_num_atoms):
        """Test that num_atoms property returns correct number of atoms."""
        space = SampleSpace(indices)
        sigma_algebra = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

        assert sigma_algebra.num_atoms == expected_num_atoms


class TestAtomIds:

    @pytest.mark.parametrize(
        "indices,atom_ids,expected_atom_ids",
        [
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1},
                {0, 1},
                id="integer_atom_ids",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": "A", "omega1": "A", "omega2": "B"},
                {"A", "B"},
                id="string_atom_ids",
            ),
            pytest.param(
                ["omega0", "omega1"],
                {"omega0": (0, 0), "omega1": (1, 1)},
                {(0, 0), (1, 1)},
                id="tuple_atom_ids",
            ),
            pytest.param(
                ["omega0"],
                {"omega0": 0},
                {0},
                id="single_atom",
            ),
        ],
    )
    def test_atom_ids_property(self, indices, atom_ids, expected_atom_ids):
        """Test that atom_ids property returns correct list of unique atom IDs."""
        space = SampleSpace(indices)
        sigma_algebra = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

        assert set(sigma_algebra.atom_ids) == expected_atom_ids


class TestAtomIdDictionaries:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    def test_atom_id_to_sample_ids(self, sigma_algebra):
        """Test that atom_id_to_sample_ids returns correct mapping."""
        atom_id_to_sample_ids = sigma_algebra.atom_id_to_sample_ids
        expected = {
            0: ["omega0", "omega1"],
            1: ["omega2", "omega3"],
        }

        assert atom_id_to_sample_ids == expected

    def test_atom_id_to_event(self, sigma_algebra, sample_space):
        """Test that atom_id_to_event returns correct mapping."""
        atom_id_to_event = sigma_algebra.atom_id_to_event
        expected = {
            0: Event(sample_space=sample_space, indices=["omega0", "omega1"], name=0),
            1: Event(sample_space=sample_space, indices=["omega2", "omega3"], name=1),
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

    @pytest.mark.parametrize(
        "indices,atom_ids,expected_atoms",
        [
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1},
                [["omega0", "omega1"], ["omega2", "omega3"]],
                id="two_equal_atoms",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 0, "omega1": 0, "omega2": 0},
                [["omega0", "omega1", "omega2"]],
                id="trivial_single_atom",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 0, "omega1": 1, "omega2": 2},
                [["omega0"], ["omega1"], ["omega2"]],
                id="power_set_three_atoms",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                {"omega0": 0, "omega1": 0, "omega2": 0, "omega3": 1},
                [["omega0", "omega1", "omega2"], ["omega3"]],
                id="uneven_partition",
            ),
        ],
    )
    def test_to_atoms(self, indices, atom_ids, expected_atoms):
        """Test that to_atoms method returns correct list of Events."""
        sample_space = SampleSpace(indices)
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids, sample_space=sample_space
        )
        atoms = sigma_algebra.to_atoms()
        expected = [
            Event(sample_space=sample_space, indices=atom) for atom in expected_atoms
        ]

        assert atoms == expected


class TestIsMeasurable:

    @pytest.fixture
    def sigma_algebra(self):
        sample_space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 1, "omega2": 1, "omega3": 2}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    @pytest.mark.parametrize(
        "indices,expected",
        [
            pytest.param(["omega0", "omega1", "omega2"], True, id="measurable_event"),
            pytest.param(["omega2", "omega3"], False, id="nonmeasurable_event"),
            pytest.param([], True, id="empty_event"),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"], True, id="full_space"
            ),
        ],
    )
    def test_is_measureable(self, indices, expected, sigma_algebra):
        """Test is_measurable method with various events."""
        event = Event(sample_space=sigma_algebra.sample_space, indices=indices)

        assert sigma_algebra.is_measurable(event) == expected

    @pytest.mark.parametrize(
        "invalid_input",
        [
            pytest.param("not an event", id="wrong_type_string"),
            pytest.param(123, id="wrong_type_int"),
            pytest.param(["omega0", "omega1"], id="wrong_type_list"),
        ],
    )
    def test_invalid_input_raises(self, sigma_algebra, invalid_input):
        """Test that invalid inputs raise TypeError."""
        with pytest.raises(TypeError):
            sigma_algebra.is_measurable(invalid_input)

    def test_event_with_different_sample_space_raises(self, sigma_algebra):
        """Test that an event with a different sample space raises ValueError."""
        different_space = SampleSpace(["a", "b", "c"])
        event = Event(sample_space=different_space, indices=["a"])
        with pytest.raises(ValueError):
            sigma_algebra.is_measurable(event)


class TestGetAtomContaining:

    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    @pytest.mark.parametrize(
        "sample_id, expected_atom_indices",
        [
            pytest.param("omega0", ["omega0", "omega1"], id="first_atom_point"),
            pytest.param("omega1", ["omega0", "omega1"], id="first_atom_point_second"),
            pytest.param("omega2", ["omega2", "omega3"], id="second_atom_point"),
            pytest.param("omega3", ["omega2", "omega3"], id="second_atom_point_second"),
        ],
    )
    def test_get_atom_containing(self, sigma_algebra, sample_id, expected_atom_indices):
        """Test that get_atom_containing returns the correct atom."""
        atom = sigma_algebra.get_atom_containing(sample_id)

        assert set(atom.indices) == set(expected_atom_indices)

    @pytest.mark.parametrize(
        "invalid_sample_id",
        [
            pytest.param("omega5", id="not_in_sample_space"),
            pytest.param("invalid", id="non_existent_id"),
        ],
    )
    def test_invalid_sample_id_raises(self, sigma_algebra, invalid_sample_id):
        """Test that invalid sample IDs raise ValueError."""
        with pytest.raises(ValueError):
            sigma_algebra.get_atom_containing(invalid_sample_id)


class TestFromPandas:

    @pytest.mark.parametrize(
        "series_data,series_name,sigma_name,expected_data_name",
        [
            pytest.param(
                {"omega0": 0, "omega1": 1, "omega2": 1},
                "atoms",
                "G",
                "atoms",
                id="custom_names",
            ),
            pytest.param(
                {"omega0": 0, "omega1": 1, "omega2": 1},
                None,
                None,
                None,
                id="default_names",
            ),
            pytest.param(
                {"s0": "A", "s1": "A", "s2": "B"},
                "partitions",
                "CustomF",
                "partitions",
                id="string_atom_ids",
            ),
        ],
    )
    def test_from_pandas(
        self, series_data, series_name, sigma_name, expected_data_name
    ):
        """Test that from_pandas creates a SigmaAlgebra correctly from a pd.Series."""
        data = pd.Series(data=series_data, name=series_name)

        if sigma_name is not None:
            sigma_algebra = SigmaAlgebra.from_pandas(data=data, name=sigma_name)
            expected_name = sigma_name
        else:
            sigma_algebra = SigmaAlgebra.from_pandas(data=data)
            expected_name = "F"

        assert isinstance(sigma_algebra, SigmaAlgebra)
        assert sigma_algebra.name == expected_name
        assert sigma_algebra.sample_space.name == "Omega"
        assert sigma_algebra.sample_id_to_atom_id == series_data
        assert sigma_algebra.data.name == expected_data_name
        pd.testing.assert_series_equal(sigma_algebra.data, data)

    @pytest.mark.parametrize(
        "invalid_data",
        [
            pytest.param(["not", "a", "series"], id="list_instead_of_series"),
            pytest.param({"key": "value"}, id="dict_instead_of_series"),
            pytest.param("string", id="string_instead_of_series"),
        ],
    )
    def test_invalid_input_raises(self, invalid_data):
        """Test that invalid inputs raise TypeError."""
        with pytest.raises(TypeError):
            SigmaAlgebra.from_pandas(data=invalid_data)


class TestPowerSet:

    @pytest.mark.parametrize(
        "indices,name,expected_num_atoms",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                None,
                3,
                id="three_samples_default_name",
            ),
            pytest.param(
                ["s0", "s1", "s2", "s3"],
                "PowerSet",
                4,
                id="four_samples_custom_name",
            ),
            pytest.param(
                ["omega0"],
                "SinglePoint",
                1,
                id="single_sample_point",
            ),
        ],
    )
    def test_power_set(self, indices, name, expected_num_atoms):
        """Test that power_set method creates the correct SigmaAlgebra."""
        sample_space = SampleSpace(indices)

        if name is not None:
            sigma_algebra = SigmaAlgebra.power_set(sample_space, name=name)
            expected_name = name
        else:
            sigma_algebra = SigmaAlgebra.power_set(sample_space)
            expected_name = "power_set"

        assert sigma_algebra.name == expected_name
        assert sigma_algebra.num_atoms == expected_num_atoms
        assert sigma_algebra.sample_space == sample_space
        for idx, sample_id in enumerate(sample_space.data):
            assert sigma_algebra.sample_id_to_atom_id[sample_id] == idx


class TestTrivial:

    @pytest.mark.parametrize(
        "indices,name",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                None,
                id="three_samples_default_name",
            ),
            pytest.param(
                ["s0", "s1", "s2", "s3"],
                "TrivialSigma",
                id="four_samples_custom_name",
            ),
            pytest.param(
                ["omega0"],
                "Trivial",
                id="single_sample_point",
            ),
        ],
    )
    def test_trivial_creation(self, indices, name):
        """Test that trivial method creates the correct SigmaAlgebra."""
        sample_space = SampleSpace(indices)

        if name is not None:
            sigma_algebra = SigmaAlgebra.trivial(sample_space, name=name)
            expected_name = name
        else:
            sigma_algebra = SigmaAlgebra.trivial(sample_space)
            expected_name = "trivial"

        assert sigma_algebra.name == expected_name
        assert sigma_algebra.num_atoms == 1
        assert sigma_algebra.sample_space == sample_space
        unique_atom_ids = set(sigma_algebra.sample_id_to_atom_id.values())
        assert len(unique_atom_ids) == 1


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

    @pytest.mark.parametrize(
        "given,other",
        [
            pytest.param(
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                ),
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 1, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                ),
                id="different_atom_ids",
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0},
                    sample_space=SampleSpace(["omega0", "omega1"]),
                ),
                SigmaAlgebra(
                    sample_id_to_atom_id={"a": 0, "b": 0},
                    sample_space=SampleSpace(["a", "b"]),
                ),
                id="different_sample_spaces",
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                ),
                "not a sigma algebra",
                id="wrong_type_string",
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                ),
                123,
                id="wrong_type_int",
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                ),
                SampleSpace(["omega0", "omega1", "omega2"]),
                id="wrong_type_sample_space",
            ),
        ],
    )
    def test_non_equality(self, given, other):
        """Test the __eq__ method for inequality."""
        assert given != other

    @pytest.mark.parametrize(
        "given,other",
        [
            pytest.param(
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                    name="F1",
                ),
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                    name="F2",
                ),
                id="same_components_different_names",
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                ),
                SigmaAlgebra(
                    sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    sample_space=SampleSpace(["omega0", "omega1", "omega2"]),
                ),
                id="identical_components",
            ),
        ],
    )
    def test_equality(self, given, other):
        """Test the __eq__ method for equality."""
        assert given == other


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
