import pandas as pd
import pytest

from sigalg.core import Event, ProbabilitySpace, SampleSpace, SigmaAlgebra


class TestConstructor:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_construction_with_integer_atom_ids(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        assert sigma.sample_space == sample_space
        assert sigma.sample_id_to_atom_id == atom_ids

    def test_construction_with_string_atom_ids(self, sample_space):
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B", "omega3": "B"}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        assert sigma.sample_id_to_atom_id == atom_ids

    def test_construction_with_generated_sample_space(self):
        sample_id_to_atom_id = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_algebra = SigmaAlgebra(sample_id_to_atom_id=sample_id_to_atom_id)
        assert sigma_algebra.sample_id_to_atom_id == sample_id_to_atom_id
        assert isinstance(sigma_algebra.sample_space, SampleSpace)

    def test_construction_with_tuple_atom_ids(self, sample_space):
        atom_ids = {
            "omega0": (0, 0),
            "omega1": (0, 1),
            "omega2": (1, 0),
            "omega3": (1, 1),
        }
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        assert sigma.sample_id_to_atom_id == atom_ids

    def test_construction_with_mixed_hashable_atom_ids(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": "special", "omega2": 0, "omega3": (1, 2)}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        assert sigma.sample_id_to_atom_id == atom_ids

    def test_construction_creates_atom_mapping(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        atom_to_sample = sigma.atom_id_to_sample_ids
        assert len(atom_to_sample) == 2
        assert set(atom_to_sample[0]) == {"omega0", "omega1"}
        assert set(atom_to_sample[1]) == {"omega2", "omega3"}

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            SigmaAlgebra(sample_id_to_atom_id={"omega0": 0}, sample_space="not a space")

    def test_construction_with_non_dict_atom_ids(self, sample_space):
        with pytest.raises(TypeError, match="must be a dictionary"):
            SigmaAlgebra(sample_id_to_atom_id=[0, 0, 1, 1], sample_space=sample_space)

    def test_construction_with_missing_sample_indices(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0}
        with pytest.raises(ValueError, match="must contain an entry for every"):
            SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    def test_construction_with_extra_sample_indices(self, sample_space):
        atom_ids = {
            "omega0": 0,
            "omega1": 0,
            "omega2": 1,
            "omega3": 1,
            "extra": 2,
        }
        with pytest.raises(ValueError, match="must contain an entry for every"):
            SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    def test_construction_preserves_atom_id_types(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        for atom_id in sigma.sample_id_to_atom_id.values():
            assert isinstance(atom_id, int)


class TestProperties:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)

    def test_sample_space_property(self, sigma_algebra, sample_space):
        assert sigma_algebra.sample_space == sample_space

    def test_sample_space_has_correct_indices(self, sigma_algebra, sample_space):
        assert sigma_algebra.sample_space.values.equals(sample_space.values)

    def test_atom_ids_property_has_correct_values(self, sigma_algebra):
        sample_id_to_atom_id = sigma_algebra.sample_id_to_atom_id
        assert sample_id_to_atom_id == {
            "omega0": 0,
            "omega1": 0,
            "omega2": 1,
            "omega3": 1,
        }

    def test_num_atoms_property(self, sigma_algebra):
        assert sigma_algebra.num_atoms == 2

    def test_num_atoms_with_single_atom(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert sigma.num_atoms == 1

    def test_num_atoms_with_all_distinct(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 1, "omega2": 2}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert sigma.num_atoms == 3


class TestAtomIdToSampleIdxList:
    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_returns_dict(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_sample_ids
        assert isinstance(result, dict)

    def test_has_correct_number_of_atoms(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_sample_ids
        assert len(result) == 2

    def test_keys_are_atom_ids(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_sample_ids
        assert set(result.keys()) == {0, 1}

    def test_values_are_lists(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_sample_ids
        for sample_list in result.values():
            assert isinstance(sample_list, list)

    def test_atoms_have_correct_samples(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_sample_ids
        assert set(result[0]) == {"omega0", "omega1"}
        assert set(result[1]) == {"omega2", "omega3"}

    def test_with_string_atom_ids(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        result = sigma.atom_id_to_sample_ids
        assert set(result.keys()) == {"A", "B"}
        assert set(result["A"]) == {"omega0", "omega1"}
        assert set(result["B"]) == {"omega2"}

    def test_with_tuple_atom_ids(self):
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
        result = sigma_algebra.atom_id_to_event
        assert isinstance(result, dict)

    def test_has_correct_number_of_atoms(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_event
        assert len(result) == 2

    def test_keys_are_atom_ids(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_event
        assert set(result.keys()) == {0, 1}

    def test_values_are_events(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_event
        for event in result.values():
            assert isinstance(event, Event)

    def test_atoms_have_correct_indices(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_event
        atom_0 = result[0]
        atom_1 = result[1]
        assert set(atom_0.values) == {"omega0", "omega1"}
        assert set(atom_1.values) == {"omega2", "omega3"}

    def test_event_names_are_atom_ids(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_event
        assert result[0].name == "0"
        assert result[1].name == "1"


class TestAtomIdToCardinality:
    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_returns_dict(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_cardinality
        assert isinstance(result, dict)

    def test_has_correct_keys(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_cardinality
        assert set(result.keys()) == {0, 1}

    def test_values_are_integers(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_cardinality
        for cardinality in result.values():
            assert isinstance(cardinality, int)

    def test_correct_cardinalities(self, sigma_algebra):
        result = sigma_algebra.atom_id_to_cardinality
        assert result[0] == 2
        assert result[1] == 2

    def test_with_uneven_partition(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0, "omega3": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        result = sigma.atom_id_to_cardinality
        assert result[0] == 3
        assert result[1] == 1


class TestIsMeasurable:
    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_atom_is_measurable(self, sigma_algebra):
        event = Event(sigma_algebra.sample_space, ["omega0", "omega1"])
        assert sigma_algebra.is_measurable(event)

    def test_union_of_atoms_is_measurable(self, sigma_algebra):
        event = Event(
            sigma_algebra.sample_space, ["omega0", "omega1", "omega2", "omega3"]
        )
        assert sigma_algebra.is_measurable(event)

    def test_partial_atom_is_not_measurable(self, sigma_algebra):
        event = Event(sigma_algebra.sample_space, ["omega0"])
        assert not sigma_algebra.is_measurable(event)

    def test_empty_event_is_measurable(self, sigma_algebra):
        event = Event(sigma_algebra.sample_space, [])
        assert sigma_algebra.is_measurable(event)

    def test_full_space_is_measurable(self, sigma_algebra):
        event = Event(
            sigma_algebra.sample_space, list(sigma_algebra.sample_space.values)
        )
        assert sigma_algebra.is_measurable(event)

    def test_mixed_atoms_not_measurable(self, sigma_algebra):
        event = Event(sigma_algebra.sample_space, ["omega0", "omega2"])
        assert not sigma_algebra.is_measurable(event)

    def test_subset_of_atom_not_measurable(self, sigma_algebra):
        event = Event(sigma_algebra.sample_space, ["omega2"])
        assert not sigma_algebra.is_measurable(event)

    def test_invalid_event_type_raises_error(self, sigma_algebra):
        with pytest.raises(TypeError, match="must be an Event"):
            sigma_algebra.is_measurable("not an event")

    def test_event_from_different_space_raises_error(self, sigma_algebra):
        other_space = SampleSpace(["a", "b", "c"])
        event = Event(other_space, ["a", "b"])
        with pytest.raises(ValueError, match="same sample_space"):
            sigma_algebra.is_measurable(event)


class TestGetAtomContaining:
    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_get_atom_containing_valid_id(self, sigma_algebra):
        atom = sigma_algebra.get_atom_containing("omega0")
        assert isinstance(atom, Event)
        assert set(atom.values) == {"omega0", "omega1"}

    def test_get_atom_containing_returns_correct_atom(self, sigma_algebra):
        atom = sigma_algebra.get_atom_containing("omega2")
        assert set(atom.values) == {"omega2", "omega3"}

    def test_get_atom_containing_each_sample_point(self, sigma_algebra):
        atom0 = sigma_algebra.get_atom_containing("omega0")
        atom1 = sigma_algebra.get_atom_containing("omega1")
        atom2 = sigma_algebra.get_atom_containing("omega2")
        atom3 = sigma_algebra.get_atom_containing("omega3")
        assert atom0 == atom1
        assert atom2 == atom3
        assert atom0 != atom2

    def test_get_atom_containing_invalid_id(self, sigma_algebra):
        with pytest.raises(ValueError, match="not in sample space"):
            sigma_algebra.get_atom_containing("invalid")


class TestPowerSet:
    def test_power_set_creation(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.power_set(space)
        assert isinstance(sigma, SigmaAlgebra)

    def test_power_set_has_unique_atom_for_each_point(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.power_set(space)
        assert sigma.num_atoms == 3

    def test_power_set_atom_ids_are_integers(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.power_set(space)
        atom_ids = sigma.sample_id_to_atom_id
        assert set(atom_ids.values()) == {0, 1, 2}

    def test_power_set_singletons_are_measurable(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.power_set(space)
        for idx in space.values:
            event = Event(space, [idx])
            assert sigma.is_measurable(event)

    def test_power_set_all_subsets_measurable(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.power_set(space)
        event1 = Event(space, ["omega0", "omega1"])
        event2 = Event(space, ["omega1"])
        event3 = Event(space, ["omega0", "omega2"])
        assert sigma.is_measurable(event1)
        assert sigma.is_measurable(event2)
        assert sigma.is_measurable(event3)

    def test_power_set_with_single_element_space(self):
        space = SampleSpace(["omega0"])
        sigma = SigmaAlgebra.power_set(space)
        assert sigma.num_atoms == 1


class TestTrivial:
    def test_trivial_creation(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.trivial(space)
        assert isinstance(sigma, SigmaAlgebra)

    def test_trivial_has_single_atom(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.trivial(space)
        assert sigma.num_atoms == 1

    def test_trivial_all_points_have_same_atom_id(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.trivial(space)
        atom_ids = sigma.sample_id_to_atom_id
        assert len(set(atom_ids.values())) == 1
        assert 0 in atom_ids.values()

    def test_trivial_only_empty_and_full_measurable(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.trivial(space)
        empty = Event(space, [])
        full = Event(space, list(space.values))
        partial = Event(space, ["omega0"])
        assert sigma.is_measurable(empty)
        assert sigma.is_measurable(full)
        assert not sigma.is_measurable(partial)

    def test_trivial_single_atom_contains_all_points(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        sigma = SigmaAlgebra.trivial(space)
        events = sigma.atom_id_to_event

        assert len(events) == 1
        atom = list(events.values())[0]
        assert set(atom.values) == set(space.values)


class TestIteration:
    @pytest.fixture
    def sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

    def test_iteration_yields_tuples(self, sigma_algebra):
        for atom_id, event in sigma_algebra:
            assert isinstance(atom_id, (int, str, tuple))
            assert isinstance(event, Event)

    def test_iteration_covers_all_atoms(self, sigma_algebra):
        atom_ids_seen = set()
        for atom_id, _ in sigma_algebra:
            atom_ids_seen.add(atom_id)
        assert atom_ids_seen == {0, 1}

    def test_can_convert_to_dict(self, sigma_algebra):
        atoms_dict = dict(sigma_algebra)
        assert len(atoms_dict) == 2
        assert all(isinstance(event, Event) for event in atoms_dict.values())

    def test_iteration_matches_atom_id_to_event(self, sigma_algebra):
        from_iter = dict(sigma_algebra)
        from_property = sigma_algebra.atom_id_to_event
        assert set(from_iter.keys()) == set(from_property.keys())
        for atom_id in from_iter:
            assert set(from_iter[atom_id].values) == set(from_property[atom_id].values)

    def test_iteration_with_string_atom_ids(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        atom_ids_seen = []
        for atom_id, _ in sigma:
            atom_ids_seen.append(atom_id)
        assert set(atom_ids_seen) == {"A", "B"}

    def test_iteration_order_is_consistent(self, sigma_algebra):
        keys1 = [atom_id for atom_id, _ in sigma_algebra]
        keys2 = [atom_id for atom_id, _ in sigma_algebra]
        assert keys1 == keys2


class TestEquality:
    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["omega0", "omega1", "omega2"])

    def test_equality_same_components(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma1 = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        sigma2 = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        assert sigma1 == sigma2

    def test_equality_different_atom_ids(self, sample_space):
        atom_ids1 = {"omega0": 0, "omega1": 0, "omega2": 1}
        atom_ids2 = {"omega0": 0, "omega1": 1, "omega2": 1}
        sigma1 = SigmaAlgebra(sample_id_to_atom_id=atom_ids1, sample_space=sample_space)
        sigma2 = SigmaAlgebra(sample_id_to_atom_id=atom_ids2, sample_space=sample_space)
        assert sigma1 != sigma2

    def test_equality_different_sample_spaces(self):
        space1 = SampleSpace(["omega0", "omega1"])
        space2 = SampleSpace(["a", "b"])
        atom_ids1 = {"omega0": 0, "omega1": 0}
        atom_ids2 = {"a": 0, "b": 0}
        sigma1 = SigmaAlgebra(sample_id_to_atom_id=atom_ids1, sample_space=space1)
        sigma2 = SigmaAlgebra(sample_id_to_atom_id=atom_ids2, sample_space=space2)
        assert sigma1 != sigma2

    def test_equality_with_non_sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=sample_space)
        assert sigma != "not a sigma algebra"
        assert sigma != 123
        assert sigma != sample_space


class TestEdgeCases:
    def test_single_atom_sigma_algebra(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert sigma.num_atoms == 1
        events = sigma.atom_id_to_event
        atom = events[0]
        assert len(atom) == 3

    def test_all_distinct_atoms(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 1, "omega2": 2}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert sigma.num_atoms == 3

    def test_partition_into_two_equal_parts(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        atoms = sigma.atom_id_to_event
        assert len(atoms[0]) == 2
        assert len(atoms[1]) == 2

    def test_uneven_partition(self):
        space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0, "omega3": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        atoms = sigma.atom_id_to_event
        assert len(atoms[0]) == 3
        assert len(atoms[1]) == 1

    def test_single_element_space(self):
        space = SampleSpace(["omega0"])
        atom_ids = {"omega0": 0}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert sigma.num_atoms == 1
        assert sigma.is_measurable(Event(space, []))
        assert sigma.is_measurable(Event(space, ["omega0"]))

    def test_large_partition(self):
        indices = [f"omega{i}" for i in range(100)]
        space = SampleSpace(indices)
        atom_ids = {idx: i // 10 for i, idx in enumerate(indices)}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        assert sigma.num_atoms == 10


class TestAtomIdTypes:
    def test_integer_atom_ids(self):
        space = SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": 0, "omega1": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        events = sigma.atom_id_to_event
        assert 0 in events
        assert 1 in events

    def test_string_atom_ids(self):
        space = SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": "atom_A", "omega1": "atom_B"}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        events = sigma.atom_id_to_event
        assert "atom_A" in events
        assert "atom_B" in events

    def test_tuple_atom_ids(self):
        space = SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": (0, 0), "omega1": (1, 1)}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        events = sigma.atom_id_to_event
        assert (0, 0) in events
        assert (1, 1) in events

    def test_float_atom_ids(self):
        space = SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": 0.5, "omega1": 1.5}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)
        events = sigma.atom_id_to_event
        assert 0.5 in events
        assert 1.5 in events


class TestMeasurabilityWithDifferentAtomTypes:
    def test_measurability_with_string_atoms(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

        atom_A = Event(space, ["omega0", "omega1"])
        atom_B = Event(space, ["omega2"])
        partial = Event(space, ["omega0"])

        assert sigma.is_measurable(atom_A)
        assert sigma.is_measurable(atom_B)
        assert not sigma.is_measurable(partial)

    def test_measurability_with_tuple_atoms(self):
        space = SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": (0, 0), "omega1": (0, 0), "omega2": (1, 1)}
        sigma = SigmaAlgebra(sample_id_to_atom_id=atom_ids, sample_space=space)

        atom_00 = Event(space, ["omega0", "omega1"])
        assert sigma.is_measurable(atom_00)


class TestOrderRelations:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace(["s0", "s1", "s2", "s3"])

    def test_le_trivial_and_power_set(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial <= power_set
        assert not power_set <= trivial

    def test_le_reflexive(self, sample_space):
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma = SigmaAlgebra(sample_space=sample_space, sample_id_to_atom_id=atom_ids)
        assert sigma <= sigma

    def test_le_coarser_and_finer(self, sample_space):
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
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        A = SigmaAlgebra.trivial(sample_space=sample_space)
        B_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        B = SigmaAlgebra(sample_space=sample_space, sample_id_to_atom_id=B_atom_ids)
        C = SigmaAlgebra.power_set(sample_space=sample_space)
        assert A <= B
        assert B <= C
        assert A <= C

    def test_le_with_different_sample_spaces_raises_error(self):
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        sigma1 = SigmaAlgebra.trivial(sample_space=sample_space1)
        sigma2 = SigmaAlgebra.trivial(sample_space=sample_space2)
        with pytest.raises(ValueError, match="same sample space"):
            _ = sigma1 <= sigma2

    def test_le_with_non_sigma_algebra_returns_not_implemented(self, sample_space):
        sigma = SigmaAlgebra.trivial(sample_space=sample_space)
        with pytest.raises(TypeError):
            _ = sigma <= "not a sigma algebra"

    def test_lt_proper_sub_algebra(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial < power_set
        assert not power_set < trivial

    def test_lt_not_proper_when_equal(self, sample_space):
        sigma = SigmaAlgebra.trivial(sample_space=sample_space)
        assert not sigma < sigma

    def test_lt_with_different_sample_spaces_raises_error(self):
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        sigma1 = SigmaAlgebra.trivial(sample_space=sample_space1)
        sigma2 = SigmaAlgebra.trivial(sample_space=sample_space2)
        with pytest.raises(ValueError, match="same sample space"):
            _ = sigma1 < sigma2

    def test_lt_with_non_sigma_algebra_returns_not_implemented(self, sample_space):
        sigma = SigmaAlgebra.trivial(sample_space=sample_space)
        with pytest.raises(TypeError):
            _ = sigma < "not a sigma algebra"

    def test_ge_power_set_and_trivial(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert power_set >= trivial
        assert not trivial >= power_set

    def test_ge_reflexive(self, sample_space):
        sigma = SigmaAlgebra.trivial(sample_space=sample_space)
        assert sigma >= sigma

    def test_ge_finer_and_coarser(self, sample_space):
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

    def test_ge_with_different_sample_spaces_raises_error(self):
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        sigma1 = SigmaAlgebra.trivial(sample_space=sample_space1)
        sigma2 = SigmaAlgebra.trivial(sample_space=sample_space2)
        with pytest.raises(ValueError, match="same sample space"):
            _ = sigma1 >= sigma2

    def test_ge_with_non_sigma_algebra_returns_not_implemented(self, sample_space):
        sigma = SigmaAlgebra.trivial(sample_space=sample_space)
        with pytest.raises(TypeError):
            _ = sigma >= "not a sigma algebra"

    def test_gt_proper_super_algebra(self, sample_space):
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert power_set > trivial
        assert not trivial > power_set

    def test_gt_not_proper_when_equal(self, sample_space):
        sigma = SigmaAlgebra.trivial(sample_space=sample_space)
        assert not sigma > sigma

    def test_gt_with_different_sample_spaces_raises_error(self):
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        sigma1 = SigmaAlgebra.trivial(sample_space=sample_space1)
        sigma2 = SigmaAlgebra.trivial(sample_space=sample_space2)
        with pytest.raises(ValueError, match="same sample space"):
            _ = sigma1 > sigma2

    def test_gt_with_non_sigma_algebra_returns_not_implemented(self, sample_space):
        sigma = SigmaAlgebra.trivial(sample_space=sample_space)
        with pytest.raises(TypeError):
            _ = sigma > "not a sigma algebra"

    def test_incomparable_sigma_algebras(self, sample_space):
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
        atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        sigma1 = SigmaAlgebra(sample_space=sample_space, sample_id_to_atom_id=atom_ids)
        sigma2 = SigmaAlgebra(sample_space=sample_space, sample_id_to_atom_id=atom_ids)
        assert sigma1 <= sigma2
        assert sigma2 <= sigma1
        assert sigma1 == sigma2

    def test_order_with_probability_spaces(self):
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_space = ProbabilitySpace.from_probabilities(
            sample_space=sample_space, probabilities=probabilities
        )
        coarse_atom_ids = {"s0": 0, "s1": 0, "s2": 0, "s3": 1}
        coarse = SigmaAlgebra(
            sample_id_to_atom_id=coarse_atom_ids, sample_space=sample_space
        )
        coarse.probability_space = prob_space
        fine_atom_ids = {"s0": 0, "s1": 0, "s2": 1, "s3": 2}
        fine = SigmaAlgebra(
            sample_id_to_atom_id=fine_atom_ids, sample_space=sample_space
        )
        fine.probability_space = prob_space
        assert coarse <= fine
        assert coarse < fine
        assert fine >= coarse
        assert fine > coarse

    def test_single_element_sample_space_all_equal(self):
        sample_space = SampleSpace(["s0"])
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        assert trivial <= power_set
        assert power_set <= trivial
        assert not trivial < power_set
        assert not power_set < trivial
        assert trivial == power_set


class TestValidation:
    """Test validation of SigmaAlgebra constructor parameters."""

    def test_cannot_provide_both_sample_id_to_atom_id_and_values(self):
        """Cannot provide both sample_id_to_atom_id and values."""
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1}
        values = pd.Series([0, 0, 1], index=["s0", "s1", "s2"])
        with pytest.raises(
            ValueError,
            match="Cannot provide both sample_id_to_atom_id/sample_space and values",
        ):
            SigmaAlgebra(sample_id_to_atom_id=sample_id_to_atom_id, values=values)

    def test_cannot_provide_sample_space_and_values(self):
        """Cannot provide both sample_space and values."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        values = pd.Series([0, 0, 1], index=["s0", "s1", "s2"])
        with pytest.raises(
            ValueError,
            match="Cannot provide both sample_id_to_atom_id/sample_space and values",
        ):
            SigmaAlgebra(sample_space=sample_space, values=values)

    def test_must_provide_either_sample_id_to_atom_id_or_values(self):
        """Must provide either sample_id_to_atom_id or values."""
        with pytest.raises(
            ValueError,
            match="Must provide either sample_id_to_atom_id or values",
        ):
            SigmaAlgebra(name="F")

    def test_sample_id_to_atom_id_must_be_dict(self):
        """sample_id_to_atom_id must be a dictionary."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            SigmaAlgebra(sample_id_to_atom_id=[0, 0, 1, 1])

    def test_sample_id_to_atom_id_must_be_dict_not_list_of_tuples(self):
        """sample_id_to_atom_id must be dict, not list of tuples."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            SigmaAlgebra(sample_id_to_atom_id=[("s0", 0), ("s1", 1)])

    def test_sample_space_must_be_sample_space_instance(self):
        """sample_space must be a SampleSpace instance."""
        sample_id_to_atom_id = {"s0": 0, "s1": 1}
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            SigmaAlgebra(
                sample_id_to_atom_id=sample_id_to_atom_id,
                sample_space="not a sample space",
            )

    def test_sample_space_must_be_sample_space_not_list(self):
        """sample_space must be SampleSpace, not a list."""
        sample_id_to_atom_id = {"s0": 0, "s1": 1}
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            SigmaAlgebra(
                sample_id_to_atom_id=sample_id_to_atom_id, sample_space=["s0", "s1"]
            )

    def test_sample_id_to_atom_id_keys_must_match_sample_space(self):
        """Keys in sample_id_to_atom_id must match sample_space samples."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s3": 1}
        with pytest.raises(
            ValueError, match="must contain an entry for every sample index"
        ):
            SigmaAlgebra(
                sample_id_to_atom_id=sample_id_to_atom_id, sample_space=sample_space
            )

    def test_values_must_be_series(self):
        """values must be a pandas Series."""
        with pytest.raises(TypeError, match="values must be a pandas Series"):
            SigmaAlgebra(values=[0, 0, 1])

    def test_values_must_be_series_not_dataframe(self):
        """values must be Series, not DataFrame."""
        df = pd.DataFrame([[0], [1], [2]])
        with pytest.raises(TypeError, match="values must be a pandas Series"):
            SigmaAlgebra(values=df)

    def test_values_must_be_series_not_dict(self):
        """values must be Series, not dict."""
        with pytest.raises(TypeError, match="values must be a pandas Series"):
            SigmaAlgebra(values={"s0": 0, "s1": 1})

    def test_name_must_be_string(self):
        """name must be a string."""
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1}
        with pytest.raises(TypeError, match="name must be a string"):
            SigmaAlgebra(sample_id_to_atom_id=sample_id_to_atom_id, name=123)

    def test_name_must_be_string_not_none(self):
        """name must be a string, not None."""
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1}
        with pytest.raises(TypeError, match="name must be a string"):
            SigmaAlgebra(sample_id_to_atom_id=sample_id_to_atom_id, name=None)

    def test_name_setter_validation(self):
        """Setting name to non-string raises TypeError."""
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1}
        sigma = SigmaAlgebra(sample_id_to_atom_id=sample_id_to_atom_id)
        with pytest.raises(TypeError, match="name must be a string"):
            sigma.name = 456


class TestValuesConstruction:
    """Test construction of SigmaAlgebra from values parameter."""

    def test_construction_from_values(self):
        """Can construct SigmaAlgebra from values Series."""
        values = pd.Series([0, 0, 1, 1], index=["s0", "s1", "s2", "s3"], name="F")
        sigma = SigmaAlgebra(values=values)
        assert sigma.name == "F"
        expected_values = pd.Series(
            [0, 0, 1, 1], index=["s0", "s1", "s2", "s3"], name="F"
        )
        pd.testing.assert_series_equal(sigma.values, expected_values)

    def test_construction_from_values_with_explicit_name(self):
        """Explicit name parameter overrides Series name."""
        values = pd.Series([0, 0, 1], index=["s0", "s1", "s2"], name="G")
        sigma = SigmaAlgebra(values=values, name="H")
        assert sigma.name == "G"

    def test_construction_from_values_no_name(self):
        """Can construct from values without name in Series."""
        values = pd.Series([0, 1, 2], index=["s0", "s1", "s2"])
        sigma = SigmaAlgebra(values=values, name="F")
        assert sigma.name == "F"

    def test_sample_space_derived_from_values(self):
        """sample_space is derived from values index."""
        values = pd.Series([0, 0, 1], index=["a", "b", "c"])
        sigma = SigmaAlgebra(values=values)
        assert isinstance(sigma.sample_space, SampleSpace)
        assert sigma.sample_space == SampleSpace(["a", "b", "c"])

    def test_sample_id_to_atom_id_derived_from_values(self):
        """sample_id_to_atom_id is derived from values."""
        values = pd.Series([0, 0, 1, 1], index=["s0", "s1", "s2", "s3"])
        sigma = SigmaAlgebra(values=values)
        expected = {"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        assert sigma.sample_id_to_atom_id == expected

    def test_atom_id_to_sample_ids_with_values(self):
        """atom_id_to_sample_ids works correctly with values construction."""
        values = pd.Series([0, 0, 1], index=["x", "y", "z"])
        sigma = SigmaAlgebra(values=values)
        result = sigma.atom_id_to_sample_ids
        assert set(result[0]) == {"x", "y"}
        assert set(result[1]) == {"z"}

    def test_is_measurable_with_values(self):
        """is_measurable works with values construction."""
        values = pd.Series([0, 0, 1, 1], index=["s0", "s1", "s2", "s3"])
        sigma = SigmaAlgebra(values=values)
        sample_space = sigma.sample_space
        assert sigma.is_measurable(sample_space.get_event(["s0", "s1"]))
        assert sigma.is_measurable(sample_space.get_event(["s2", "s3"]))
        assert sigma.is_measurable(sample_space.get_event(["s0", "s1", "s2", "s3"]))
        assert not sigma.is_measurable(sample_space.get_event(["s0", "s2"]))

    def test_equality_with_values_construction(self):
        """SigmaAlgebras constructed from values can be compared."""
        values1 = pd.Series([0, 0, 1], index=["s0", "s1", "s2"])
        sigma1 = SigmaAlgebra(values=values1)
        values2 = pd.Series([0, 0, 1], index=["s0", "s1", "s2"])
        sigma2 = SigmaAlgebra(values=values2)
        assert sigma1 == sigma2

    def test_equality_mixed_construction_methods(self):
        """SigmaAlgebras from different construction methods are equal if same partition."""
        values = pd.Series([0, 0, 1], index=["s0", "s1", "s2"])
        sigma1 = SigmaAlgebra(values=values)
        sample_id_to_atom_id = {"s0": 0, "s1": 0, "s2": 1}
        sigma2 = SigmaAlgebra(sample_id_to_atom_id=sample_id_to_atom_id)
        assert sigma1 == sigma2

    def test_values_with_string_atom_ids(self):
        """Can use string atom IDs in values."""
        values = pd.Series(["A", "A", "B", "B"], index=["s0", "s1", "s2", "s3"])
        sigma = SigmaAlgebra(values=values)
        assert sigma.sample_id_to_atom_id == {
            "s0": "A",
            "s1": "A",
            "s2": "B",
            "s3": "B",
        }

    def test_order_relations_with_values(self):
        """Order relations work with values construction."""
        coarse_values = pd.Series([0, 0, 0, 1], index=["s0", "s1", "s2", "s3"])
        coarse = SigmaAlgebra(values=coarse_values)
        fine_values = pd.Series([0, 0, 1, 2], index=["s0", "s1", "s2", "s3"])
        fine = SigmaAlgebra(values=fine_values)
        assert coarse <= fine
        assert coarse < fine
        assert fine >= coarse
        assert fine > coarse

    def test_factory_methods_preserve_behavior(self):
        """Factory methods like power_set and trivial still work correctly."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        power_set = SigmaAlgebra.power_set(sample_space=sample_space)
        trivial = SigmaAlgebra.trivial(sample_space=sample_space)
        assert power_set > trivial
        assert trivial < power_set
