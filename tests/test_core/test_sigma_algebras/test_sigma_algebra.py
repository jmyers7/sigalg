import pytest

import sigalg as sa


class TestConstruction:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_construction_with_integer_atom_ids(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        assert sigma.sample_space == sample_space
        assert sigma.atom_ids == atom_ids

    def test_construction_with_string_atom_ids(self, sample_space):
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B", "omega3": "B"}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        assert sigma.atom_ids == atom_ids

    def test_construction_with_tuple_atom_ids(self, sample_space):
        atom_ids = {
            "omega0": (0, 0),
            "omega1": (0, 1),
            "omega2": (1, 0),
            "omega3": (1, 1),
        }
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        assert sigma.atom_ids == atom_ids

    def test_construction_with_mixed_hashable_atom_ids(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": "special", "omega2": 0, "omega3": (1, 2)}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        assert sigma.atom_ids == atom_ids

    def test_construction_creates_atom_mapping(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        assert len(sigma._atom_id_to_sample_ids) == 2
        assert set(sigma._atom_id_to_sample_ids[0]) == {"omega0", "omega1"}
        assert set(sigma._atom_id_to_sample_ids[1]) == {"omega2", "omega3"}

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.SigmaAlgebra("not a space", {"omega0": 0})

    def test_construction_with_non_dict_atom_ids(self, sample_space):
        with pytest.raises(TypeError, match="must be a dictionary"):
            sa.SigmaAlgebra(sample_space, [0, 0, 1, 1])

    def test_construction_with_missing_sample_indices(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0}
        with pytest.raises(ValueError, match="must contain an entry for every"):
            sa.SigmaAlgebra(sample_space, atom_ids)

    def test_construction_with_extra_sample_indices(self, sample_space):
        atom_ids = {
            "omega0": 0,
            "omega1": 0,
            "omega2": 1,
            "omega3": 1,
            "extra": 2,
        }
        with pytest.raises(ValueError, match="must contain an entry for every"):
            sa.SigmaAlgebra(sample_space, atom_ids)

    def test_construction_with_unhashable_atom_ids(self, sample_space):
        atom_ids = {
            "omega0": [1, 2],
            "omega1": [1, 2],
            "omega2": [3, 4],
            "omega3": [3, 4],
        }
        with pytest.raises(TypeError, match="must be hashable"):
            sa.SigmaAlgebra(sample_space, atom_ids)

    def test_construction_preserves_atom_id_types(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        for atom_id in sigma.atom_ids.values():
            assert isinstance(atom_id, int)


class TestProperties:

    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return sa.SigmaAlgebra(sample_space, atom_ids)

    def test_sample_space_property(self, sigma_algebra, sample_space):
        assert sigma_algebra.sample_space == sample_space

    def test_sample_space_has_correct_indices(self, sigma_algebra, sample_space):
        assert sigma_algebra.sample_space.index.equals(sample_space.index)

    def test_atom_ids_property_returns_copy(self, sigma_algebra):
        atom_ids = sigma_algebra.atom_ids
        atom_ids["omega0"] = 999
        assert sigma_algebra.atom_ids["omega0"] == 0

    def test_atom_ids_property_has_correct_values(self, sigma_algebra):
        atom_ids = sigma_algebra.atom_ids
        assert atom_ids == {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}

    def test_num_atoms_property(self, sigma_algebra):
        assert sigma_algebra.num_atoms == 2

    def test_num_atoms_with_single_atom(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        assert sigma.num_atoms == 1

    def test_num_atoms_with_all_distinct(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 1, "omega2": 2}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        assert sigma.num_atoms == 3


class TestToEvents:
    @pytest.fixture
    def sigma_algebra(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return sa.SigmaAlgebra(space, atom_ids)

    def test_to_events_returns_dict(self, sigma_algebra):
        events = sigma_algebra.to_events()
        assert isinstance(events, dict)

    def test_to_events_has_correct_number_of_atoms(self, sigma_algebra):
        events = sigma_algebra.to_events()
        assert len(events) == 2

    def test_to_events_keys_are_atom_ids(self, sigma_algebra):
        events = sigma_algebra.to_events()
        assert set(events.keys()) == {0, 1}

    def test_to_events_values_are_events(self, sigma_algebra):
        events = sigma_algebra.to_events()
        for event in events.values():
            assert isinstance(event, sa.Event)

    def test_to_events_atoms_have_correct_indices(self, sigma_algebra):
        events = sigma_algebra.to_events()
        atom_0 = events[0]
        atom_1 = events[1]
        assert set(atom_0.index) == {"omega0", "omega1"}
        assert set(atom_1.index) == {"omega2", "omega3"}

    def test_to_events_with_string_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        events = sigma.to_events()
        assert set(events.keys()) == {"A", "B"}
        assert set(events["A"].index) == {"omega0", "omega1"}
        assert set(events["B"].index) == {"omega2"}

    def test_to_events_with_tuple_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": (0, 0), "omega1": (1, 1)}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        events = sigma.to_events()
        assert (0, 0) in events
        assert (1, 1) in events


class TestIsMeasurable:
    @pytest.fixture
    def sigma_algebra(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return sa.SigmaAlgebra(space, atom_ids)

    def test_atom_is_measurable(self, sigma_algebra):
        event = sa.Event(sigma_algebra.sample_space, ["omega0", "omega1"])
        assert sigma_algebra.is_measurable(event)

    def test_union_of_atoms_is_measurable(self, sigma_algebra):
        event = sa.Event(
            sigma_algebra.sample_space, ["omega0", "omega1", "omega2", "omega3"]
        )
        assert sigma_algebra.is_measurable(event)

    def test_partial_atom_is_not_measurable(self, sigma_algebra):
        event = sa.Event(sigma_algebra.sample_space, ["omega0"])
        assert not sigma_algebra.is_measurable(event)

    def test_empty_event_is_measurable(self, sigma_algebra):
        event = sa.Event(sigma_algebra.sample_space, [])
        assert sigma_algebra.is_measurable(event)

    def test_full_space_is_measurable(self, sigma_algebra):
        event = sa.Event(
            sigma_algebra.sample_space, list(sigma_algebra.sample_space.index)
        )
        assert sigma_algebra.is_measurable(event)

    def test_mixed_atoms_not_measurable(self, sigma_algebra):
        event = sa.Event(sigma_algebra.sample_space, ["omega0", "omega2"])
        assert not sigma_algebra.is_measurable(event)

    def test_subset_of_atom_not_measurable(self, sigma_algebra):
        event = sa.Event(sigma_algebra.sample_space, ["omega2"])
        assert not sigma_algebra.is_measurable(event)

    def test_invalid_event_type_raises_error(self, sigma_algebra):
        with pytest.raises(TypeError, match="must be an Event"):
            sigma_algebra.is_measurable("not an event")

    def test_event_from_different_space_raises_error(self, sigma_algebra):
        other_space = sa.SampleSpace(["a", "b", "c"])
        event = sa.Event(other_space, ["a", "b"])
        with pytest.raises(ValueError, match="same sample_space"):
            sigma_algebra.is_measurable(event)


class TestGetAtomContaining:
    @pytest.fixture
    def sigma_algebra(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return sa.SigmaAlgebra(space, atom_ids)

    def test_get_atom_containing_valid_id(self, sigma_algebra):
        atom = sigma_algebra.get_atom_containing("omega0")
        assert isinstance(atom, sa.Event)
        assert set(atom.index) == {"omega0", "omega1"}

    def test_get_atom_containing_returns_correct_atom(self, sigma_algebra):
        atom = sigma_algebra.get_atom_containing("omega2")
        assert set(atom.index) == {"omega2", "omega3"}

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
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.power_set(space)
        assert isinstance(sigma, sa.SigmaAlgebra)

    def test_power_set_has_unique_atom_for_each_point(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.power_set(space)
        assert sigma.num_atoms == 3

    def test_power_set_atom_ids_are_integers(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.power_set(space)
        atom_ids = sigma.atom_ids
        assert set(atom_ids.values()) == {0, 1, 2}

    def test_power_set_singletons_are_measurable(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.power_set(space)
        for idx in space.index:
            event = sa.Event(space, [idx])
            assert sigma.is_measurable(event)

    def test_power_set_all_subsets_measurable(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.power_set(space)
        event1 = sa.Event(space, ["omega0", "omega1"])
        event2 = sa.Event(space, ["omega1"])
        event3 = sa.Event(space, ["omega0", "omega2"])
        assert sigma.is_measurable(event1)
        assert sigma.is_measurable(event2)
        assert sigma.is_measurable(event3)

    def test_power_set_with_single_element_space(self):
        space = sa.SampleSpace(["omega0"])
        sigma = sa.SigmaAlgebra.power_set(space)
        assert sigma.num_atoms == 1


class TestTrivial:
    def test_trivial_creation(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.trivial(space)
        assert isinstance(sigma, sa.SigmaAlgebra)

    def test_trivial_has_single_atom(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.trivial(space)
        assert sigma.num_atoms == 1

    def test_trivial_all_points_have_same_atom_id(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.trivial(space)
        atom_ids = sigma.atom_ids
        assert len(set(atom_ids.values())) == 1
        assert 0 in atom_ids.values()

    def test_trivial_only_empty_and_full_measurable(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.trivial(space)
        empty = sa.Event(space, [])
        full = sa.Event(space, list(space.index))
        partial = sa.Event(space, ["omega0"])
        assert sigma.is_measurable(empty)
        assert sigma.is_measurable(full)
        assert not sigma.is_measurable(partial)

    def test_trivial_single_atom_contains_all_points(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        sigma = sa.SigmaAlgebra.trivial(space)
        events = sigma.to_events()

        assert len(events) == 1
        atom = list(events.values())[0]
        assert set(atom.index) == set(space.index)


class TestIteration:
    @pytest.fixture
    def sigma_algebra(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        return sa.SigmaAlgebra(space, atom_ids)

    def test_iteration_yields_tuples(self, sigma_algebra):
        for atom_id, event in sigma_algebra:
            assert isinstance(atom_id, (int, str, tuple))
            assert isinstance(event, sa.Event)

    def test_iteration_covers_all_atoms(self, sigma_algebra):
        atom_ids_seen = set()
        for atom_id, _ in sigma_algebra:
            atom_ids_seen.add(atom_id)
        assert atom_ids_seen == {0, 1}

    def test_can_convert_to_dict(self, sigma_algebra):
        atoms_dict = dict(sigma_algebra)
        assert len(atoms_dict) == 2
        assert all(isinstance(event, sa.Event) for event in atoms_dict.values())

    def test_iteration_with_string_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = sa.SigmaAlgebra(space, atom_ids)
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
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_equality_same_components(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma1 = sa.SigmaAlgebra(sample_space, atom_ids)
        sigma2 = sa.SigmaAlgebra(sample_space, atom_ids)
        assert sigma1 == sigma2

    def test_equality_different_atom_ids(self, sample_space):
        atom_ids1 = {"omega0": 0, "omega1": 0, "omega2": 1}
        atom_ids2 = {"omega0": 0, "omega1": 1, "omega2": 1}
        sigma1 = sa.SigmaAlgebra(sample_space, atom_ids1)
        sigma2 = sa.SigmaAlgebra(sample_space, atom_ids2)
        assert sigma1 != sigma2

    def test_equality_different_sample_spaces(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["a", "b"])
        atom_ids1 = {"omega0": 0, "omega1": 0}
        atom_ids2 = {"a": 0, "b": 0}
        sigma1 = sa.SigmaAlgebra(space1, atom_ids1)
        sigma2 = sa.SigmaAlgebra(space2, atom_ids2)
        assert sigma1 != sigma2

    def test_equality_with_non_sigma_algebra(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        assert sigma != "not a sigma algebra"
        assert sigma != 123
        assert sigma != sample_space

    def test_equality_same_partition_different_atom_id_values(self, sample_space):
        atom_ids1 = {"omega0": 0, "omega1": 0, "omega2": 1}
        atom_ids2 = {"omega0": 5, "omega1": 5, "omega2": 10}
        sigma1 = sa.SigmaAlgebra(sample_space, atom_ids1)
        sigma2 = sa.SigmaAlgebra(sample_space, atom_ids2)
        assert sigma1 != sigma2


class TestHashing:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_sigma_algebra_is_hashable(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        hash_value = hash(sigma)
        assert isinstance(hash_value, int)

    def test_equal_sigma_algebras_have_same_hash(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma1 = sa.SigmaAlgebra(sample_space, atom_ids)
        sigma2 = sa.SigmaAlgebra(sample_space, atom_ids)
        assert hash(sigma1) == hash(sigma2)

    def test_can_use_in_set(self, sample_space):
        atom_ids1 = {"omega0": 0, "omega1": 0, "omega2": 1}
        atom_ids2 = {"omega0": 0, "omega1": 1, "omega2": 1}
        sigma1 = sa.SigmaAlgebra(sample_space, atom_ids1)
        sigma2 = sa.SigmaAlgebra(sample_space, atom_ids2)
        sigma_set = {sigma1, sigma2}
        assert len(sigma_set) == 2

    def test_can_use_as_dict_key(self, sample_space):
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        sigma_dict = {sigma: "value"}
        assert sigma_dict[sigma] == "value"

    def test_hash_with_string_atom_ids(self, sample_space):
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = sa.SigmaAlgebra(sample_space, atom_ids)
        hash_value = hash(sigma)
        assert isinstance(hash_value, int)


class TestEdgeCases:
    def test_single_atom_sigma_algebra(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        assert sigma.num_atoms == 1
        events = sigma.to_events()
        atom = events[0]
        assert len(atom) == 3

    def test_all_distinct_atoms(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": 1, "omega2": 2}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        assert sigma.num_atoms == 3

    def test_partition_into_two_equal_parts(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        atoms = sigma.to_events()
        assert len(atoms[0]) == 2
        assert len(atoms[1]) == 2

    def test_uneven_partition(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 0, "omega3": 1}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        atoms = sigma.to_events()
        assert len(atoms[0]) == 3
        assert len(atoms[1]) == 1

    def test_single_element_space(self):
        space = sa.SampleSpace(["omega0"])
        atom_ids = {"omega0": 0}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        assert sigma.num_atoms == 1
        assert sigma.is_measurable(sa.Event(space, []))
        assert sigma.is_measurable(sa.Event(space, ["omega0"]))

    def test_large_partition(self):
        indices = [f"omega{i}" for i in range(100)]
        space = sa.SampleSpace(indices)
        atom_ids = {idx: i // 10 for i, idx in enumerate(indices)}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        assert sigma.num_atoms == 10


class TestAtomIdTypes:
    def test_integer_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": 0, "omega1": 1}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        events = sigma.to_events()
        assert 0 in events
        assert 1 in events

    def test_string_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": "atom_A", "omega1": "atom_B"}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        events = sigma.to_events()
        assert "atom_A" in events
        assert "atom_B" in events

    def test_tuple_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": (0, 0), "omega1": (1, 1)}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        events = sigma.to_events()
        assert (0, 0) in events
        assert (1, 1) in events

    def test_mixed_type_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": 0, "omega1": "special", "omega2": 0}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        events = sigma.to_events()
        assert 0 in events
        assert "special" in events
        assert len(events) == 2

    def test_float_atom_ids(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        atom_ids = {"omega0": 0.5, "omega1": 1.5}
        sigma = sa.SigmaAlgebra(space, atom_ids)
        events = sigma.to_events()
        assert 0.5 in events
        assert 1.5 in events


class TestMeasurabilityWithDifferentAtomTypes:
    def test_measurability_with_string_atoms(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": "A", "omega1": "A", "omega2": "B"}
        sigma = sa.SigmaAlgebra(space, atom_ids)

        atom_A = sa.Event(space, ["omega0", "omega1"])
        atom_B = sa.Event(space, ["omega2"])
        partial = sa.Event(space, ["omega0"])

        assert sigma.is_measurable(atom_A)
        assert sigma.is_measurable(atom_B)
        assert not sigma.is_measurable(partial)

    def test_measurability_with_tuple_atoms(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        atom_ids = {"omega0": (0, 0), "omega1": (0, 0), "omega2": (1, 1)}
        sigma = sa.SigmaAlgebra(space, atom_ids)

        atom_00 = sa.Event(space, ["omega0", "omega1"])
        assert sigma.is_measurable(atom_00)
