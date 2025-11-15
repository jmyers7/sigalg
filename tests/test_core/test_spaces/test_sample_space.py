import pytest
import sigalg as sa


class TestConstruction:
    def test_construction_from_list(self):
        labels = ["omega0", "omega1", "omega2"]
        space = sa.SampleSpace(labels)
        assert len(space) == 3
        assert list(space) == labels

    def test_construction_from_tuple(self):
        labels = ("a", "b", "c")
        space = sa.SampleSpace(labels)
        assert len(space) == 3
        assert list(space) == list(labels)

    def test_construction_empty(self):
        space = sa.SampleSpace([])
        assert len(space) == 0

    def test_construction_with_duplicates_raises_error(self):
        with pytest.raises(ValueError, match="must be unique"):
            sa.SampleSpace(["omega0", "omega1", "omega0"])


class TestIndexing:
    @pytest.fixture
    def space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_getitem_by_position(self, space):
        assert space[0] == "omega0"
        assert space[2] == "omega2"

    def test_getitem_negative_index(self, space):
        assert space[-1] == "omega3"
        assert space[-2] == "omega2"

    def test_getitem_slice(self, space):
        assert list(space[1:3]) == ["omega1", "omega2"]

    def test_getitem_out_of_bounds(self, space):
        with pytest.raises(IndexError):
            _ = space[10]

    def test_getitem_with_list_returns_event(self, space):
        event = space[["omega0", "omega2"]]
        assert isinstance(event, sa.Event)

    def test_getitem_with_list_contains_correct_indices(self, space):
        event = space[["omega0", "omega2"]]
        assert list(event.index) == ["omega0", "omega2"]

    def test_getitem_with_list_references_parent_space(self, space):
        event = space[["omega0", "omega2"]]
        assert event.sample_space is space

    def test_getitem_with_empty_list(self, space):
        event = space[[]]
        assert isinstance(event, sa.Event)
        assert len(event) == 0

    def test_getitem_with_single_element_list(self, space):
        event = space[["omega1"]]
        assert isinstance(event, sa.Event)
        assert list(event.index) == ["omega1"]

    def test_getitem_with_all_indices(self, space):
        all_indices = ["omega0", "omega1", "omega2", "omega3"]
        event = space[all_indices]
        assert isinstance(event, sa.Event)
        assert list(event.index) == all_indices

    def test_getitem_with_duplicate_indices(self, space):
        with pytest.raises(ValueError, match="must be unique"):
            _ = space[["omega0", "omega0", "omega1"]]

    def test_getitem_with_invalid_index_in_list(self, space):
        with pytest.raises((ValueError, KeyError)):
            _ = space[["omega0", "invalid_index"]]

    def test_getitem_with_list_order_preserved(self, space):
        event = space[["omega3", "omega1", "omega0"]]
        assert list(event.index) == ["omega3", "omega1", "omega0"]

    def test_getitem_single_index_still_returns_string(self, space):
        result = space[0]
        assert result == "omega0"
        assert not isinstance(result, sa.Event)

    def test_getitem_slice_still_works(self, space):
        result = space[1:3]
        assert not isinstance(result, sa.Event)
        assert list(result) == ["omega1", "omega2"]


class TestIteration:
    def test_iteration(self):
        labels = ["a", "b", "c"]
        space = sa.SampleSpace(labels)
        result = [label for label in space]
        assert result == labels

    def test_iteration_empty(self):
        space = sa.SampleSpace([])
        result = list(space)
        assert result == []


class TestProperties:
    def test_index_property(self):
        labels = ["omega0", "omega1"]
        space = sa.SampleSpace(labels)
        assert list(space.index) == labels

    def test_len(self):
        space = sa.SampleSpace(["a", "b", "c", "d"])
        assert len(space) == 4

    def test_default_sigma_algebra(self):
        labels = ["x1", "x2", "x3"]
        space = sa.SampleSpace(labels)
        sigma_algebra = space.sigma_algebra
        expected_atom_ids = {"x1": 0, "x2": 1, "x3": 2}
        assert sigma_algebra._sample_space == space
        assert sigma_algebra._atom_ids == expected_atom_ids

    def test_set_sigma_algebra(self):
        labels = ["a", "b"]
        space = sa.SampleSpace(labels)
        new_atom_ids = {"a": 1, "b": 1}
        new_sigma_algebra = sa.SigmaAlgebra(sample_space=space, atom_ids=new_atom_ids)
        space.set_sigma_algebra(new_sigma_algebra)
        assert space.sigma_algebra == new_sigma_algebra

    def test_add_probability_measure(self):
        labels = ["omega0", "omega1", "omega2"]
        space = sa.SampleSpace(labels)
        probabilities = {"omega0": 0.2, "omega1": 0.5, "omega2": 0.3}
        prob_measure = sa.ProbabilityMeasure(space, probabilities)
        prob_space = space.add_probability_measure(prob_measure)
        assert isinstance(prob_space, sa.ProbabilitySpace)
        assert prob_space.probability_measure == prob_measure


class TestEquality:
    def test_equality_same_labels(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1"])
        assert space1 == space2

    def test_equality_different_labels(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega2"])
        assert space1 != space2

    def test_equality_different_order(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega1", "omega0"])
        assert space1 != space2

    def test_equality_different_length(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert space1 != space2

    def test_equality_with_non_sample_space(self):
        space = sa.SampleSpace(["omega0", "omega1"])
        assert space != ["omega0", "omega1"]
        assert space != "not a sample space"


class TestHashing:
    def test_hashable(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1"])
        spaces_set = {space1, space2}
        assert len(spaces_set) == 1

    def test_hash_equality(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1"])
        assert hash(space1) == hash(space2)

    def test_hash_inequality(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega2"])
        assert hash(space1) != hash(space2)

    def test_can_use_as_dict_key(self):
        space1 = sa.SampleSpace(["omega0", "omega1"])
        space2 = sa.SampleSpace(["omega0", "omega1"])
        d = {space1: "value1"}
        assert d[space2] == "value1"
