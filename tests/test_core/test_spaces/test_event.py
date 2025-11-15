import pytest
import sigalg as sa


class TestConstructionFromSampleSpace:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])

    def test_construction_valid_event(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega2"])
        assert len(event) == 2
        assert list(event.index) == ["omega0", "omega2"]

    def test_construction_empty_event(self, sample_space):
        event = sa.Event(sample_space, [])
        assert len(event) == 0
        assert list(event.index) == []

    def test_construction_single_element_event(self, sample_space):
        event = sa.Event(sample_space, ["omega1"])
        assert len(event) == 1
        assert list(event.index) == ["omega1"]

    def test_construction_full_space_event(self, sample_space):
        all_indices = ["omega0", "omega1", "omega2", "omega3"]
        event = sa.Event(sample_space, all_indices)
        assert len(event) == 4
        assert list(event.index) == all_indices

    def test_construction_with_invalid_sample_space(self):
        with pytest.raises(TypeError, match="must be a SampleSpace"):
            sa.Event("not a space", ["omega0"])

    def test_construction_with_duplicate_indices(self, sample_space):
        with pytest.raises(ValueError, match="must be unique"):
            sa.Event(sample_space, ["omega0", "omega0", "omega1"])

    def test_construction_with_invalid_index(self, sample_space):
        with pytest.raises(ValueError, match="not found in sample space"):
            sa.Event(sample_space, ["omega0", "invalid_index"])

    def test_construction_preserves_order(self, sample_space):
        event = sa.Event(sample_space, ["omega3", "omega1", "omega0"])
        assert list(event.index) == ["omega3", "omega1", "omega0"]


class TestConstructionFromProbabilitySpace:
    @pytest.fixture
    def prob_space(self):
        probs = {"omega0": 0.4, "omega1": 0.3, "omega2": 0.2, "omega3": 0.1}
        return sa.ProbabilitySpace(["omega0", "omega1", "omega2", "omega3"], probs)

    def test_event_has_probability(self, prob_space):
        event = sa.Event(prob_space, ["omega0", "omega2"])
        assert abs(event.probability - 0.6) < 1e-10

    def test_event_has_probability_measure(self, prob_space):
        event = sa.Event(prob_space, ["omega0", "omega2"])
        assert event.probability_measure is not None
        assert isinstance(event.probability_measure, sa.ProbabilityMeasure)

    def test_conditional_probability_normalized(self, prob_space):
        event = sa.Event(prob_space, ["omega0", "omega2"])
        assert abs(event.P("omega0") - 2 / 3) < 1e-10
        assert abs(event.P("omega2") - 1 / 3) < 1e-10

    def test_empty_event_has_zero_probability(self, prob_space):
        event = sa.Event(prob_space, [])
        assert event.probability == 0.0

    def test_empty_event_has_no_probability_measure(self, prob_space):
        event = sa.Event(prob_space, [])
        assert event.probability_measure is None

    def test_full_space_event_has_probability_one(self, prob_space):
        all_indices = ["omega0", "omega1", "omega2", "omega3"]
        event = sa.Event(prob_space, all_indices)
        assert abs(event.probability - 1.0) < 1e-10

    def test_single_outcome_event_probability(self, prob_space):
        event = sa.Event(prob_space, ["omega1"])
        assert event.probability == 0.3


class TestProperties:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    @pytest.fixture
    def prob_space(self):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        return sa.ProbabilitySpace(["omega0", "omega1", "omega2"], probs)

    def test_sample_space_property(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert event.sample_space == sample_space

    def test_probability_property_none_for_regular_space(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert event.probability is None

    def test_probability_property_set_for_prob_space(self, prob_space):
        event = sa.Event(prob_space, ["omega0", "omega2"])
        assert event.probability == 0.7

    def test_probability_measure_property_none_for_regular_space(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert event.probability_measure is None

    def test_probability_measure_property_set_for_prob_space(self, prob_space):
        event = sa.Event(prob_space, ["omega0", "omega2"])
        assert event.probability_measure is not None

    def test_index_property(self, sample_space):
        event = sa.Event(sample_space, ["omega1", "omega2"])
        assert list(event.index) == ["omega1", "omega2"]


class TestInheritedFromSampleSpace:
    @pytest.fixture
    def event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        return sa.Event(space, ["omega0", "omega2"])

    def test_len(self, event):
        assert len(event) == 2

    def test_iteration(self, event):
        indices = list(event)
        assert indices == ["omega0", "omega2"]

    def test_getitem_by_position(self, event):
        assert event[0] == "omega0"
        assert event[1] == "omega2"

    def test_getitem_negative_index(self, event):
        assert event[-1] == "omega2"
        assert event[-2] == "omega0"

    def test_sigma_algebra_property_inherited(self):
        indices = [f"omega{i}" for i in range(6)]
        sample_space = sa.SampleSpace(indices)

        atom_ids = dict(zip(sample_space, [0, 0, 1, 1, 2, 2]))
        sigma_algebra = sa.SigmaAlgebra(sample_space, atom_ids)
        sample_space.set_sigma_algebra(sigma_algebra)

        event = sa.Event(sample_space, ["omega0", "omega2"])

        expected_atom_ids = {
            "omega0": 0,
            "omega2": 1,
        }
        expected_sigma_algebra = sa.SigmaAlgebra(event, expected_atom_ids)

        assert event.sigma_algebra == expected_sigma_algebra


class TestEquality:
    @pytest.fixture
    def sample_space(self):
        return sa.SampleSpace(["omega0", "omega1", "omega2"])

    def test_equality_same_indices(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega0", "omega1"])
        assert event1 == event2

    def test_equality_different_indices(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega1", "omega2"])
        assert event1 != event2

    def test_equality_different_order(self, sample_space):
        event1 = sa.Event(sample_space, ["omega0", "omega1"])
        event2 = sa.Event(sample_space, ["omega1", "omega0"])
        # They have different index order, so should be different
        assert event1 != event2

    def test_equality_different_sample_spaces(self):
        space1 = sa.SampleSpace(["omega0", "omega1", "omega2"])
        space2 = sa.SampleSpace(["a", "b", "c"])
        event1 = sa.Event(space1, ["omega0", "omega1"])
        event2 = sa.Event(space2, ["a", "b"])
        assert event1 != event2

    def test_equality_with_non_event(self, sample_space):
        event = sa.Event(sample_space, ["omega0", "omega1"])
        assert event != ["omega0", "omega1"]
        assert event != "not an event"

    def test_equality_ignores_probability(self):
        space1 = sa.ProbabilitySpace(
            ["omega0", "omega1"], {"omega0": 0.6, "omega1": 0.4}
        )
        space2 = sa.ProbabilitySpace(
            ["omega0", "omega1"], {"omega0": 0.5, "omega1": 0.5}
        )
        event1 = sa.Event(space1, ["omega0"])
        event2 = sa.Event(space2, ["omega0"])
        assert event1 == event2


class TestEventCreationViaGetitem:
    def test_sample_space_getitem_creates_event(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        event = space[["omega0", "omega2"]]
        assert isinstance(event, sa.Event)
        assert event.probability is None

    def test_probability_space_getitem_creates_event_with_probability(self):
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_space = sa.ProbabilitySpace(["omega0", "omega1", "omega2"], probs)
        event = prob_space[["omega0", "omega2"]]
        assert isinstance(event, sa.Event)
        assert event.probability == 0.7


class TestEdgeCases:
    def test_event_with_zero_probability_outcomes(self):
        probs = {"omega0": 0.0, "omega1": 1.0}
        prob_space = sa.ProbabilitySpace(["omega0", "omega1"], probs)
        event = sa.Event(prob_space, ["omega0"])
        assert event.probability == 0.0
        assert event.probability_measure is None

    def test_event_from_uniform_distribution(self):
        prob_space = sa.ProbabilitySpace.uniform(["omega0", "omega1", "omega2"])
        event = sa.Event(prob_space, ["omega0", "omega1"])
        assert abs(event.probability - 2 / 3) < 1e-10

    def test_nested_event_operations(self):
        space = sa.SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        event1 = space[["omega0", "omega1", "omega2"]]
        event2 = event1[["omega0", "omega2"]]
        assert isinstance(event2, sa.Event)
        assert list(event2.index) == ["omega0", "omega2"]
