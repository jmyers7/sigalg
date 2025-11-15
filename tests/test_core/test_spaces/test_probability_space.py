import pytest
import sigalg as sa


class TestConstruction:
    def test_construction_with_explicit_probabilities(self):
        indices = ["omega0", "omega1", "omega2"]
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_space = sa.ProbabilitySpace(indices, probabilities=probs)
        assert len(prob_space) == 3
        assert prob_space.probability("omega0") == 0.5
        assert prob_space.probability("omega1") == 0.3

    def test_construction_with_uniform_distribution(self):
        indices = ["omega0", "omega1", "omega2"]
        prob_space = sa.ProbabilitySpace(indices)
        assert abs(prob_space.probability("omega0") - 1 / 3) < 1e-10
        assert abs(prob_space.probability("omega1") - 1 / 3) < 1e-10
        assert abs(prob_space.probability("omega2") - 1 / 3) < 1e-10

    def test_construction_uniform_factory_method(self):
        indices = ["omega0", "omega1"]
        prob_space = sa.ProbabilitySpace.uniform(indices)
        assert prob_space.probability("omega0") == 0.5
        assert prob_space.probability("omega1") == 0.5

    def test_construction_inherits_sample_space_validation(self):
        with pytest.raises(ValueError, match="must be unique"):
            sa.ProbabilitySpace(["omega0", "omega0"], probabilities={"omega0": 1.0})

    def test_construction_with_invalid_probabilities(self):
        indices = ["omega0", "omega1"]
        probs = {"omega0": 0.7, "omega1": 0.2}
        with pytest.raises(ValueError, match="must sum to 1"):
            sa.ProbabilitySpace(indices, probabilities=probs)


class TestProperties:
    @pytest.fixture
    def prob_space(self):
        indices = ["omega0", "omega1", "omega2"]
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        return sa.ProbabilitySpace(indices, probabilities=probs)

    def test_probability_measure_property(self, prob_space):
        P = prob_space.probability_measure
        assert isinstance(P, sa.ProbabilityMeasure)
        assert P.sample_space == prob_space

    def test_index_property_inherited(self, prob_space):
        assert list(prob_space.index) == ["omega0", "omega1", "omega2"]

    def test_sigma_algebra_property_inherited(self, prob_space):
        sigma_algebra = prob_space.sigma_algebra
        atom_ids = dict(zip(prob_space.index, range(len(prob_space))))
        expected_sigma_algebra = sa.SigmaAlgebra(
            sample_space=prob_space, atom_ids=atom_ids
        )
        assert sigma_algebra == expected_sigma_algebra


class TestProbabilityMethod:
    @pytest.fixture
    def prob_space(self):
        indices = ["omega0", "omega1", "omega2"]
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        return sa.ProbabilitySpace(indices, probabilities=probs)

    def test_probability_of_single_outcome(self, prob_space):
        assert prob_space.probability("omega0") == 0.5
        assert prob_space.probability("omega1") == 0.3
        assert prob_space.probability("omega2") == 0.2

    def test_probability_of_event(self, prob_space):
        event = prob_space[["omega0", "omega2"]]
        prob = prob_space.probability(event)
        assert prob == 0.7

    def test_probability_of_empty_event(self, prob_space):
        event = prob_space[[]]
        prob = prob_space.probability(event)
        assert prob == 0.0

    def test_probability_of_full_space(self, prob_space):
        event = prob_space[["omega0", "omega1", "omega2"]]
        prob = prob_space.probability(event)
        assert abs(prob - 1.0) < 1e-10


class TestSetProbabilityMeasure:
    @pytest.fixture
    def prob_space(self):
        return sa.ProbabilitySpace(["omega0", "omega1"], probabilities=None)

    def test_set_valid_probability_measure(self, prob_space):
        new_probs = {"omega0": 0.7, "omega1": 0.3}
        new_P = sa.ProbabilityMeasure(prob_space, new_probs)
        prob_space.set_probability_measure(new_P)
        assert prob_space.probability("omega0") == 0.7
        assert prob_space.probability("omega1") == 0.3

    def test_set_probability_measure_invalid_type(self, prob_space):
        with pytest.raises(TypeError, match="must be a ProbabilityMeasure"):
            prob_space.set_probability_measure({"omega0": 0.5, "omega1": 0.5})

    def test_set_probability_measure_from_different_space(self, prob_space):
        other_space = sa.ProbabilitySpace(["a", "b"])
        other_P = sa.ProbabilityMeasure(other_space, {"a": 0.5, "b": 0.5})
        with pytest.raises(ValueError, match="must be defined on this sample space"):
            prob_space.set_probability_measure(other_P)


class TestEventCreation:
    @pytest.fixture
    def prob_space(self):
        indices = ["omega0", "omega1", "omega2"]
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        return sa.ProbabilitySpace(indices, probabilities=probs)

    def test_getitem_with_list_returns_event_with_probability(self, prob_space):
        event = prob_space[["omega0", "omega2"]]
        assert isinstance(event, sa.Event)
        assert event.probability == 0.7

    def test_getitem_with_empty_list(self, prob_space):
        event = prob_space[[]]
        assert isinstance(event, sa.Event)
        assert event.probability == 0.0

    def test_getitem_with_single_element_list(self, prob_space):
        event = prob_space[["omega1"]]
        assert isinstance(event, sa.Event)
        assert event.probability == 0.3

    def test_getitem_with_all_elements(self, prob_space):
        event = prob_space[["omega0", "omega1", "omega2"]]
        assert isinstance(event, sa.Event)
        assert abs(event.probability - 1.0) < 1e-10

    def test_getitem_single_index_returns_string(self, prob_space):
        result = prob_space[0]
        assert result == "omega0"
        assert not isinstance(result, sa.Event)

    def test_event_has_correct_sample_space_reference(self, prob_space):
        event = prob_space[["omega0", "omega1"]]
        assert event.sample_space == prob_space


class TestInheritedMethods:
    @pytest.fixture
    def prob_space(self):
        return sa.ProbabilitySpace(["omega0", "omega1", "omega2"])

    def test_len_inherited(self, prob_space):
        assert len(prob_space) == 3

    def test_iteration_inherited(self, prob_space):
        labels = list(prob_space)
        assert labels == ["omega0", "omega1", "omega2"]

    def test_getitem_by_position_inherited(self, prob_space):
        assert prob_space[0] == "omega0"
        assert prob_space[1] == "omega1"

    def test_contains_inherited(self, prob_space):
        assert 0 in range(len(prob_space))
        assert 5 not in range(len(prob_space))


class TestEquality:
    def test_equality_same_probabilities(self):
        indices = ["omega0", "omega1"]
        probs = {"omega0": 0.6, "omega1": 0.4}
        space1 = sa.ProbabilitySpace(indices, probabilities=probs)
        space2 = sa.ProbabilitySpace(indices, probabilities=probs)
        assert space1 == space2

    def test_equality_different_probabilities(self):
        indices = ["omega0", "omega1"]
        space1 = sa.ProbabilitySpace(
            indices, probabilities={"omega0": 0.6, "omega1": 0.4}
        )
        space2 = sa.ProbabilitySpace(
            indices, probabilities={"omega0": 0.5, "omega1": 0.5}
        )
        assert space1 != space2

    def test_equality_different_indices(self):
        space1 = sa.ProbabilitySpace(["omega0", "omega1"])
        space2 = sa.ProbabilitySpace(["omega0", "omega2"])
        assert space1 != space2

    def test_equality_with_regular_sample_space(self):
        prob_space = sa.ProbabilitySpace(["omega0", "omega1"])
        regular_space = sa.SampleSpace(["omega0", "omega1"])
        assert prob_space != regular_space

    def test_equality_with_non_space(self):
        prob_space = sa.ProbabilitySpace(["omega0", "omega1"])
        assert prob_space != ["omega0", "omega1"]
        assert prob_space != "not a space"


class TestRepresentation:
    def test_repr_with_explicit_probabilities(self):
        probs = {"omega0": 0.7, "omega1": 0.3}
        prob_space = sa.ProbabilitySpace(["omega0", "omega1"], probabilities=probs)
        repr_str = repr(prob_space)
        assert "ProbabilitySpace" in repr_str
        assert "omega0" in repr_str
        assert "0.7" in repr_str

    def test_repr_with_uniform_distribution(self):
        prob_space = sa.ProbabilitySpace(["omega0", "omega1"])
        repr_str = repr(prob_space)
        assert "ProbabilitySpace" in repr_str


class TestHashing:
    def test_probability_space_is_hashable(self):
        space1 = sa.ProbabilitySpace(["omega0", "omega1"])
        space2 = sa.ProbabilitySpace(["omega0", "omega1"])
        spaces = {space1, space2}
        assert len(spaces) >= 1

    def test_can_use_as_dict_key(self):
        prob_space = sa.ProbabilitySpace(["omega0", "omega1"])
        d = {prob_space: "value"}
        assert d[prob_space] == "value"


class TestEdgeCases:
    def test_single_outcome_space(self):
        prob_space = sa.ProbabilitySpace(["omega0"], probabilities={"omega0": 1.0})
        assert prob_space.probability("omega0") == 1.0
        event = prob_space[["omega0"]]
        assert event.probability == 1.0

    def test_empty_space_not_allowed(self):
        # Empty sample spaces should be caught by parent validation
        with pytest.raises(ValueError):
            sa.ProbabilitySpace([])

    def test_zero_probability_outcome(self):
        probs = {"omega0": 0.0, "omega1": 1.0}
        prob_space = sa.ProbabilitySpace(["omega0", "omega1"], probabilities=probs)
        assert prob_space.probability("omega0") == 0.0
        event = prob_space[["omega0"]]
        assert event.probability == 0.0
