import pytest

from sigalg.core import (
    Event,
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2"])

    @pytest.mark.parametrize(
        "probabilities, atom_ids",
        [
            pytest.param(
                {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2},
                {"omega0": 0, "omega1": 0, "omega2": 1},
                id="all_parameters",
            ),
            pytest.param(
                None,
                None,
                id="defaults_only",
            ),
            pytest.param(
                {"omega0": 1.0 / 3, "omega1": 1.0 / 3, "omega2": 1.0 / 3},
                None,
                id="custom_probabilities_only",
            ),
            pytest.param(
                None,
                {"omega0": 0, "omega1": 0, "omega2": 1},
                id="custom_sigma_algebra_only",
            ),
        ],
    )
    def test_constructor(self, sample_space, probabilities, atom_ids):
        """Test constructing ProbabilitySpace with various combinations of parameters."""
        if probabilities is not None:
            prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities=probabilities
            )
        else:
            prob_measure = None

        if atom_ids is not None:
            sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
                sample_id_to_atom_id=atom_ids
            )
        else:
            sigma_algebra = None

        if prob_measure and sigma_algebra:
            prob_space = ProbabilitySpace(sample_space, sigma_algebra, prob_measure)
        elif prob_measure:
            prob_space = ProbabilitySpace(
                sample_space, probability_measure=prob_measure
            )
        elif sigma_algebra:
            prob_space = ProbabilitySpace(sample_space, sigma_algebra=sigma_algebra)
        else:
            prob_space = ProbabilitySpace(sample_space)

        expected_sigma_algebra = (
            sigma_algebra if sigma_algebra else SigmaAlgebra.power_set(sample_space)
        )
        expected_prob_measure = (
            prob_measure if prob_measure else ProbabilityMeasure.uniform(sample_space)
        )

        assert prob_space.sample_space == sample_space
        assert prob_space.sigma_algebra == expected_sigma_algebra
        assert prob_space.probability_measure == expected_prob_measure

    @pytest.mark.parametrize(
        "sigma_algebra,prob_measure",
        [
            pytest.param(
                "not_a_sigma_algebra", "not_a_prob_measure", id="both_invalid_types"
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_space=SampleSpace().from_list(["omega0", "omega1"])
                ).from_dict({"omega0": 0, "omega1": 0}),
                "not_a_prob_measure",
                id="invalid_prob_measure_type",
            ),
            pytest.param(
                "not_a_sigma_algebra",
                ProbabilityMeasure(
                    sample_space=SampleSpace().from_list(["omega0", "omega1"])
                ).from_dict(
                    probabilities={"omega0": 0.5, "omega1": 0.5},
                ),
                id="invalid_sigma_algebra_type",
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_space=SampleSpace().from_list(["omega0", "omega1", "omega2"])
                ).from_dict(
                    {"omega0": 0, "omega1": 0, "omega2": 1},
                ),
                ProbabilityMeasure(
                    sample_space=SampleSpace().from_list(["omega0", "omega1"])
                ).from_dict(probabilities={"omega0": 0.5, "omega1": 0.5}),
                id="mismatched_prob_measure_sample_space",
            ),
            pytest.param(
                SigmaAlgebra(
                    sample_space=SampleSpace().from_list(["omega0", "omega1"])
                ).from_dict(sample_id_to_atom_id={"omega0": 0, "omega1": 0}),
                ProbabilityMeasure(
                    sample_space=SampleSpace().from_list(["omega0", "omega1", "omega2"])
                ).from_dict(
                    probabilities={"omega0": 0.5, "omega1": 0.5, "omega2": 0.0}
                ),
                id="mismatched_sigma_algebra_sample_space",
            ),
        ],
    )
    def test_invalid_input_raises(self, sample_space, sigma_algebra, prob_measure):
        """Test that invalid inputs raise error."""
        with pytest.raises((TypeError, ValueError)):
            ProbabilitySpace(
                sample_space,
                sigma_algebra=sigma_algebra,
                probability_measure=prob_measure,
            )


class TestSetters:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2"])

    @pytest.fixture
    def prob_space(self, sample_space):
        return ProbabilitySpace(sample_space)

    def test_set_sigma_algebra_updates_sigma_algebra(self, sample_space, prob_space):
        """Test setting a new sigma_algebra updates the ProbabilitySpace's sigma_algebra."""
        atom_ids = {"omega0": 0, "omega1": 0, "omega2": 1}
        new_sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        prob_space.sigma_algebra = new_sigma_algebra

        assert prob_space.sigma_algebra == new_sigma_algebra

    def test_set_probability_measure_updates_probability_measure(
        self, sample_space, prob_space
    ):
        """Test setting a new probability_measure updates the ProbabilitySpace's probability_measure."""
        probabilities = {"omega0": 0.4, "omega1": 0.4, "omega2": 0.2}
        new_prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space.probability_measure = new_prob_measure

        assert prob_space.probability_measure == new_prob_measure


def test_get_event():
    """Test that get_event returns an Event instance with correct indices."""
    sample_space = SampleSpace().from_list(["omega0", "omega1", "omega2"])
    prob_space = ProbabilitySpace(sample_space)
    event = prob_space.get_event(["omega0", "omega2"])

    assert isinstance(event, Event)
    assert list(event.data) == ["omega0", "omega2"]


class TestPMethod:

    @pytest.mark.parametrize(
        "input,expected_probability",
        [
            pytest.param("omega0", 0.1, id="single_omega0"),
            pytest.param("omega1", 0.2, id="single_omega1"),
            pytest.param("omega3", 0.4, id="single_omega3"),
            pytest.param(["omega0", "omega1"], 0.3, id="event_two_outcomes"),
            pytest.param([], 0.0, id="empty_event"),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                1.0,
                id="full_space_event",
            ),
        ],
    )
    def test_P_method(self, input, expected_probability):
        """Test the P method for various inputs."""
        sample_space = SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {
            "omega0": 0.1,
            "omega1": 0.2,
            "omega2": 0.3,
            "omega3": 0.4,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities,
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        if isinstance(input, list):
            input = Event(sample_space=sample_space).from_list(indices=input)
        result = prob_space.P(input)

        assert abs(result - expected_probability) < 1e-10


class TestConditionalProbability:

    @pytest.mark.parametrize(
        "indices_a,indices_b,relationship",
        [
            pytest.param(["omega0"], ["omega0", "omega1"], "intersection", id="basic"),
            pytest.param(
                ["omega0", "omega1"], ["omega2", "omega3"], "disjoint", id="disjoint"
            ),
            pytest.param(
                ["omega0"], ["omega0", "omega1", "omega2"], "subset", id="A_subset_of_B"
            ),
        ],
    )
    def test_conditional_probability(self, indices_a, indices_b, relationship):
        """Test conditional probability calculation for various event relationships."""
        sample_space = SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)
        A = prob_space.get_event(indices_a, name="A")
        B = prob_space.get_event(indices_b, name="B")
        cond_prob = prob_space.conditional_probability(A, B)
        expected_prob = prob_space.P(A & B) / prob_space.P(B)

        assert abs(cond_prob - expected_prob) < 1e-10


class TestAreIndependent:

    @pytest.fixture
    def prob_space(self):
        sample_space = SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {
            "omega0": 0.25**2,
            "omega1": 0.75 * 0.25,
            "omega2": 0.25 * 0.75,
            "omega3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        return ProbabilitySpace(sample_space, probability_measure=prob_measure)

    def test_independent_events(self, prob_space):
        """Test that two independent events are correctly identified."""
        A = prob_space.get_event(["omega0", "omega2"], name="A")
        B = prob_space.get_event(["omega0", "omega1"], name="B")
        assert prob_space.are_independent(A, B)

    def test_dependent_events(self, prob_space):
        """Test that two dependent events are correctly identified."""
        A = prob_space.get_event(["omega0", "omega2"], name="A")
        B = prob_space.get_event(["omega1", "omega3"], name="B")
        assert not prob_space.are_independent(A, B)


class TestGetEventAsProbabilitySpace:

    @pytest.mark.parametrize(
        "indices,expected_atom_ids,expected_probabilities",
        [
            pytest.param(
                ["omega0", "omega2"],
                {"omega0": 0, "omega2": 1},
                {"omega0": 0.1 / 0.4, "omega2": 0.3 / 0.4},
                id="proper_event",
            ),
            pytest.param(
                ["omega1"],
                {"omega1": 0},
                {"omega1": 1.0},
                id="single_outcome_event",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1},
                {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4},
                id="full_space",
            ),
        ],
    )
    def test_get_event_as_probability_space(
        self, indices, expected_atom_ids, expected_probabilities
    ):
        """Test getting a conditional ProbabilitySpace given an event."""
        sample_space = SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        sample_id_to_atom_id = {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id
        )
        prob_space = ProbabilitySpace(sample_space, sigma_algebra, prob_measure)
        conditional_space = prob_space.get_event_as_probability_space(indices)
        expected_sigma_algebra = SigmaAlgebra(
            sample_space=conditional_space.sample_space
        ).from_dict(expected_atom_ids)
        expected_prob_measure = ProbabilityMeasure(
            sample_space=conditional_space.sample_space
        ).from_dict(expected_probabilities)

        assert isinstance(conditional_space, ProbabilitySpace)
        assert set(conditional_space.sample_space.data) == set(indices)
        assert conditional_space.sigma_algebra == expected_sigma_algebra
        assert conditional_space.probability_measure == expected_prob_measure

    def test_conditioned_on_event_with_zero_probability_raises(self):
        """Test that conditioning on an event with zero probability raises an error."""
        sample_space = SampleSpace().from_list(["omega0", "omega1"])
        probabilities = {"omega0": 0.0, "omega1": 1.0}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        prob_space = ProbabilitySpace(sample_space, probability_measure=prob_measure)

        with pytest.raises(ValueError):
            prob_space.get_event_as_probability_space(["omega0"])


class TestEquality:

    @pytest.mark.parametrize(
        "given, other",
        [
            pytest.param(
                ProbabilitySpace(
                    SampleSpace().from_list(["omega0", "omega1"]),
                    probability_measure=ProbabilityMeasure(
                        sample_space=SampleSpace().from_list(["omega0", "omega1"])
                    ).from_dict(
                        probabilities={"omega0": 0.5, "omega1": 0.5},
                    ),
                ),
                ProbabilitySpace(
                    SampleSpace().from_list(["omega0", "omega1"]),
                    probability_measure=ProbabilityMeasure(
                        sample_space=SampleSpace().from_list(["omega0", "omega1"])
                    ).from_dict(probabilities={"omega0": 0.7, "omega1": 0.3}),
                ),
                id="different_probability_measures",
            ),
            pytest.param(
                ProbabilitySpace(
                    SampleSpace().from_list(["omega0", "omega1", "omega2"]),
                    sigma_algebra=SigmaAlgebra(
                        sample_space=SampleSpace().from_list(
                            ["omega0", "omega1", "omega2"]
                        )
                    ).from_dict(
                        sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
                    ),
                ),
                ProbabilitySpace(
                    SampleSpace().from_list(["omega0", "omega1", "omega2"]),
                    sigma_algebra=SigmaAlgebra(
                        sample_space=SampleSpace().from_list(
                            ["omega0", "omega1", "omega2"]
                        )
                    ).from_dict(
                        sample_id_to_atom_id={"omega0": 0, "omega1": 1, "omega2": 1},
                    ),
                ),
                id="different_sigma_algebras",
            ),
            pytest.param(
                ProbabilitySpace(SampleSpace().from_list(["omega0", "omega1"])),
                ProbabilitySpace(SampleSpace().from_list(["a", "b"])),
                id="different_sample_spaces",
            ),
            pytest.param(
                ProbabilitySpace(SampleSpace().from_list(["omega0", "omega1"])),
                "not a probability space",
                id="wrong_type_string",
            ),
            pytest.param(
                ProbabilitySpace(SampleSpace().from_list(["omega0", "omega1"])),
                123,
                id="wrong_type_int",
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
                ProbabilitySpace(
                    SampleSpace().from_list(["omega0", "omega1"]),
                    SigmaAlgebra(
                        sample_space=SampleSpace().from_list(["omega0", "omega1"])
                    ).from_dict(
                        sample_id_to_atom_id={"omega0": 0, "omega1": 1},
                    ),
                    ProbabilityMeasure(
                        sample_space=SampleSpace().from_list(["omega0", "omega1"])
                    ).from_dict(
                        probabilities={"omega0": 0.5, "omega1": 0.5},
                    ),
                ),
                ProbabilitySpace(
                    SampleSpace().from_list(["omega0", "omega1"]),
                    SigmaAlgebra(
                        sample_space=SampleSpace().from_list(["omega0", "omega1"])
                    ).from_dict(
                        sample_id_to_atom_id={"omega0": 0, "omega1": 1},
                    ),
                    ProbabilityMeasure(
                        sample_space=SampleSpace().from_list(["omega0", "omega1"])
                    ).from_dict(
                        probabilities={"omega0": 0.5, "omega1": 0.5},
                    ),
                ),
                id="same_components",
            ),
        ],
    )
    def test_equality(self, given, other):
        """Test the __eq__ method for equality."""
        assert given == other


class TestProbabilityAxioms:

    @pytest.fixture
    def prob_space(self):
        space = SampleSpace().from_list(["omega0", "omega1", "omega2"])
        probs = {"omega0": 0.5, "omega1": 0.3, "omega2": 0.2}
        prob_measure = ProbabilityMeasure(sample_space=space).from_dict(
            probabilities=probs
        )
        return ProbabilitySpace(space, probability_measure=prob_measure)

    def test_axiom_non_negativity(self, prob_space):
        """Test that probabilities are non-negative."""
        for idx in prob_space.sample_space.data:
            assert prob_space.P(idx) >= 0

    def test_axiom_normalization(self, prob_space):
        """Test that the probability of the entire sample space is 1."""
        full_event = Event(sample_space=prob_space.sample_space).from_list(
            indices=list(prob_space.sample_space.data),
        )
        assert abs(prob_space.P(full_event) - 1.0) < 1e-10

    def test_axiom_additivity_disjoint_events(self, prob_space):
        """Test that the probability of the union of disjoint events equals the sum of their probabilities."""
        event_A = Event(sample_space=prob_space.sample_space).from_list(
            indices=["omega0"]
        )
        event_B = Event(sample_space=prob_space.sample_space).from_list(
            indices=["omega1"]
        )
        union = event_A | event_B
        prob_union = prob_space.P(union)
        prob_sum = prob_space.P(event_A) + prob_space.P(event_B)
        assert abs(prob_union - prob_sum) < 1e-10

    def test_complement_rule(self, prob_space):
        """Test that the probability of an event and its complement sum to 1."""
        event = Event(sample_space=prob_space.sample_space).from_list(
            indices=["omega0", "omega1"]
        )
        complement = ~event
        assert abs(prob_space.P(event) + prob_space.P(complement) - 1.0) < 1e-10
