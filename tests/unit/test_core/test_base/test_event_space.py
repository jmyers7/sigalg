import pytest

from sigalg.core import (
    Event,
    EventSpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2"])

    def test_constructor(self, sample_space):
        custom_sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            {"omega0": 0, "omega1": 1, "omega2": 1},
        )
        event_space1 = EventSpace(
            sample_space=sample_space, sigma_algebra=custom_sigma_algebra
        )
        event_space2 = EventSpace(sample_space=sample_space)
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)

        assert event_space1.sample_space == sample_space
        assert event_space1.sigma_algebra == custom_sigma_algebra
        assert event_space2.sample_space == sample_space
        assert event_space2.sigma_algebra == expected_sigma_algebra

    @pytest.mark.parametrize(
        "invalid_sigma_algebra",
        [
            pytest.param("not a sigma algebra", id="wrong_type"),
            pytest.param(
                SigmaAlgebra(
                    sample_space=SampleSpace().from_list(["omega0", "omega1"])
                ).from_dict({"omega0": 0, "omega1": 0}),
                id="mismatched_sample_space",
            ),
        ],
    )
    def test_invalid_input_raises(self, sample_space, invalid_sigma_algebra):
        """Test that invalid inputs to the constructor raise appropriate exceptions."""
        with pytest.raises((TypeError, ValueError)):
            EventSpace(sample_space=sample_space, sigma_algebra=invalid_sigma_algebra)


def test_set_sigma_algebra():
    """Test that the sigma-algebra setter correctly updates the sigma-algebra."""
    sample_space = SampleSpace().from_list(["omega0", "omega1", "omega2"])
    event_space = EventSpace(sample_space=sample_space)
    new_sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
        {"omega0": 0, "omega1": 1, "omega2": 1},
    )
    event_space.sigma_algebra = new_sigma_algebra
    assert event_space.sigma_algebra == new_sigma_algebra


class TestGetEventMethod:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])

    @pytest.fixture
    def event_space(self, sample_space):
        return EventSpace(sample_space=sample_space)

    @pytest.mark.parametrize(
        "indices,name",
        [
            pytest.param(["omega1", "omega3"], "TestEvent", id="subset_indices"),
            pytest.param(["omega0"], "SingleEvent", id="single_index"),
            pytest.param([], "EmptyEvent", id="empty_indices"),
            pytest.param(
                ["omega0", "omega1", "omega2", "omega3"],
                "FullEvent",
                id="all_indices",
            ),
        ],
    )
    def test_get_event_returns_correct_event(
        self, event_space, sample_space, indices, name
    ):
        """Test that get_event method returns the correct Event instance."""
        event = event_space.get_event(indices, name=name)
        expected_event = Event(sample_space=sample_space, name=name).from_list(
            indices,
        )

        assert event == expected_event


class TestEquality:

    @pytest.mark.parametrize(
        "given,other",
        [
            pytest.param(
                EventSpace(
                    sample_space=SampleSpace().from_list(["omega0", "omega1"]),
                ),
                EventSpace(
                    sample_space=SampleSpace().from_list(
                        ["omega0", "omega1", "omega2"]
                    ),
                ),
                id="different_sample_spaces",
            ),
            pytest.param(
                EventSpace(
                    sample_space=SampleSpace().from_list(
                        ["omega0", "omega1", "omega2"]
                    ),
                    sigma_algebra=SigmaAlgebra.power_set(
                        SampleSpace().from_list(["omega0", "omega1", "omega2"])
                    ),
                ),
                EventSpace(
                    sample_space=SampleSpace().from_list(
                        ["omega0", "omega1", "omega2"]
                    ),
                    sigma_algebra=SigmaAlgebra(
                        sample_space=SampleSpace().from_list(
                            ["omega0", "omega1", "omega2"]
                        )
                    ).from_dict({"omega0": 0, "omega1": 0, "omega2": 1}),
                ),
                id="different_sigma_algebras",
            ),
            pytest.param(
                EventSpace(sample_space=SampleSpace().from_list(["omega0", "omega1"])),
                "not an event space",
                id="wrong_type",
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
                EventSpace(
                    sample_space=SampleSpace().from_list(
                        ["omega0", "omega1", "omega2"]
                    ),
                    sigma_algebra=SigmaAlgebra.power_set(
                        SampleSpace().from_list(["omega0", "omega1", "omega2"])
                    ),
                ),
                EventSpace(
                    sample_space=SampleSpace().from_list(
                        ["omega0", "omega1", "omega2"]
                    ),
                    sigma_algebra=SigmaAlgebra.power_set(
                        SampleSpace().from_list(["omega0", "omega1", "omega2"])
                    ),
                ),
                id="same_parameters",
            ),
        ],
    )
    def test_equality(self, given, other):
        """Test the __eq__ method for equality."""
        assert given == other


def test_make_probability_space():
    """Test that make_probability_space creates a ProbabilitySpace correctly."""
    sample_space = SampleSpace().from_list(["s0", "s1", "s2"])
    sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
        {"s0": 0, "s1": 0, "s2": 1}
    )
    event_space = EventSpace(sample_space=sample_space, sigma_algebra=sigma_algebra)
    custom_prob_measure = ProbabilityMeasure(
        sample_space=SampleSpace().from_list(["s0", "s1", "s2"])
    ).from_dict(
        {"s0": 0.5, "s1": 0.3, "s2": 0.2},
    )
    uniform_prob_measure = ProbabilityMeasure.uniform(sample_space=sample_space)

    prob_space1 = event_space.make_probability_space(
        probability_measure=custom_prob_measure
    )
    prob_space2 = event_space.make_probability_space()

    assert isinstance(prob_space1, ProbabilitySpace)
    assert prob_space1.sample_space == sample_space
    assert prob_space1.sigma_algebra == sigma_algebra
    assert prob_space1.probability_measure == custom_prob_measure

    assert isinstance(prob_space2, ProbabilitySpace)
    assert prob_space2.sample_space == sample_space
    assert prob_space2.sigma_algebra == sigma_algebra
    assert prob_space2.probability_measure == uniform_prob_measure
