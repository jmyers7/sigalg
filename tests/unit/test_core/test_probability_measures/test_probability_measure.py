import pandas as pd
import pytest

from sigalg.core import Event, ProbabilityMeasure, SampleSpace, SigmaAlgebra


class TestConstructor:

    @pytest.mark.parametrize(
        "probabilities, sample_space_indices, name",
        [
            pytest.param(
                {"omega0": 0.5, "omega1": 0.5},
                ["omega0", "omega1"],
                None,
                id="default_name",
            ),
            pytest.param(
                {"a": 0.2, "b": 0.3, "c": 0.5}, ["a", "b", "c"], "Q", id="custom_name"
            ),
        ],
    )
    def test_constructor(self, probabilities, sample_space_indices, name):
        """Test the constructor of ProbabilityMeasure."""
        sample_space = SampleSpace(indices=sample_space_indices)
        if name is None:
            prob_measure = ProbabilityMeasure(
                probabilities=probabilities, sample_space=sample_space
            )
            name = "P"
        else:
            prob_measure = ProbabilityMeasure(
                probabilities=probabilities, sample_space=sample_space, name=name
            )

        assert prob_measure.sample_space == sample_space
        assert prob_measure.probabilities == probabilities
        assert prob_measure.name == name

    @pytest.mark.parametrize(
        "probabilities",
        [
            pytest.param(
                {"omega0": 0.6, "omega1": 0.5},
                id="probabilities_not_summing_to_1",
            ),
            pytest.param(
                {"omega0": -0.1, "omega1": 1.1},
                id="negative_and_greater_than_one_probability",
            ),
            pytest.param(
                {"omega0": "not a number", "omega1": 1.0},
                id="non_numeric_probability",
            ),
        ],
    )
    def test_invalid_input_raises(self, probabilities):
        """Test that invalid inputs raise appropriate errors."""
        sample_space = SampleSpace(indices=["omega0", "omega1"])
        with pytest.raises((ValueError, TypeError)):
            ProbabilityMeasure(probabilities=probabilities, sample_space=sample_space)


class TestFromPandas:

    @pytest.mark.parametrize(
        "series_data, name",
        [
            pytest.param(
                {"omega0": 0.4, "omega1": 0.6},
                "Q",
                id="custom_name",
            ),
            pytest.param(
                {"omega0": 0.7, "omega1": 0.3},
                None,
                id="default_name",
            ),
        ],
    )
    def test_from_pandas(self, series_data, name):
        """Test the from_pandas class method of ProbabilityMeasure."""
        data = pd.Series(series_data, name="dummy_name")
        if name is None:
            prob_measure = ProbabilityMeasure.from_pandas(data=data)
            name = "P"
        else:
            prob_measure = ProbabilityMeasure.from_pandas(data=data, name=name)

        pd.testing.assert_series_equal(prob_measure.data, data)
        assert prob_measure.name == name


class TestEquality:

    @pytest.mark.parametrize(
        "given, other",
        [
            pytest.param(
                ProbabilityMeasure(
                    probabilities={"omega0": 0.5, "omega1": 0.5},
                    sample_space=SampleSpace(["omega0", "omega1"]),
                ),
                ProbabilityMeasure(
                    probabilities={"a": 0.5, "b": 0.5},
                    sample_space=SampleSpace(["a", "b"]),
                ),
                id="different_sample_spaces",
            ),
            pytest.param(
                ProbabilityMeasure(
                    probabilities={"omega0": 0.6, "omega1": 0.4},
                    sample_space=SampleSpace(["omega0", "omega1"]),
                ),
                ProbabilityMeasure(
                    probabilities={"omega0": 0.5, "omega1": 0.5},
                    sample_space=SampleSpace(["omega0", "omega1"]),
                ),
                id="different_probabilities",
            ),
        ],
    )
    def test_non_equality(self, given, other):
        """Test the __eq__ method for inequality."""
        assert given != other

    @pytest.mark.parametrize(
        "given, other",
        [
            pytest.param(
                ProbabilityMeasure(
                    probabilities={"omega0": 0.5, "omega1": 0.5},
                    sample_space=SampleSpace(["omega0", "omega1"]),
                ),
                ProbabilityMeasure(
                    probabilities={"omega0": 0.5, "omega1": 0.5},
                    sample_space=SampleSpace(["omega0", "omega1"]),
                ),
                id="same_probabilities_and_sample_space",
            ),
            pytest.param(
                ProbabilityMeasure(
                    probabilities={"a": 0.2, "b": 0.8},
                    sample_space=SampleSpace(["a", "b"], name="S"),
                    name="Q",
                ),
                ProbabilityMeasure(
                    probabilities={"a": 0.2, "b": 0.8},
                    sample_space=SampleSpace(["a", "b"], name="T"),
                    name="R",
                ),
                id="same_components_different_names",
            ),
        ],
    )
    def test_equality(self, given, other):
        """Test the __eq__ method for equality."""
        assert given == other


class TestCallMethod:

    @pytest.mark.parametrize(
        "indices, type, expected",
        [
            pytest.param(
                ["omega0", "omega2"],
                "list",
                0.4,
                id="list_of_indices",
            ),
            pytest.param(
                "omega1",
                "hashable",
                0.2,
                id="single_hashable_index",
            ),
            pytest.param(
                ["omega1", "omega3"],
                "event",
                0.6,
                id="event_instance",
            ),
        ],
    )
    def test_call(self, indices, type, expected):
        """Test the __call__ method of ProbabilityMeasure."""
        sample_space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {"omega0": 0.1, "omega1": 0.2, "omega2": 0.3, "omega3": 0.4}
        prob_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        if type in ["hashable", "list"]:
            result = prob_measure(indices)
        if type == "event":
            event = Event(indices, sample_space=prob_measure.sample_space)
            result = prob_measure(event)

        assert abs(result - expected) < 1e-9


def test_uniform():
    """Test the uniform probability measure constructor."""
    sample_space = SampleSpace(indices=["a", "b", "c", "d"])
    prob_measure = ProbabilityMeasure.uniform(sample_space=sample_space, name="U")

    expected_probabilities = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert prob_measure.probabilities == expected_probabilities
    assert prob_measure.name == "U"


class TestConditionalProbability:

    @pytest.mark.parametrize(
        "event_A_indices, event_B_indices",
        [
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega1", "omega2"],
                id="subset_of_conditioning_event",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega1", "omega2"],
                id="non_trivial_overlap",
            ),
            pytest.param(
                ["omega2", "omega3"],
                ["omega0", "omega1"],
                id="no_overlap",
            ),
        ],
    )
    def test_conditional_probability(self, event_A_indices, event_B_indices):
        """Test the conditional_probability method of ProbabilityMeasure."""
        sample_space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {"omega0": 0.2, "omega1": 0.3, "omega2": 0.4, "omega3": 0.1}
        prob_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        event_A = Event(event_A_indices, sample_space=sample_space)
        event_B = Event(event_B_indices, sample_space=sample_space)
        result = prob_measure.conditional_probability(event_A, event_B)
        expected = prob_measure(event_A & event_B) / prob_measure(event_B)

        assert abs(result - expected) < 1e-9

    def test_conditioning_on_impossible_event(self):
        """Test that conditional_probability raises ValueError when P(B) = 0."""
        sample_space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {"omega0": 0.5, "omega1": 0.5, "omega2": 0.0, "omega3": 0.0}
        prob_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        event_A = Event(["omega0", "omega1"], sample_space=sample_space)
        event_B = Event(["omega2", "omega3"], sample_space=sample_space)

        with pytest.raises(ValueError):
            prob_measure.conditional_probability(event_A, event_B)


class TestAreIndependent:

    @pytest.mark.parametrize(
        "event_A_indices, event_B_indices, expected",
        [
            pytest.param(
                ["omega0", "omega1"],
                ["omega0", "omega2"],
                True,
                id="independent_events",
            ),
            pytest.param(
                ["omega0", "omega1"],
                ["omega2", "omega3"],
                False,
                id="dependent_events",
            ),
        ],
    )
    def test_are_independent(self, event_A_indices, event_B_indices, expected):
        """Test the are_independent method of ProbabilityMeasure."""
        sample_space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {
            "omega0": 0.25**2,
            "omega1": 0.25 * 0.75,
            "omega2": 0.75 * 0.25,
            "omega3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        event_A = Event(event_A_indices, sample_space=sample_space)
        event_B = Event(event_B_indices, sample_space=sample_space)
        result = prob_measure.are_independent(event1=event_A, event2=event_B)

        assert result == expected

    @pytest.mark.parametrize(
        "atom_ids1, atom_ids2, expected",
        [
            pytest.param(
                {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1},
                {"omega0": 0, "omega1": 1, "omega2": 0, "omega3": 1},
                True,
                id="independent_sigma_algebras",
            ),
            pytest.param(
                {"omega0": 0, "omega1": 1, "omega2": 1, "omega3": 1},
                {"omega0": 0, "omega1": 0, "omega2": 1, "omega3": 1},
                False,
                id="dependent_sigma_algebras",
            ),
        ],
    )
    def test_are_independent_sigma_algebras(self, atom_ids1, atom_ids2, expected):
        """Test the are_independent method for sigma algebras."""
        sample_space = SampleSpace(["omega0", "omega1", "omega2", "omega3"])
        probabilities = {
            "omega0": 0.25**2,
            "omega1": 0.25 * 0.75,
            "omega2": 0.75 * 0.25,
            "omega3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        sigma1 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids1, sample_space=sample_space, name="sigma1"
        )
        sigma2 = SigmaAlgebra(
            sample_id_to_atom_id=atom_ids2, sample_space=sample_space, name="sigma2"
        )
        result = prob_measure.are_independent(algebra1=sigma1, algebra2=sigma2)

        assert result == expected

    def test_are_independent_raises_for_both_events_and_algebras(self):
        """Test that are_independent raises ValueError when both events and algebras are provided."""
        sample_space = SampleSpace(["omega0", "omega1"])
        probabilities = {"omega0": 0.5, "omega1": 0.5}
        prob_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )
        event_A = Event(["omega0"], sample_space=sample_space)
        event_B = Event(["omega1"], sample_space=sample_space)
        sigma1 = SigmaAlgebra(
            sample_id_to_atom_id={"omega0": 0, "omega1": 1}, sample_space=sample_space
        )
        sigma2 = SigmaAlgebra(
            sample_id_to_atom_id={"omega0": 0, "omega1": 1}, sample_space=sample_space
        )

        with pytest.raises(ValueError, match="Cannot provide both"):
            prob_measure.are_independent(
                event1=event_A, event2=event_B, algebra1=sigma1, algebra2=sigma2
            )

    def test_are_independent_raises_for_neither_events_nor_algebras(self):
        """Test that are_independent raises ValueError when neither events nor algebras are provided."""
        sample_space = SampleSpace(["omega0", "omega1"])
        probabilities = {"omega0": 0.5, "omega1": 0.5}
        prob_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=sample_space
        )

        with pytest.raises(ValueError, match="Must provide either"):
            prob_measure.are_independent()
