import pandas as pd
import pytest

from sigalg.core import Event, ProbabilityMeasure, SampleSpace, SigmaAlgebra
from sigalg.core.random_objects.random_vector import RandomVector


class TestConstructor:

    def test_constructor_default_name(self):
        """Test the constructor of ProbabilityMeasure with default name."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.5, "omega_1": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        assert prob_measure.sample_space == sample_space
        assert prob_measure.probabilities == probabilities
        assert prob_measure.name == "P"

    def test_constructor_custom_name(self):
        """Test the constructor of ProbabilityMeasure with custom name."""
        sample_space = SampleSpace().from_list(["a", "b", "c"])
        probabilities = {"a": 0.2, "b": 0.3, "c": 0.5}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, name="Q"
        ).from_dict(probabilities=probabilities)

        assert prob_measure.sample_space == sample_space
        assert prob_measure.probabilities == probabilities
        assert prob_measure.name == "Q"

    def test_invalid_input_probabilities_not_summing_to_1(self):
        """Test that probabilities not summing to 1 raises ValueError."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.6, "omega_1": 0.5}
        with pytest.raises((ValueError, TypeError)):
            ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities=probabilities
            )

    def test_invalid_input_negative_and_greater_than_one_probability(self):
        """Test that negative and greater than one probabilities raise ValueError."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": -0.1, "omega_1": 1.1}
        with pytest.raises((ValueError, TypeError)):
            ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities=probabilities
            )

    def test_invalid_input_non_numeric_probability(self):
        """Test that non-numeric probabilities raise TypeError."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": "not a number", "omega_1": 1.0}
        with pytest.raises((ValueError, TypeError)):
            ProbabilityMeasure(sample_space=sample_space).from_dict(
                probabilities=probabilities
            )


class TestFromPandas:

    def test_from_pandas_custom_name(self):
        """Test the from_pandas instance method of ProbabilityMeasure with custom name."""
        series_data = {"omega_0": 0.4, "omega_1": 0.6}
        data = pd.Series(series_data, name="dummy_name")
        prob_measure = ProbabilityMeasure(name="Q").from_pandas(data=data)

        data.name = "probability"
        pd.testing.assert_series_equal(prob_measure.data, data)
        assert prob_measure.name == "Q"

    def test_from_pandas_default_name(self):
        """Test the from_pandas instance method of ProbabilityMeasure with default name."""
        series_data = {"omega_0": 0.7, "omega_1": 0.3}
        data = pd.Series(series_data, name="dummy_name")
        prob_measure = ProbabilityMeasure().from_pandas(data=data)

        data.name = "probability"
        pd.testing.assert_series_equal(prob_measure.data, data)
        assert prob_measure.name == "P"


class TestEquality:

    def test_non_equality_different_sample_spaces(self):
        """Test the __eq__ method for inequality with different sample spaces."""
        sample_space1 = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        sample_space2 = SampleSpace().from_list(["a", "b"])
        given = ProbabilityMeasure(sample_space=sample_space1).from_dict(
            probabilities={"omega_0": 0.5, "omega_1": 0.5}
        )
        other = ProbabilityMeasure(sample_space=sample_space2).from_dict(
            probabilities={"a": 0.5, "b": 0.5}
        )
        assert given != other

    def test_non_equality_different_probabilities(self):
        """Test the __eq__ method for inequality with different probabilities."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        given = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities={"omega_0": 0.6, "omega_1": 0.4}
        )
        other = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities={"omega_0": 0.5, "omega_1": 0.5}
        )
        assert given != other

    def test_equality_same_probabilities_and_sample_space(self):
        """Test the __eq__ method for equality with same probabilities and sample space."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        given = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities={"omega_0": 0.5, "omega_1": 0.5}
        )
        other = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities={"omega_0": 0.5, "omega_1": 0.5}
        )
        assert given == other

    def test_equality_same_components_different_names(self):
        """Test the __eq__ method for equality with same components but different names."""
        sample_space_s = SampleSpace(name="S").from_list(["a", "b"])
        sample_space_t = SampleSpace(name="T").from_list(["a", "b"])
        given = ProbabilityMeasure(sample_space=sample_space_s, name="Q").from_dict(
            probabilities={"a": 0.2, "b": 0.8}
        )
        other = ProbabilityMeasure(sample_space=sample_space_t, name="R").from_dict(
            probabilities={"a": 0.2, "b": 0.8}
        )
        assert given == other


class TestFromFeatures:

    def test_from_features(self):
        """Test adding a ProbabilityMeasure to the domain of a RandomVector."""
        domain = SampleSpace.generate_sequence(size=4)
        outputs = {
            "omega_0": (0, 0),
            "omega_1": (0, 1),
            "omega_2": (1, 0),
            "omega_3": (1, 1),
        }
        X = RandomVector(domain=domain, name="X").from_dict(outputs=outputs)

        def pmf(feature_vector):
            v0, v1 = feature_vector
            return 0.75**v0 * 0.25 ** (1 - v0) * 0.6**v1 * 0.4 ** (1 - v1)

        probability_measure = ProbabilityMeasure.from_features(rv=X, pmf=pmf)

        expected_probability_measure = ProbabilityMeasure(
            sample_space=domain
        ).from_dict(
            probabilities={
                "omega_0": 0.25 * 0.4,
                "omega_1": 0.25 * 0.6,
                "omega_2": 0.75 * 0.4,
                "omega_3": 0.75 * 0.6,
            }
        )

        assert probability_measure.sample_space == domain
        assert probability_measure == expected_probability_measure


class TestCallMethod:

    def test_call_list_of_indices(self):
        """Test the __call__ method of ProbabilityMeasure with list of indices."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.1, "omega_1": 0.2, "omega_2": 0.3, "omega_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        indices = ["omega_0", "omega_2"]
        result = prob_measure(indices)
        expected = 0.4
        assert abs(result - expected) < 1e-9

    def test_call_single_hashable_index(self):
        """Test the __call__ method of ProbabilityMeasure with single hashable index."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.1, "omega_1": 0.2, "omega_2": 0.3, "omega_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        result = prob_measure("omega_1")
        expected = 0.2
        assert abs(result - expected) < 1e-9

    def test_call_event_instance(self):
        """Test the __call__ method of ProbabilityMeasure with event instance."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.1, "omega_1": 0.2, "omega_2": 0.3, "omega_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event = Event(sample_space=prob_measure.sample_space).from_list(
            ["omega_1", "omega_3"]
        )
        result = prob_measure(event)
        expected = 0.6
        assert abs(result - expected) < 1e-9


def test_uniform():
    """Test the uniform probability measure constructor."""
    sample_space = SampleSpace().from_list(["a", "b", "c", "d"])
    prob_measure = ProbabilityMeasure.uniform(sample_space=sample_space, name="U")

    expected_probabilities = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    assert prob_measure.probabilities == expected_probabilities
    assert prob_measure.name == "U"


class TestConditionalProbability:

    def test_conditional_probability_subset_of_conditioning_event(self):
        """Test conditional_probability method when event A is subset of B."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.2, "omega_1": 0.3, "omega_2": 0.4, "omega_3": 0.1}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event_A = Event(sample_space=sample_space).from_list(["omega_0", "omega_1"])
        event_B = Event(sample_space=sample_space).from_list(
            ["omega_0", "omega_1", "omega_2"]
        )
        result = prob_measure.conditional_probability(event_A, event_B)
        expected = prob_measure(event_A & event_B) / prob_measure(event_B)
        assert abs(result - expected) < 1e-9

    def test_conditional_probability_non_trivial_overlap(self):
        """Test conditional_probability method with non-trivial overlap."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.2, "omega_1": 0.3, "omega_2": 0.4, "omega_3": 0.1}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event_A = Event(sample_space=sample_space).from_list(["omega_0", "omega_1"])
        event_B = Event(sample_space=sample_space).from_list(["omega_1", "omega_2"])
        result = prob_measure.conditional_probability(event_A, event_B)
        expected = prob_measure(event_A & event_B) / prob_measure(event_B)
        assert abs(result - expected) < 1e-9

    def test_conditional_probability_no_overlap(self):
        """Test conditional_probability method with no overlap."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.2, "omega_1": 0.3, "omega_2": 0.4, "omega_3": 0.1}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event_A = Event(sample_space=sample_space).from_list(["omega_2", "omega_3"])
        event_B = Event(sample_space=sample_space).from_list(["omega_0", "omega_1"])
        result = prob_measure.conditional_probability(event_A, event_B)
        expected = prob_measure(event_A & event_B) / prob_measure(event_B)
        assert abs(result - expected) < 1e-9

    def test_conditioning_on_impossible_event(self):
        """Test that conditional_probability raises ValueError when P(B) = 0."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.5, "omega_1": 0.5, "omega_2": 0.0, "omega_3": 0.0}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event_A = Event(sample_space=sample_space).from_list(["omega_0", "omega_1"])
        event_B = Event(sample_space=sample_space).from_list(["omega_2", "omega_3"])

        with pytest.raises(ValueError):
            prob_measure.conditional_probability(event_A, event_B)


class TestAreIndependent:

    def test_are_independent_events_independent(self):
        """Test the are_independent method with independent events."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {
            "omega_0": 0.25**2,
            "omega_1": 0.25 * 0.75,
            "omega_2": 0.75 * 0.25,
            "omega_3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event_A = Event(sample_space=sample_space).from_list(["omega_0", "omega_1"])
        event_B = Event(sample_space=sample_space).from_list(["omega_0", "omega_2"])
        result = prob_measure.are_independent(event1=event_A, event2=event_B)
        assert result

    def test_are_independent_events_dependent(self):
        """Test the are_independent method with dependent events."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {
            "omega_0": 0.25**2,
            "omega_1": 0.25 * 0.75,
            "omega_2": 0.75 * 0.25,
            "omega_3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event_A = Event(sample_space=sample_space).from_list(["omega_0", "omega_1"])
        event_B = Event(sample_space=sample_space).from_list(["omega_2", "omega_3"])
        result = prob_measure.are_independent(event1=event_A, event2=event_B)
        assert not result

    def test_are_independent_sigma_algebras_independent(self):
        """Test the are_independent method for independent sigma algebras."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {
            "omega_0": 0.25**2,
            "omega_1": 0.25 * 0.75,
            "omega_2": 0.75 * 0.25,
            "omega_3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        atom_ids1 = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        atom_ids2 = {"omega_0": 0, "omega_1": 1, "omega_2": 0, "omega_3": 1}
        sigma1 = SigmaAlgebra(sample_space=sample_space, name="sigma1").from_dict(
            sample_id_to_atom_id=atom_ids1
        )
        sigma2 = SigmaAlgebra(sample_space=sample_space, name="sigma2").from_dict(
            sample_id_to_atom_id=atom_ids2
        )
        result = prob_measure.are_independent(algebra1=sigma1, algebra2=sigma2)
        assert result

    def test_are_independent_sigma_algebras_dependent(self):
        """Test the are_independent method for dependent sigma algebras."""
        sample_space = SampleSpace.generate_sequence(
            size=4, initial_index=0, prefix="omega"
        )
        probabilities = {
            "omega_0": 0.25**2,
            "omega_1": 0.25 * 0.75,
            "omega_2": 0.75 * 0.25,
            "omega_3": 0.75**2,
        }
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        atom_ids1 = {"omega_0": 0, "omega_1": 1, "omega_2": 1, "omega_3": 1}
        atom_ids2 = {"omega_0": 0, "omega_1": 0, "omega_2": 1, "omega_3": 1}
        sigma1 = SigmaAlgebra(sample_space=sample_space, name="sigma1").from_dict(
            sample_id_to_atom_id=atom_ids1
        )
        sigma2 = SigmaAlgebra(sample_space=sample_space, name="sigma2").from_dict(
            sample_id_to_atom_id=atom_ids2
        )
        result = prob_measure.are_independent(algebra1=sigma1, algebra2=sigma2)
        assert not result

    def test_are_independent_raises_for_both_events_and_algebras(self):
        """Test that are_independent raises ValueError when both events and algebras are provided."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.5, "omega_1": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        event_A = Event(sample_space=sample_space).from_list(["omega_0"])
        event_B = Event(sample_space=sample_space).from_list(["omega_1"])
        sigma1 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 1}
        )
        sigma2 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id={"omega_0": 0, "omega_1": 1}
        )

        with pytest.raises(ValueError, match="Cannot provide both"):
            prob_measure.are_independent(
                event1=event_A, event2=event_B, algebra1=sigma1, algebra2=sigma2
            )

    def test_are_independent_raises_for_neither_events_nor_algebras(self):
        """Test that are_independent raises ValueError when neither events nor algebras are provided."""
        sample_space = SampleSpace.generate_sequence(
            size=2, initial_index=0, prefix="omega"
        )
        probabilities = {"omega_0": 0.5, "omega_1": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        with pytest.raises(ValueError, match="Must provide either"):
            prob_measure.are_independent()
