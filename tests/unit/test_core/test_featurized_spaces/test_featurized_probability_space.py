import pytest

from sigalg.core import (
    Event,
    FeaturizedProbabilitySpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:

    @pytest.fixture
    def sample_space(self):
        return SampleSpace.generate_sequence(size=3, prefix="s", initial_index=0)

    @pytest.fixture
    def feature_embedding(self, sample_space):
        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        return RandomVector(domain=sample_space, name="X").from_dict(outputs)

    def test_construction_from_all_parameters(self, sample_space, feature_embedding):
        """Test constructing a FeaturizedProbabilitySpace with all parameters provided."""
        sample_id_to_atom_id = {"s_0": "A", "s_1": "B", "s_2": "C"}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=sample_id_to_atom_id
        )

        probabilities = {"s_0": 0.2, "s_1": 0.3, "s_2": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )

        assert fps.sample_space == sample_space
        assert fps.feature_embedding == feature_embedding
        assert fps.sigma_algebra == sigma_algebra
        assert fps.probability_measure == prob_measure

    def test_construction_with_default_sigma_algebra_and_probability_measure(
        self, sample_space, feature_embedding
    ):
        """Test constructing a FeaturizedProbabilitySpace with default sigma algebra and probability measure."""
        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space)
        expected_prob_measure = ProbabilityMeasure.uniform(sample_space)

        assert fps.sigma_algebra == expected_sigma_algebra
        assert fps.probability_measure == expected_prob_measure


class TestSigmaAlgebraProperty:

    def test_getter_and_setter(self):
        """Test the getter and setter for the sigma_algebra property."""
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}

        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs
        )

        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )

        atom_ids = {"s_0": "A", "s_1": "A", "s_2": "B"}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )
        fps.sigma_algebra = sigma_algebra

        assert fps.sigma_algebra == sigma_algebra


class TestProbabilityMeasureProperty:

    def test_getter_and_setter(self):
        """Test the getter and setter for the probability_measure property."""
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )

        probabilities = {"s_0": 0.2, "s_1": 0.3, "s_2": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )
        fps.probability_measure = prob_measure

        assert fps.probability_measure == prob_measure


class TestFeatureEmbeddingProperty:

    def test_getter_and_setter(self):
        """Test the getter and setter for the feature_embedding property."""
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs1 = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        feature_embedding1 = RandomVector(domain=sample_space, name="X").from_dict(
            outputs1
        )

        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding1
        )

        outputs2 = {"s_0": (7, 8), "s_1": (9, 10), "s_2": (11, 12)}
        feature_embedding2 = RandomVector(domain=sample_space, name="Y").from_dict(
            outputs2
        )
        fps.feature_embedding = feature_embedding2

        assert fps.feature_embedding == feature_embedding2


class TestProbabilitySpaceProperty:

    def test_getter(self):
        """Test the getter for the probability_space property."""
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        sample_id_to_atom_id = {"s_0": "A", "s_1": "B", "s_2": "C"}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=sample_id_to_atom_id
        )

        probabilities = {"s_0": 0.2, "s_1": 0.3, "s_2": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )
        expected_prob_space = ProbabilitySpace(
            sample_space=sample_space,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )

        assert fps.probability_space == expected_prob_space


class TestSampleSpaceMethods:

    @pytest.fixture
    def fps(self):
        sample_space = SampleSpace.generate_sequence(
            size=4, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6), "s_3": (7, 8)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        probabilities = {"s_0": 0.1, "s_1": 0.2, "s_2": 0.3, "s_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        return FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_get_event(self, fps):
        """Test the get_event method with custom name."""
        event = fps.get_event(["s_0", "s_2"], name="E")
        assert isinstance(event, Event)
        assert event.name == "E"
        assert set(event.data) == {"s_0", "s_2"}
        assert event.sample_space == fps.sample_space

    def test_get_event_with_default_name(self, fps):
        """Test the get_event method with default name."""
        event = fps.get_event(["s_1", "s_3"])
        assert event.name == "A"
        assert set(event.data) == {"s_1", "s_3"}


class TestSigmaAlgebraMethods:

    @pytest.fixture
    def fps(self):
        sample_space = SampleSpace.generate_sequence(
            size=4, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6), "s_3": (7, 8)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        atom_ids = {"s_0": "A", "s_1": "A", "s_2": "B", "s_3": "B"}
        sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids
        )

        probabilities = {"s_0": 0.1, "s_1": 0.2, "s_2": 0.3, "s_3": 0.4}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        return FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )

    def test_is_measurable_with_measurable_event(self, fps):
        """Test is_measurable method with a measurable event."""
        event = Event(sample_space=fps.sample_space).from_list(["s_0", "s_1"])
        assert fps.is_measurable(event) is True

    def test_is_measurable_with_full_sample_space(self, fps):
        """Test is_measurable method with the full sample space event."""
        event = Event(sample_space=fps.sample_space).from_list(
            ["s_0", "s_1", "s_2", "s_3"]
        )
        assert fps.is_measurable(event) is True

    def test_is_measurable_with_non_measurable_event(self, fps):
        """Test is_measurable method with a non-measurable event."""
        event = Event(sample_space=fps.sample_space).from_list(["s_0"])
        assert fps.is_measurable(event) is False

    def test_get_atom_containing(self, fps):
        """Test get_atom_containing method."""
        atom = fps.get_atom_containing("s_0")
        assert isinstance(atom, Event)
        assert set(atom.data) == {"s_0", "s_1"}


class TestProbabilityMeasureMethods:

    @pytest.fixture
    def fps(self):
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        probabilities = {"s_0": 0.2, "s_1": 0.3, "s_2": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        return FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_P_with_sample_index(self, fps):
        """Test the P method with a sample index."""
        assert abs(fps.P("s_0") - 0.2) < 1e-10
        assert abs(fps.P("s_1") - 0.3) < 1e-10
        assert abs(fps.P("s_2") - 0.5) < 1e-10

    def test_P_with_event(self, fps):
        """Test the P method with an Event."""
        event = Event(sample_space=fps.sample_space).from_list(["s_0", "s_1"])
        assert abs(fps.P(event) - 0.5) < 1e-10


class TestEquality:

    def test_equal_fps(self):
        """Test equality of two identical FeaturizedProbabilitySpace instances."""
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        probabilities = {"s_0": 0.2, "s_1": 0.3, "s_2": 0.5}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        fps1 = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

        fps2 = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

        assert fps1 == fps2

    def test_not_equal_different_probability_measures(self):
        """Test inequality of two FeaturizedProbabilitySpace instances with different probability measures."""
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        probabilities1 = {"s_0": 0.2, "s_1": 0.3, "s_2": 0.5}
        prob_measure1 = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities1
        )

        probabilities2 = {"s_0": 0.1, "s_1": 0.4, "s_2": 0.5}
        prob_measure2 = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities2
        )

        fps1 = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure1,
        )
        fps2 = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure2,
        )

        assert fps1 != fps2

    def test_not_equal_different_sigma_algebra(self):
        """Test inequality of two FeaturizedProbabilitySpace instances with different sigma algebras."""
        sample_space = SampleSpace.generate_sequence(
            size=4, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6), "s_3": (7, 8)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        probabilities = {"s_0": 0.25, "s_1": 0.25, "s_2": 0.25, "s_3": 0.25}
        prob_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities=probabilities
        )

        atom_ids1 = {"s_0": "A", "s_1": "A", "s_2": "B", "s_3": "B"}
        sigma_algebra1 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids1
        )

        atom_ids2 = {"s_0": "A", "s_1": "B", "s_2": "C", "s_3": "D"}
        sigma_algebra2 = SigmaAlgebra(sample_space=sample_space).from_dict(
            sample_id_to_atom_id=atom_ids2
        )

        fps1 = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra1,
            probability_measure=prob_measure,
        )
        fps2 = FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra2,
            probability_measure=prob_measure,
        )

        assert fps1 != fps2

    def test_not_equal_different_sample_space(self):
        """Test inequality of two FeaturizedProbabilitySpace instances with different sample spaces."""
        sample_space1 = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )
        sample_space2 = SampleSpace().from_list(["a", "b", "c"])

        outputs1 = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        outputs2 = {"a": (1, 2), "b": (3, 4), "c": (5, 6)}
        feature_embedding1 = RandomVector(domain=sample_space1, name="X").from_dict(
            outputs=outputs1
        )
        feature_embedding2 = RandomVector(domain=sample_space2, name="X").from_dict(
            outputs=outputs2
        )

        probabilities1 = {"s_0": 0.2, "s_1": 0.3, "s_2": 0.5}
        probabilities2 = {"a": 0.2, "b": 0.3, "c": 0.5}
        prob_measure1 = ProbabilityMeasure(sample_space=sample_space1).from_dict(
            probabilities=probabilities1
        )
        prob_measure2 = ProbabilityMeasure(sample_space=sample_space2).from_dict(
            probabilities=probabilities2
        )

        fps1 = FeaturizedProbabilitySpace(
            sample_space=sample_space1,
            feature_embedding=feature_embedding1,
            probability_measure=prob_measure1,
        )
        fps2 = FeaturizedProbabilitySpace(
            sample_space=sample_space2,
            feature_embedding=feature_embedding2,
            probability_measure=prob_measure2,
        )

        assert fps1 != fps2

    def test_not_equal_different_type(self):
        """Test inequality of FeaturizedProbabilitySpace with different types."""
        sample_space = SampleSpace.generate_sequence(
            size=3, prefix="s", initial_index=0
        )

        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        feature_embedding = RandomVector(domain=sample_space, name="X").from_dict(
            outputs=outputs
        )

        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )

        assert fps != "not a featurized probability space"
        assert fps != 42
        assert fps is not None
