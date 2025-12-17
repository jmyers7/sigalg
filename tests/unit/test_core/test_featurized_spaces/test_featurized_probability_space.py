import pandas as pd
import pytest

from sigalg.core import (
    Event,
    FeatureIndex,
    FeaturizedProbabilitySpace,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)


class TestConstructor:

    def test_construction_from_all_parameters(self):
        """Test constructing a FeaturizedProbabilitySpace with all parameters provided."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        sample_id_to_atom_id = {"s0": "A", "s1": "B", "s2": "C"}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id, sample_space=sample_space
        )
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
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

    def test_construction_with_default_sigma_algebra_and_probability_measure(self):
        """Test constructing a FeaturizedProbabilitySpace with default sigma algebra and probability measure."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
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
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        atom_ids = {"s0": "A", "s1": "A", "s2": "B"}
        sigma_algebra = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        fps.sigma_algebra = sigma_algebra
        assert fps.sigma_algebra == sigma_algebra


class TestProbabilityMeasureProperty:

    def test_getter_and_setter(self):
        """Test the getter and setter for the probability_measure property."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        fps.probability_measure = prob_measure
        assert fps.probability_measure == prob_measure


class TestFeatureEmbeddingProperty:

    def test_getter_and_setter(self):
        """Test the getter and setter for the feature_embedding property."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values1 = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding1 = RandomVector.from_values(values=values1, name="X")
        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding1
        )
        values2 = pd.DataFrame(
            [[7, 8], [9, 10], [11, 12]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding2 = RandomVector.from_values(values=values2, name="Y")
        fps.feature_embedding = feature_embedding2
        assert fps.feature_embedding == feature_embedding2


class TestProbabilitySpaceProperty:

    def test_getter(self):
        """Test the getter for the probability_space property."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        sample_id_to_atom_id = {"s0": "A", "s1": "B", "s2": "C"}
        sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=sample_id_to_atom_id, sample_space=sample_space
        )
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
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


class TestGetComponents:

    def test_get_components_with_single_index(self):
        """Test the get_component method with a single index."""
        Omega = SampleSpace.generate_default(size=3)
        outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
        X = RandomVector(outputs=outputs, domain=Omega, name="X")
        X0 = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5}, domain=Omega, name="X0"
        )
        X1 = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6}, domain=Omega, name="X1"
        )
        assert X.get_components("X0") == X0
        assert X.get_components("X1") == X1

    def test_get_components_with_list(self):
        """Test the get_components method with a list of indices."""
        Omega = SampleSpace.generate_default(size=3)
        outputs = {"omega0": (1, 2, 3), "omega1": (4, 5, 6), "omega2": (7, 8, 9)}
        X = RandomVector(outputs=outputs, domain=Omega, name="X")
        expected_rv = RandomVector(
            outputs={"omega0": (1, 3), "omega1": (4, 6), "omega2": (7, 9)},
            domain=Omega,
            name="X_sub",
        )
        expected_rv.feature_index = FeatureIndex(["X0", "X2"])
        components = X.get_components(["X0", "X2"])
        assert components == expected_rv


class TestSampleSpaceMethods:

    @pytest.fixture
    def fps(self):
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_get_event(self, fps):
        """Test the get_event method with custom name."""
        event = fps.get_event(["s0", "s2"], name="E")
        assert isinstance(event, Event)
        assert event.name == "E"
        assert set(event.values) == {"s0", "s2"}
        assert event.sample_space == fps.sample_space

    def test_get_event_with_default_name(self, fps):
        """Test the get_event method with default name."""
        event = fps.get_event(["s1", "s3"])
        assert event.name == "A"
        assert set(event.values) == {"s1", "s3"}


class TestSigmaAlgebraMethods:

    @pytest.fixture
    def fps(self):
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        atom_ids = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )

    def test_is_measurable_with_measurable_event(self, fps):
        """Test is_measurable method with a measurable event."""
        event = Event(sample_space=fps.sample_space, event_indices=["s0", "s1"])
        assert fps.is_measurable(event) is True

    def test_is_measurable_with_full_sample_space(self, fps):
        """Test is_measurable method with the full sample space event."""
        event = Event(
            sample_space=fps.sample_space, event_indices=["s0", "s1", "s2", "s3"]
        )
        assert fps.is_measurable(event) is True

    def test_is_measurable_with_non_measurable_event(self, fps):
        """Test is_measurable method with a non-measurable event."""
        event = Event(sample_space=fps.sample_space, event_indices=["s0"])
        assert fps.is_measurable(event) is False

    def test_get_atom_containing(self, fps):
        """Test get_atom_containing method."""
        atom = fps.get_atom_containing("s0")
        assert isinstance(atom, Event)
        assert set(atom.values) == {"s0", "s1"}


class TestProbabilityMeasureMethods:

    @pytest.fixture
    def fps(self):
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_P_with_sample_index(self, fps):
        """Test the P method with a sample index."""
        assert abs(fps.P("s0") - 0.2) < 1e-10
        assert abs(fps.P("s1") - 0.3) < 1e-10
        assert abs(fps.P("s2") - 0.5) < 1e-10

    def test_P_with_event(self, fps):
        """Test the P method with an Event."""
        event = Event(sample_space=fps.sample_space, event_indices=["s0", "s1"])
        assert abs(fps.P(event) - 0.5) < 1e-10


class TestEquality:

    def test_equal_fps(self):
        """Test equality of two identical FeaturizedProbabilitySpace instances."""
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
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
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        probabilities1 = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure1 = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities1
        )
        probabilities2 = {"s0": 0.1, "s1": 0.4, "s2": 0.5}
        prob_measure2 = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities2
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
        sample_space = SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        probabilities = {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        atom_ids1 = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra1 = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids1
        )
        atom_ids2 = {"s0": "A", "s1": "B", "s2": "C", "s3": "D"}
        sigma_algebra2 = SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids2
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
        sample_space1 = SampleSpace(["s0", "s1", "s2"])
        sample_space2 = SampleSpace(["a", "b", "c"])
        feature_index = FeatureIndex(["X0", "X1"])
        values1 = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space1.values,
            columns=feature_index.values,
        )
        values2 = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space2.values,
            columns=feature_index.values,
        )
        feature_embedding1 = RandomVector.from_values(values=values1, name="X")
        feature_embedding2 = RandomVector.from_values(values=values2, name="X")
        probabilities1 = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        probabilities2 = {"a": 0.2, "b": 0.3, "c": 0.5}
        prob_measure1 = ProbabilityMeasure(
            sample_space=sample_space1, probabilities=probabilities1
        )
        prob_measure2 = ProbabilityMeasure(
            sample_space=sample_space2, probabilities=probabilities2
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
        sample_space = SampleSpace(["s0", "s1", "s2"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        fps = FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        assert fps != "not a featurized probability space"
        assert fps != 42
        assert fps is not None


class TestValidation:

    def test_invalid_sample_space_type(self):
        """Test invalid sample_space type raises TypeError."""
        sample_space1 = SampleSpace(["s0", "s1"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space1.values, columns=feature_index.values
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        with pytest.raises(TypeError, match="sample_space must be a SampleSpace"):
            FeaturizedProbabilitySpace(
                sample_space="invalid",
                feature_embedding=feature_embedding,
            )

    def test_invalid_feature_embedding_type(self):
        """Test invalid feature_embedding type raises TypeError."""
        sample_space = SampleSpace(["s0", "s1"])
        with pytest.raises(TypeError, match="feature_embedding must be a RandomVector"):
            FeaturizedProbabilitySpace(
                sample_space=sample_space,
                feature_embedding="invalid",
            )

    def test_invalid_sigma_algebra_type(self):
        """Test invalid sigma_algebra type raises TypeError."""
        sample_space = SampleSpace(["s0", "s1"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space.values, columns=feature_index.values
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            FeaturizedProbabilitySpace(
                sample_space=sample_space,
                feature_embedding=feature_embedding,
                sigma_algebra="invalid",
            )

    def test_invalid_probability_measure_type(self):
        """Test invalid probability_measure type raises TypeError."""
        sample_space = SampleSpace(["s0", "s1"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space.values, columns=feature_index.values
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        with pytest.raises(
            TypeError, match="probability_measure must be a ProbabilityMeasure"
        ):
            FeaturizedProbabilitySpace(
                sample_space=sample_space,
                feature_embedding=feature_embedding,
                probability_measure="invalid",
            )

    def test_mismatched_feature_embedding_indices(self):
        """Test mismatched feature_embedding and sample_space raises ValueError."""
        sample_space1 = SampleSpace(["s0", "s1", "s2"])
        sample_space2 = SampleSpace(["a", "b"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space2.values, columns=feature_index.values
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        with pytest.raises(
            ValueError,
            match="feature_embedding must be defined on the given sample_space",
        ):
            FeaturizedProbabilitySpace(
                sample_space=sample_space1,
                feature_embedding=feature_embedding,
            )

    def test_mismatched_sigma_algebra_sample_space(self):
        """Test mismatched sigma_algebra and sample_space raises ValueError."""
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space1.values, columns=feature_index.values
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        sigma_algebra = SigmaAlgebra.power_set(sample_space2)
        with pytest.raises(
            ValueError, match="sigma_algebra must be defined on the given sample_space"
        ):
            FeaturizedProbabilitySpace(
                sample_space=sample_space1,
                feature_embedding=feature_embedding,
                sigma_algebra=sigma_algebra,
            )

    def test_mismatched_probability_measure_sample_space(self):
        """Test mismatched probability_measure and sample_space raises ValueError."""
        sample_space1 = SampleSpace(["s0", "s1"])
        sample_space2 = SampleSpace(["a", "b"])
        feature_index = FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space1.values, columns=feature_index.values
        )
        feature_embedding = RandomVector.from_values(values=values, name="X")
        probabilities = {"a": 0.5, "b": 0.5}
        prob_measure = ProbabilityMeasure(
            sample_space=sample_space2, probabilities=probabilities
        )
        with pytest.raises(
            ValueError,
            match="probability_measure must be defined on the given sample_space",
        ):
            FeaturizedProbabilitySpace(
                sample_space=sample_space1,
                feature_embedding=feature_embedding,
                probability_measure=prob_measure,
            )
