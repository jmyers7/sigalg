import pandas as pd
import pytest

import sigalg as sa


class TestConstructor:

    def test_basic_construction(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )
        assert fps.sample_space == sample_space
        assert fps.probability_measure == prob_measure
        pd.testing.assert_frame_equal(fps.feature_embedding.values, values)

    def test_construction_with_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        atom_ids = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )
        assert fps.sigma_algebra == sigma_algebra
        assert fps.sigma_algebra.num_atoms == 2

    def test_construction_with_default_probability_measure(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        assert abs(fps.P("s0") - 1 / 3) < 1e-10
        assert abs(fps.P("s1") - 1 / 3) < 1e-10
        assert abs(fps.P("s2") - 1 / 3) < 1e-10

    def test_construction_with_default_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space.values, columns=feature_index.values
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        assert fps.sigma_algebra.num_atoms == 2


class TestProperties:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_sample_space_property(self, fps):
        assert isinstance(fps.sample_space, sa.SampleSpace)
        assert len(fps.sample_space) == 3

    def test_sigma_algebra_property(self, fps):
        assert isinstance(fps.sigma_algebra, sa.SigmaAlgebra)

    def test_probability_measure_property(self, fps):
        assert isinstance(fps.probability_measure, sa.ProbabilityMeasure)

    def test_feature_embedding_property(self, fps):
        assert isinstance(fps.feature_embedding, sa.FeatureEmbedding)
        expected_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["s0", "s1", "s2"], name="sample"),
            columns=pd.Index(["X0", "X1"], name="feature"),
        )
        pd.testing.assert_frame_equal(fps.feature_embedding.values, expected_df)

    def test_probability_space_property(self, fps):
        assert isinstance(fps.probability_space, sa.ProbabilitySpace)


class TestProbabilityMethods:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_P_with_sample_index(self, fps):
        assert abs(fps.P("s0") - 0.2) < 1e-10
        assert abs(fps.P("s1") - 0.3) < 1e-10
        assert abs(fps.P("s2") - 0.5) < 1e-10

    def test_P_with_event(self, fps):
        event = sa.Event(sample_space=fps.sample_space, event_indices=["s0", "s1"])
        assert abs(fps.P(event) - 0.5) < 1e-10


class TestFeatureMethods:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_get_sample_features(self, fps):
        sf = fps.get_sample_features("s1")
        expected_series = pd.Series(
            [3, 4], index=pd.Index(["X0", "X1"], name="feature"), name="s1"
        )
        pd.testing.assert_series_equal(sf.values, expected_series)

    def test_get_sample_features_at(self, fps):
        sf = fps.get_sample_features_at[2]
        expected_series = pd.Series(
            [5, 6], index=pd.Index(["X0", "X1"], name="feature"), name="s2"
        )
        pd.testing.assert_series_equal(sf.values, expected_series)

    def test_get_event_features(self, fps):
        ef = fps.get_event_features(["s0", "s2"])
        expected_df = pd.DataFrame(
            [[1, 2], [5, 6]],
            index=pd.Index(["s0", "s2"], name="sample"),
            columns=pd.Index(["X0", "X1"], name="feature"),
        )
        pd.testing.assert_frame_equal(ef.values, expected_df)

    def test_get_feature_rv(self, fps):
        rv = fps.get_feature_rv("X0")
        assert rv.name == "X0"
        expected_series = pd.Series([1, 3, 5], index=["s0", "s1", "s2"], name="X0")
        expected_series.index.name = "sample"
        pd.testing.assert_series_equal(rv.values, expected_series)
        assert abs(rv.P(1) - 0.2) < 1e-10
        assert abs(rv.P(3) - 0.3) < 1e-10
        assert abs(rv.P(5) - 0.5) < 1e-10

    def test_get_sub_features(self, fps):
        sub = fps.get_sub_features(["X1"])
        expected_df = pd.DataFrame(
            [[2], [4], [6]],
            index=pd.Index(["s0", "s1", "s2"], name="sample"),
            columns=pd.Index(["X1"], name="feature"),
        )
        pd.testing.assert_frame_equal(sub.values, expected_df)


class TestEquality:
    def test_equal_fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        fps1 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )
        fps2 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )
        assert fps1 == fps2

    def test_not_equal_different_probabilities(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities1 = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure1 = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities1
        )
        probabilities2 = {"s0": 0.1, "s1": 0.4, "s2": 0.5}
        prob_measure2 = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities2
        )
        fps1 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure1,
        )
        fps2 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure2,
        )
        assert fps1 != fps2

    def test_not_equal_different_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        atom_ids1 = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra1 = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids1
        )
        atom_ids2 = {"s0": "A", "s1": "B", "s2": "C", "s3": "D"}
        sigma_algebra2 = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids2
        )
        fps1 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra1,
            probability_measure=prob_measure,
        )
        fps2 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra2,
            probability_measure=prob_measure,
        )
        assert fps1 != fps2

    def test_not_equal_different_sample_space(self):
        sample_space1 = sa.SampleSpace(["s0", "s1", "s2"])
        sample_space2 = sa.SampleSpace(["a", "b", "c"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
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
        feature_embedding1 = sa.FeatureEmbedding(values=values1, name="X")
        feature_embedding2 = sa.FeatureEmbedding(values=values2, name="X")
        probabilities1 = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        probabilities2 = {"a": 0.2, "b": 0.3, "c": 0.5}
        prob_measure1 = sa.ProbabilityMeasure(
            sample_space=sample_space1, probabilities=probabilities1
        )
        prob_measure2 = sa.ProbabilityMeasure(
            sample_space=sample_space2, probabilities=probabilities2
        )
        fps1 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space1,
            feature_embedding=feature_embedding1,
            probability_measure=prob_measure1,
        )
        fps2 = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space2,
            feature_embedding=feature_embedding2,
            probability_measure=prob_measure2,
        )
        assert fps1 != fps2

    def test_not_equal_different_type(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        assert fps != "not a featurized probability space"
        assert fps != 42
        assert fps is not None


class TestSetters:

    def test_set_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        atom_ids = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        fps.sigma_algebra = sigma_algebra
        assert fps.sigma_algebra == sigma_algebra

    def test_set_probability_measure(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding
        )
        probabilities = {"s0": 0.1, "s1": 0.4, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        fps.probability_measure = prob_measure
        assert fps.probability_measure == prob_measure

    def test_set_feature_embedding(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values1 = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space.values, columns=feature_index.values
        )
        feature_embedding1 = sa.FeatureEmbedding(values=values1, name="X")
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space, feature_embedding=feature_embedding1
        )
        feature_index2 = sa.FeatureIndex(["Y0", "Y1"])
        values2 = pd.DataFrame(
            [[5, 6], [7, 8]], index=sample_space.values, columns=feature_index2.values
        )
        feature_embedding2 = sa.FeatureEmbedding(values=values2, name="Y")
        fps.feature_embedding = feature_embedding2
        assert fps.feature_embedding == feature_embedding2


class TestSampleSpaceMethods:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            probability_measure=prob_measure,
        )

    def test_get_event(self, fps):
        event = fps.get_event(["s0", "s2"], name="E")
        assert isinstance(event, sa.Event)
        assert event.name == "E"
        assert set(event.values) == {"s0", "s2"}
        assert event.sample_space == fps.sample_space

    def test_get_event_with_default_name(self, fps):
        event = fps.get_event(["s1", "s3"])
        assert event.name == "A"
        assert set(event.values) == {"s1", "s3"}


class TestSigmaAlgebraMethods:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            index=sample_space.values,
            columns=feature_index.values,
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        atom_ids = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra = sa.SigmaAlgebra(
            sample_space=sample_space, sample_id_to_atom_id=atom_ids
        )
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        return sa.FeaturizedProbabilitySpace(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )

    def test_is_measurable_with_measurable_event(self, fps):
        event = sa.Event(sample_space=fps.sample_space, event_indices=["s0", "s1"])
        assert fps.is_measurable(event) is True

    def test_is_measurable_with_full_sample_space(self, fps):
        event = sa.Event(
            sample_space=fps.sample_space, event_indices=["s0", "s1", "s2", "s3"]
        )
        assert fps.is_measurable(event) is True

    def test_is_measurable_with_non_measurable_event(self, fps):
        event = sa.Event(sample_space=fps.sample_space, event_indices=["s0"])
        assert fps.is_measurable(event) is False

    def test_is_measurable_with_multiple_atoms(self, fps):
        event = sa.Event(
            sample_space=fps.sample_space, event_indices=["s0", "s1", "s2", "s3"]
        )
        assert fps.is_measurable(event) is True

    def test_get_atom_containing(self, fps):
        atom = fps.get_atom_containing("s0")
        assert isinstance(atom, sa.Event)
        assert set(atom.values) == {"s0", "s1"}

    def test_get_atom_containing_other_atom(self, fps):
        atom = fps.get_atom_containing("s3")
        assert set(atom.values) == {"s2", "s3"}

    def test_get_atom_containing_invalid_sample_id(self, fps):
        with pytest.raises(ValueError, match="not in sample space"):
            fps.get_atom_containing("invalid_id")


class TestValidation:

    def test_invalid_sample_space_type(self):
        sample_space1 = sa.SampleSpace(["s0", "s1"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space1.values, columns=feature_index.values
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        with pytest.raises(TypeError, match="sample_space must be a SampleSpace"):
            sa.FeaturizedProbabilitySpace(
                sample_space="invalid",
                feature_embedding=feature_embedding,
            )

    def test_invalid_feature_embedding_type(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        with pytest.raises(
            TypeError, match="feature_embedding must be a FeatureEmbedding"
        ):
            sa.FeaturizedProbabilitySpace(
                sample_space=sample_space,
                feature_embedding="invalid",
            )

    def test_invalid_sigma_algebra_type(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space.values, columns=feature_index.values
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        with pytest.raises(TypeError, match="sigma_algebra must be a SigmaAlgebra"):
            sa.FeaturizedProbabilitySpace(
                sample_space=sample_space,
                feature_embedding=feature_embedding,
                sigma_algebra="invalid",
            )

    def test_invalid_probability_measure_type(self):
        sample_space = sa.SampleSpace(["s0", "s1"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space.values, columns=feature_index.values
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        with pytest.raises(
            TypeError, match="probability_measure must be a ProbabilityMeasure"
        ):
            sa.FeaturizedProbabilitySpace(
                sample_space=sample_space,
                feature_embedding=feature_embedding,
                probability_measure="invalid",
            )

    def test_mismatched_feature_embedding_indices(self):
        sample_space1 = sa.SampleSpace(["s0", "s1", "s2"])
        sample_space2 = sa.SampleSpace(["a", "b"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space2.values, columns=feature_index.values
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        with pytest.raises(
            ValueError,
            match="feature_embedding must be defined on the given sample_space",
        ):
            sa.FeaturizedProbabilitySpace(
                sample_space=sample_space1,
                feature_embedding=feature_embedding,
            )

    def test_mismatched_sigma_algebra_sample_space(self):
        sample_space1 = sa.SampleSpace(["s0", "s1"])
        sample_space2 = sa.SampleSpace(["a", "b"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space1.values, columns=feature_index.values
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        sigma_algebra = sa.SigmaAlgebra.power_set(sample_space2)
        with pytest.raises(
            ValueError, match="sigma_algebra must be defined on the given sample_space"
        ):
            sa.FeaturizedProbabilitySpace(
                sample_space=sample_space1,
                feature_embedding=feature_embedding,
                sigma_algebra=sigma_algebra,
            )

    def test_mismatched_probability_measure_sample_space(self):
        sample_space1 = sa.SampleSpace(["s0", "s1"])
        sample_space2 = sa.SampleSpace(["a", "b"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space1.values, columns=feature_index.values
        )
        feature_embedding = sa.FeatureEmbedding(values=values, name="X")
        probabilities = {"a": 0.5, "b": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space2, probabilities=probabilities
        )
        with pytest.raises(
            ValueError,
            match="probability_measure must be defined on the given sample_space",
        ):
            sa.FeaturizedProbabilitySpace(
                sample_space=sample_space1,
                feature_embedding=feature_embedding,
                probability_measure=prob_measure,
            )

    def test_set_feature_embedding_mismatched_indices(self):
        sample_space1 = sa.SampleSpace(["s0", "s1", "s2"])
        sample_space2 = sa.SampleSpace(["a", "b"])
        feature_index = sa.FeatureIndex(["X0", "X1"])
        values1 = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=sample_space1.values,
            columns=feature_index.values,
        )
        values2 = pd.DataFrame(
            [[1, 2], [3, 4]], index=sample_space2.values, columns=feature_index.values
        )
        feature_embedding1 = sa.FeatureEmbedding(values=values1, name="X")
        feature_embedding2 = sa.FeatureEmbedding(values=values2, name="X")
        fps = sa.FeaturizedProbabilitySpace(
            sample_space=sample_space1,
            feature_embedding=feature_embedding1,
        )
        with pytest.raises(
            ValueError,
            match="feature_embedding must be defined on the given sample_space",
        ):
            fps.feature_embedding = feature_embedding2
