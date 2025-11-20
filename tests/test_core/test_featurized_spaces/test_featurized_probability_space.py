import pandas as pd
import pytest

import sigalg as sa


class TestConstruction:
    def test_construction_with_all_components(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probability_measure=prob_measure
        )
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        assert fps.probability_space == prob_space
        assert fps.sample_space == sample_space
        assert fps.sigma_algebra == prob_space.sigma_algebra
        assert fps.probability_measure == prob_measure
        assert fps.featurized_sample_space == featurized_sample_space

    def test_construction_with_default_probability_space(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        sample_space = featurized_sample_space.sample_space
        prob_space = sa.ProbabilitySpace(sample_space=sample_space)
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        assert abs(fps.P("omega0") - 1 / 3) < 1e-10
        assert abs(fps.P("omega1") - 1 / 3) < 1e-10
        assert abs(fps.P("omega2") - 1 / 3) < 1e-10

    def test_construction_with_custom_sigma_algebra(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        atom_ids = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra = sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids)
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )
        features = [[1, 2], [3, 4], [5, 6], [7, 8]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        assert fps.sigma_algebra.num_atoms == 2


class TestProperties:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probability_measure=prob_measure
        )
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        return sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )

    def test_probability_space_property(self, fps):
        assert isinstance(fps.probability_space, sa.ProbabilitySpace)

    def test_sample_space_property(self, fps):
        assert isinstance(fps.sample_space, sa.SampleSpace)
        assert len(fps.sample_space) == 3

    def test_sigma_algebra_property(self, fps):
        assert isinstance(fps.sigma_algebra, sa.SigmaAlgebra)

    def test_probability_measure_property(self, fps):
        assert isinstance(fps.probability_measure, sa.ProbabilityMeasure)

    def test_featurized_sample_space_property(self, fps):
        assert isinstance(fps.featurized_sample_space, sa.FeaturizedSampleSpace)


class TestProbabilitySpaceMethods:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probability_measure=prob_measure
        )
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        return sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )

    def test_P_with_sample_index(self, fps):
        assert abs(fps.P("s0") - 0.2) < 1e-10
        assert abs(fps.P("s1") - 0.3) < 1e-10
        assert abs(fps.P("s2") - 0.5) < 1e-10

    def test_P_with_event(self, fps):
        event = sa.Event(sample_space=fps.sample_space, event_indices=["s0", "s1"])
        assert abs(fps.P(event) - 0.5) < 1e-10

    def test_sample(self, fps):
        samples = fps.sample(size=100, random_state=42)
        assert len(samples) == 100
        assert all(s in ["s0", "s1", "s2"] for s in samples)

    def test_get_event_as_probability_space(self, fps):
        event_prob_space = fps.get_event_as_probability_space(["s0", "s1"])
        assert isinstance(event_prob_space, sa.ProbabilitySpace)
        assert len(event_prob_space.sample_space) == 2
        # Conditional probabilities should sum to 1
        total_prob = event_prob_space.P("s0") + event_prob_space.P("s1")
        assert abs(total_prob - 1.0) < 1e-10


class TestSigmaAlgebraMethods:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        atom_ids = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra = sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids)
        probabilities = {"s0": 0.1, "s1": 0.2, "s2": 0.3, "s3": 0.4}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space,
            sigma_algebra=sigma_algebra,
            probability_measure=prob_measure,
        )
        features = [[1, 2], [3, 4], [5, 6], [7, 8]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        return sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )

    def test_to_events(self, fps):
        events = fps.to_events()
        assert isinstance(events, dict)
        assert len(events) == 2
        assert "A" in events
        assert "B" in events

    def test_is_measurable(self, fps):
        measurable_event = sa.Event(
            sample_space=fps.sample_space, event_indices=["s0", "s1"]
        )
        assert fps.is_measurable(measurable_event) is True

        non_measurable_event = sa.Event(
            sample_space=fps.sample_space, event_indices=["s0"]
        )
        assert fps.is_measurable(non_measurable_event) is False

    def test_get_atom_containing(self, fps):
        atom = fps.get_atom_containing("s0")
        assert isinstance(atom, sa.Event)
        assert set(atom.values) == {"s0", "s1"}


class TestFeaturizedSampleSpaceMethods:
    @pytest.fixture
    def fps(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probability_measure=prob_measure
        )
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        return sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )

    def test_get_sample_features(self, fps):
        sample_features = fps.get_sample_features("s1")
        expected_features = pd.Series(data=[3, 4], index=["X0", "X1"], name="s1")
        pd.testing.assert_series_equal(sample_features.features, expected_features)

    def test_get_event_features(self, fps):
        event_features = fps.get_event_features(["s0", "s2"])
        expected_features = pd.DataFrame(
            data=[[1, 2], [5, 6]], index=["s0", "s2"], columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(event_features.features, expected_features)

    def test_get_sample_features_at(self, fps):
        sample_features = fps.get_sample_features_at[1]
        expected_features = pd.Series(data=[3, 4], index=["X0", "X1"], name="s1")
        pd.testing.assert_series_equal(sample_features.features, expected_features)

    def test_get_event_features_at(self, fps):
        event_features = fps.get_event_features_at[[0, 2]]
        expected_features = pd.DataFrame(
            data=[[1, 2], [5, 6]], index=["s0", "s2"], columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(event_features.features, expected_features)

    def test_get_feature_rv(self, fps):
        feature_rv = fps.get_feature_rv("X1")
        expected_values = pd.Series(data=[2, 4, 6], index=["s0", "s1", "s2"], name="X1")
        pd.testing.assert_series_equal(feature_rv.values, expected_values)

    def test_get_sub_features(self, fps):
        # Add a third feature for this test
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        features = [[1, 2, 7], [3, 4, 8], [5, 6, 9]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        prob_space = sa.ProbabilitySpace(sample_space=sample_space)
        fps_extended = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )

        sub_features = fps_extended.get_sub_features(["X0", "X2"])
        expected_features = pd.DataFrame(
            data=[[1, 7], [3, 8], [5, 9]],
            index=["s0", "s1", "s2"],
            columns=["X0", "X2"],
        )
        pd.testing.assert_frame_equal(sub_features.features, expected_features)


class TestIntegration:
    def test_probabilities_and_features_together(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probability_measure=prob_measure
        )
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        assert abs(fps.P("s1") - 0.3) < 1e-10
        sample_features = fps.get_sample_features("s1")
        expected_features = pd.Series(data=[3, 4], index=["X0", "X1"], name="s1")
        pd.testing.assert_series_equal(sample_features.features, expected_features)

    def test_measurability_and_features(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2", "s3"])
        atom_ids = {"s0": "A", "s1": "A", "s2": "B", "s3": "B"}
        sigma_algebra = sa.SigmaAlgebra(sample_space=sample_space, atom_ids=atom_ids)
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, sigma_algebra=sigma_algebra
        )
        features = [[1, 2], [3, 4], [5, 6], [7, 8]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        event = sa.Event(sample_space=sample_space, event_indices=["s0", "s1"])
        assert fps.is_measurable(event) is True
        event_features = fps.get_event_features(["s0", "s1"])
        expected_features = pd.DataFrame(
            data=[[1, 2], [3, 4]], index=["s0", "s1"], columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(event_features.features, expected_features)

    def test_random_variable_from_features(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        probabilities = {"s0": 0.2, "s1": 0.3, "s2": 0.5}
        prob_measure = sa.ProbabilityMeasure(
            sample_space=sample_space, probabilities=probabilities
        )
        prob_space = sa.ProbabilitySpace(
            sample_space=sample_space, probability_measure=prob_measure
        )
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        rv = fps.get_feature_rv("X0")
        assert isinstance(rv, sa.RandomVariable)
        assert rv.domain == sample_space
        expected_values = pd.Series(data=[1, 3, 5], index=["s0", "s1", "s2"], name="X0")
        pd.testing.assert_series_equal(rv.values, expected_values)


class TestEdgeCases:
    def test_single_sample(self):
        sample_space = sa.SampleSpace(["s0"])
        prob_space = sa.ProbabilitySpace(sample_space=sample_space)
        features = [[1, 2]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        assert len(fps.sample_space) == 1
        assert abs(fps.P("s0") - 1.0) < 1e-10

    def test_single_feature(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        prob_space = sa.ProbabilitySpace(sample_space=sample_space)
        features = [[1], [2], [3]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features, sample_space=sample_space
        )
        fps = sa.FeaturizedProbabilitySpace(
            probability_space=prob_space,
            featurized_sample_space=featurized_sample_space,
        )
        assert fps.featurized_sample_space.n_features == 1
        rv = fps.get_feature_rv("X")
        expected_values = pd.Series(data=[1, 2, 3], index=["s0", "s1", "s2"], name="X")
        pd.testing.assert_series_equal(rv.values, expected_values)
