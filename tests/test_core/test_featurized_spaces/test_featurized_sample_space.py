import pandas as pd

import sigalg as sa


class TestConstruction:
    def test_with_generated_sample_space_feature_index(
        self,
    ):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        expected_features = pd.DataFrame(
            data=features,
            index=pd.Index(["omega0", "omega1", "omega2"], name="Omega"),
            columns=["X0", "X1"],
        )
        expected_sample_space = sa.SampleSpace(["omega0", "omega1", "omega2"])
        assert featurized_sample_space.sample_space == expected_sample_space
        pd.testing.assert_frame_equal(
            featurized_sample_space.features, expected_features
        )

    def test_with_default_sample_space_feature_index(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features,
            overwrite_default_sample_space=False,
            overwrite_default_feature_index=False,
        )
        expected_features = pd.DataFrame(
            data=features, index=pd.Index([0, 1, 2]), columns=[0, 1]
        )
        expected_sample_space = sa.SampleSpace([0, 1, 2])
        assert featurized_sample_space.sample_space == expected_sample_space
        pd.testing.assert_frame_equal(
            featurized_sample_space.features, expected_features
        )

    def test_with_provided_sample_space_feature_index(self):
        features = [[1, 2], [3, 4], [5, 6]]
        sample_space = sa.SampleSpace(["a", "b", "c"])
        feature_index = ["f1", "f2"]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features,
            sample_space=sample_space,
            feature_index=feature_index,
        )
        expected_features = pd.DataFrame(
            data=features,
            index=sample_space,
            columns=feature_index,
        )
        expected_features.index.name = "Omega"
        assert featurized_sample_space.sample_space == sample_space
        pd.testing.assert_frame_equal(
            featurized_sample_space.features, expected_features
        )

    def test_with_provided_initial_indices(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features,
            initial_sample_index=1,
            initial_feature_index=1,
        )
        expected_features = pd.DataFrame(
            data=features,
            index=["omega1", "omega2", "omega3"],
            columns=["X1", "X2"],
        )
        expected_features.index.name = "Omega"
        expected_sample_space = sa.SampleSpace(["omega1", "omega2", "omega3"])
        assert featurized_sample_space.sample_space == expected_sample_space
        pd.testing.assert_frame_equal(
            featurized_sample_space.features, expected_features
        )

    def test_with_provided_df(self):
        features = pd.DataFrame(
            data=[[1, 2], [3, 4], [5, 6]],
            index=["s1", "s2", "s3"],
            columns=["f1", "f2"],
        )
        features.index.name = "Omega"
        featurized_sample_space = sa.FeaturizedSampleSpace(
            features=features,
            overwrite_default_sample_space=True,
            overwrite_default_feature_index=True,
        )
        expected_sample_space = sa.SampleSpace(["s1", "s2", "s3"])
        assert featurized_sample_space.sample_space == expected_sample_space
        pd.testing.assert_frame_equal(featurized_sample_space.features, features)


class TestGetSampleFeatures:
    def test_get_sample_features(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        sample_features = featurized_sample_space.get_sample_features("omega1")
        expected_sample_features = pd.Series(
            data=[3, 4], index=["X0", "X1"], name="omega1"
        )
        pd.testing.assert_series_equal(
            sample_features.features, expected_sample_features
        )


class TestGetEventFeatures:
    def test_get_event_features(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        event_indices = ["omega0", "omega2"]
        featurized_event = featurized_sample_space.get_event_features(
            event_indices, "B"
        )
        expected_event_features = pd.DataFrame(
            data=[[1, 2], [5, 6]],
            index=["omega0", "omega2"],
            columns=["X0", "X1"],
        )
        expected_event_features.index.name = "B"
        pd.testing.assert_frame_equal(
            featurized_event.features, expected_event_features
        )


class TestGetSampleFeaturesAt:
    def test_get_sample_features_at(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        sample_features = featurized_sample_space.get_sample_features_at[1]
        expected_sample_features = pd.Series(
            data=[3, 4], index=["X0", "X1"], name="omega1"
        )
        pd.testing.assert_series_equal(
            sample_features.values, expected_sample_features
        )


class TestGetEventFeaturesAt:
    def test_get_event_features_at(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        event_features = featurized_sample_space.get_event_features_at[[0, 2]]
        expected_event_features = pd.DataFrame(
            data=[[1, 2], [5, 6]],
            index=["omega0", "omega2"],
            columns=["X0", "X1"],
        )
        expected_event_features.index.name = "Omega"
        pd.testing.assert_frame_equal(event_features.features, expected_event_features)


class TestGetFeatureRV:
    def test_get_feature_rv(self):
        features = [[1, 2], [3, 4], [5, 6]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        feature_rv = featurized_sample_space.get_feature_rv("X1")
        expected_values = pd.Series(
            data=[2, 4, 6],
            index=["omega0", "omega1", "omega2"],
            name="X1",
        )
        expected_values.index.name = "Omega"
        pd.testing.assert_series_equal(feature_rv.values, expected_values)


class TestGetSubFeatures:
    def test_get_sub_features(self):
        features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        featurized_sample_space = sa.FeaturizedSampleSpace(features=features)
        sub_featurized_sample_space = featurized_sample_space.get_sub_features(
            ["X0", "X2"]
        )
        expected_features = pd.DataFrame(
            data=[[1, 3], [4, 6], [7, 9]],
            index=["omega0", "omega1", "omega2"],
            columns=["X0", "X2"],
        )
        expected_features.index.name = "Omega"
        pd.testing.assert_frame_equal(
            sub_featurized_sample_space.features, expected_features
        )


class TestFromSequences:
    def test_from_sequences(self):
        state_space = [0, 1]
        featurized_sample_space = sa.FeaturizedSampleSpace.from_sequences(
            state_space=state_space,
            sequence_length=3,
            initial_sample_index=1,
            initial_feature_index=1,
            sample_prefix="s",
            name="f",
        )
        expected_indices = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        expected_columns = ["f1", "f2", "f3"]
        expected_data = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
        expected_features = pd.DataFrame(
            data=expected_data,
            index=expected_indices,
            columns=expected_columns,
        )
        expected_features.index.name = "Omega"
        assert featurized_sample_space.sample_space == sa.SampleSpace(expected_indices)
        pd.testing.assert_frame_equal(
            featurized_sample_space.features, expected_features
        )


class TestAddProbabilityMeasure:
    def test_add_probability_measure(self):
        state_space = [0, 1]
        fss = sa.FeaturizedSampleSpace.from_sequences(
            state_space=state_space, sequence_length=3
        )

        def pmf(sample_features: sa.SamplePointFeatures) -> float:
            num_ones = sample_features.sum()
            return 0.25**num_ones * 0.75 ** (3 - num_ones)

        fps = fss.add_probability_measure_from_features(pmf)
        assert isinstance(fps, sa.FeaturizedProbabilitySpace)
        expected_probabilities = {
            "omega0": 0.75**3,  # 000
            "omega1": 0.25 * 0.75**2,  # 001
            "omega2": 0.25 * 0.75**2,  # 010
            "omega3": 0.25**2 * 0.75,  # 011
            "omega4": 0.25 * 0.75**2,  # 100
            "omega5": 0.25**2 * 0.75,  # 101
            "omega6": 0.25**2 * 0.75,  # 110
            "omega7": 0.25**3,  # 111
        }

        for sample_index, expected_probability in expected_probabilities.items():
            actual_probability = fps.P(sample_index)
            assert abs(actual_probability - expected_probability) < 1e-10
