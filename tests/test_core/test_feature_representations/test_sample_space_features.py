# import pytest
import sigalg as sa
import pandas as pd


class TestConstructionAndBasicProperties:

    def test_construction_from_sample_space(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        features = [[10, 20], [30, 40], [50, 60]]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=sample_space
        )
        assert space_features._sample_space == sample_space
        expected_df = pd.DataFrame(
            data=features, index=sample_space.index, columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_from_probability_space(self):
        sample_space = sa.SampleSpace(["s0", "s1", "s2"])
        prob_measure = sa.ProbabilityMeasure(
            sample_space, probabilities={"s0": 0.2, "s1": 0.5, "s2": 0.3}
        )
        probability_space = sample_space.add_probability_measure(prob_measure)
        features = [[10, 20], [30, 40], [50, 60]]
        space_features = sa.SampleSpaceFeatures(
            features=features, sample_space=probability_space
        )
        assert space_features.sample_space == sample_space
        assert space_features.probability_space == probability_space
        expected_df = pd.DataFrame(
            data=features, index=sample_space.index, columns=["X0", "X1"]
        )
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_from_data_with_generated_indices(self):
        data = [[1, 2], [3, 4]]
        space_features = sa.SampleSpaceFeatures(
            features=data,
            overwrite_default_sample_space=True,
            overwrite_default_rv_index=True,
        )
        expected_sample_space = sa.SampleSpace(["omega0", "omega1"])
        expected_df = pd.DataFrame(
            data=data,
            index=expected_sample_space.index,
            columns=["X0", "X1"],
        )
        assert space_features._sample_space == expected_sample_space
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_from_data_with_default_indices(self):
        data = [[1, 2], [3, 4]]
        space_features = sa.SampleSpaceFeatures(
            features=data,
            overwrite_default_sample_space=False,
            overwrite_default_rv_index=False,
        )
        expected_sample_space = sa.SampleSpace([0, 1])
        expected_df = pd.DataFrame(
            data=data,
            index=[0, 1],
            columns=[0, 1],
        )
        assert space_features._sample_space == expected_sample_space
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_with_custom_indices(self):
        data = [[1, 2], [3, 4]]
        sample_index = ["s0", "s1"]
        feature_index = ["T0", "T1"]
        space_features = sa.SampleSpaceFeatures(
            data,
            sample_space=sa.SampleSpace(sample_index),
            feature_index=feature_index,
        )
        expected_df = pd.DataFrame(
            data=data,
            index=sample_index,
            columns=feature_index,
        )
        assert space_features._sample_space == sa.SampleSpace(sample_index)
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_construction_with_custom_prefixes_and_initial_indices(self):
        data = [[1, 2], [3, 4]]
        space_features = sa.SampleSpaceFeatures(
            features=data,
            sample_prefix="sample_",
            feature_prefix="feat_",
            initial_sample_index=1,
            initial_feature_index=1,
        )
        expected_sample_space = sa.SampleSpace(["sample_1", "sample_2"])
        expected_df = pd.DataFrame(
            data=data,
            index=expected_sample_space.index,
            columns=["feat_1", "feat_2"],
        )
        assert space_features._sample_space == expected_sample_space
        pd.testing.assert_frame_equal(space_features._data, expected_df)

    def test_from_sequences(self):
        state_space = [0, 1]
        sequence_length = 3
        space_features = sa.SampleSpaceFeatures.from_sequences(
            state_space=state_space,
            sequence_length=sequence_length,
            sample_prefix="s",
            feature_prefix="F",
            threshold=10,
        )
        expected_sample_space = sa.SampleSpace(
            ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        )
        expected_space_features = sa.SampleSpaceFeatures(
            features=[
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ],
            sample_space=expected_sample_space,
            feature_index=["F0", "F1", "F2"],
        )
        assert space_features == expected_space_features
        assert space_features.sample_space == expected_sample_space

    def test_from_sequences_with_initial_feature_index(self):
        state_space = ["a", "b"]
        sequence_length = 2
        space_features = sa.SampleSpaceFeatures.from_sequences(
            state_space=state_space,
            sequence_length=sequence_length,
            initial_sample_index=1,
            initial_feature_index=1,
            sample_prefix="sample_",
            feature_prefix="feature_",
        )
        expected_sample_space = sa.SampleSpace(
            ["sample_1", "sample_2", "sample_3", "sample_4"]
        )
        expected_space_features = sa.SampleSpaceFeatures(
            features=[
                ("a", "a"),
                ("a", "b"),
                ("b", "a"),
                ("b", "b"),
            ],
            sample_space=expected_sample_space,
            feature_index=["feature_1", "feature_2"],
        )
        assert space_features == expected_space_features
        assert space_features.sample_space == expected_sample_space


# class TestArrayIndexingAndDataAccess:

#     @pytest.fixture
#     def space_features(self):
#         data = [[1, 2], [3, 4], [5, 6]]
#         return sa.SampleSpaceFeatures(
#             features=data, sample_index=["s0", "s1", "s2"], feature_index=["A", "B"]
#         )

#     def test_getitem_single_index_str(self, space_features):
#         sample_features = space_features["s2"]
#         assert isinstance(sample_features, sa.SampleFeatures)
#         assert sample_features.sample_index == "s2"

#     def test_getitem_with_list_of_one_index_str(self, space_features):
#         sample_features = space_features[["s2"]]
#         assert isinstance(sample_features, sa.EventFeatures)
#         assert sample_features.sample_index == "s2"

#     def test_get_event_with_one_index_str(self, space_features):
#         sample_features = space_features.get_event(["s2"])
#         assert isinstance(sample_features, sa.EventFeatures)
#         assert sample_features.sample_index == "s2"

#     def test_get_event(self, space_features):
#         event_features = space_features.get_event(["s0", "s2"])
#         assert isinstance(event_features, sa.EventFeatures)
#         assert event_features.sample_space_features == space_features
#         assert list(event_features.sample_index) == ["s0", "s2"]

#     def test_get_sample_features_at(self, space_features):
#         sample_features = space_features.get_sample_features_at[1]
#         assert isinstance(sample_features, sa.SampleFeatures)
#         expected_series = pd.Series(data=[3, 4], index=["A", "B"], name="s1")
#         pd.testing.assert_series_equal(sample_features._data, expected_series)

#     def test_iter(self, space_features):
#         points = list(space_features)
#         assert len(points) == 3
#         assert all(isinstance(p, sa.SampleFeatures) for p in points)
#         assert [p.sample_index for p in points] == ["s0", "s1", "s2"]

#     def test_iter_samples(self, space_features):
#         points = list(space_features.iter_samples())
#         assert len(points) == 3
#         assert all(isinstance(p, sa.SampleFeatures) for p in points)
#         assert [p.sample_index for p in points] == ["s0", "s1", "s2"]

#     def test_get_feature_rv_with_one_index_str(self, space_features):
#         rv = space_features.get_feature_rv("A")
#         expected_values = {"s0": 1, "s1": 3, "s2": 5}
#         expected_rv = sa.RandomVariable(
#             domain_features=space_features, values=expected_values, name="A"
#         )
#         assert isinstance(rv, sa.RandomVariable)
#         assert rv == expected_rv
#         assert rv.name == "A"


# class TestSigmaAlgebra:

#     @pytest.fixture
#     def space_features(self):
#         data = [[1, 2], [3, 4], [5, 6]]
#         return sa.SampleSpaceFeatures(
#             features=data, sample_index=["s0", "s1", "s2"], feature_index=["A", "B"]
#         )

#     def test_default_sigma_algebra(self, space_features):
#         sigma_algebra = space_features.sigma_algebra
#         assert isinstance(sigma_algebra, sa.SigmaAlgebra)
#         expected_atom_ids = {"s0": 0, "s1": 1, "s2": 2}
#         assert sigma_algebra._atom_ids == expected_atom_ids

#     def test_set_sigma_algebra(self, space_features):
#         atom_ids = {"s0": 0, "s1": 0, "s2": 1}
#         sigma_algebra = sa.SigmaAlgebra(
#             sample_space=space_features, atom_ids=atom_ids
#         )
#         space_features.set_sigma_algebra(sigma_algebra)
#         assert space_features.sigma_algebra == sigma_algebra
