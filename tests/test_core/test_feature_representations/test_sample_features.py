import pytest
import sigalg as sa


class TestConstructionAndBasicProperties:

    @pytest.fixture
    def sample_space(self):
        indices = ["s0", "s1", "s2"]
        return sa.SampleSpace(indices)

    @pytest.fixture
    def space_features(self, sample_space):
        features = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        feature_index = ["T0", "T1", "T2"]
        return sa.SampleSpaceFeatures(
            features=features,
            sample_space=sample_space,
            feature_index=feature_index,
        )

    def test_basic_construction_with_string_index(self, space_features):
        sample_features = sa.SampleFeatures(
            sample_space_features=space_features, sample_index="s1"
        )
        assert sample_features.sample_index == "s1"
        assert sample_features.feature_index[0] == "T0"
        assert sample_features.feature_index[1] == "T1"
        assert sample_features.feature_index[2] == "T2"
        assert sample_features.shape == (3,)
        assert sample_features.n_features == 3
        assert sample_features.n_samples == 1

    def test_construction_without_space_features(self):
        data = [10, 20, 30]
        sample_features = sa.SampleFeatures(features=data)
        assert sample_features.sample_index == "omega"
        assert sample_features.feature_index[0] == "X0"
        assert sample_features.feature_index[1] == "X1"
        assert sample_features.feature_index[2] == "X2"
        assert sample_features.shape == (3,)
        assert sample_features.n_features == 3
        assert sample_features.n_samples == 1


class TestIndexingAndDataAccess:

    @pytest.fixture
    def space_features(self):
        features = [[1, 2, 3], [4, 5, 6]]
        return sa.SampleSpaceFeatures(
            features=features,
            sample_prefix="s",
            feature_prefix="T",
        )

    @pytest.fixture
    def sample_features(self, space_features):
        return sa.SampleFeatures(sample_space_features=space_features, sample_index="s1")

    @pytest.fixture
    def unattached_sample_features(self):
        data = [10, 20, 30]
        return sa.SampleFeatures(features=data)

    def test_data_access(self, sample_features, unattached_sample_features):
        assert sample_features.get_feature_rv("T0") == 4
        assert sample_features.get_feature_rv("T1") == 5
        assert sample_features.get_feature_rv("T2") == 6
        assert sample_features["T0"] == 4
        assert sample_features["T1"] == 5
        assert sample_features["T2"] == 6
        assert sample_features.feature_at[0] == 4
        assert sample_features.feature_at[1] == 5
        assert sample_features.feature_at[2] == 6
        assert unattached_sample_features.get_feature_rv("X0") == 10
        assert unattached_sample_features.get_feature_rv("X1") == 20
        assert unattached_sample_features.get_feature_rv("X2") == 30
        assert unattached_sample_features["X0"] == 10
        assert unattached_sample_features["X1"] == 20
        assert unattached_sample_features["X2"] == 30
        assert unattached_sample_features.feature_at[0] == 10
        assert unattached_sample_features.feature_at[1] == 20
        assert unattached_sample_features.feature_at[2] == 30

    def test_get_sample_features_random_vector(self, sample_features):
        rv_vector_values = sample_features.get_feature_rv(["T0", "T2"])
        assert list(rv_vector_values) == [4, 6]


# class TestMethods:

#     def test_equality_to_numeric(self):
#         data_float = [5.0]
#         sample_features = sa.SampleFeatures(features=data_float)
#         assert sample_features == 5.0
#         assert sample_features != 10.0
#         data_int = [5]
#         sample_features = sa.SampleFeatures(features=data_int)
#         assert sample_features == 5
#         assert sample_features != 10
