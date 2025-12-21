# import numpy as np
# import pandas as pd
# import pytest

# from sigalg.core import FeatureEmbedding, FeatureIndex, RandomVariable, SampleSpace


# class TestConstructor:

#     def test_basic_construction_with_all_parameters(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="X")
#         Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="Y")
#         feature_index = FeatureIndex(indices=["f0", "f1"], values_name="test")
#         feature_embedding = FeatureEmbedding(
#             random_variables=[X, Y], feature_index=feature_index, name="Z"
#         )
#         assert feature_embedding.domain == domain
#         pd.testing.assert_index_equal(
#             feature_embedding.feature_index.values, feature_index.values
#         )
#         assert feature_embedding.name == "Z"
#         expected_df = pd.DataFrame(
#             [[1, 2], [3, 4], [5, 6]],
#             index=pd.Index(["s0", "s1", "s2"], name="sample"),
#             columns=pd.Index(["f0", "f1"], name="test"),
#         )
#         pd.testing.assert_frame_equal(feature_embedding.values, expected_df)

#     def test_basic_construction_with_default_parameters(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V])
#         expected_index = FeatureIndex(["U", "V"])
#         assert feature_embedding.domain == domain
#         pd.testing.assert_index_equal(
#             feature_embedding.feature_index.values, expected_index.values
#         )
#         assert feature_embedding.name == "X"
#         expected_df = pd.DataFrame(
#             [[1, 2], [3, 4], [5, 6]],
#             index=pd.Index(["s0", "s1", "s2"], name="sample"),
#             columns=pd.Index(["U", "V"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(feature_embedding.values, expected_df)


# class TestProperties:

#     def test_domain_property_with_construction_from_values(self):
#         values = pd.DataFrame(
#             [[1], [2], [3]],
#             index=pd.Index(["x", "y", "z"], name="XYZ"),
#             columns=pd.Index(["a"], name="A"),
#         )
#         expected_domain = SampleSpace(["x", "y", "z"], name="XYZ")
#         feature_embedding = FeatureEmbedding(values=values, domain_name="XYZ")
#         assert feature_embedding.domain == expected_domain

#     def test_domain_property_with_construction_from_basic_values(self):
#         values = pd.DataFrame([[1, 2], [3, 4]])
#         expected_domain = SampleSpace([0, 1], name="Omega", values_name="sample")
#         expected_feature_index = FeatureIndex([0, 1], values_name="feature")
#         feature_embedding = FeatureEmbedding(values=values)
#         assert feature_embedding.domain == expected_domain
#         assert feature_embedding.feature_index == expected_feature_index

#     def test_domain_property_with_construction_from_rvs(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V])
#         assert feature_embedding.domain == domain

#     def test_feature_index_property_with_construction_from_rvs(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V])
#         expected_feature_index = FeatureIndex(["U", "V"])
#         assert feature_embedding.feature_index == expected_feature_index

#     def test_name_property(self):
#         values = pd.DataFrame(
#             [[100]],
#             index=pd.Index(["a"], name="A"),
#             columns=pd.Index(["b"], name="B"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="TestName")
#         assert feature_embedding.name == "TestName"

#     def test_shape_property(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4], [5, 6]],
#             index=pd.Index(["r1", "r2", "r3"], name="R"),
#             columns=pd.Index(["c1", "c2"], name="C"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         assert feature_embedding.shape == (3, 2)

#     def test_len_method(self):
#         values = pd.DataFrame(
#             [[1], [2], [3], [4]],
#             index=pd.Index(["a", "b", "c", "d"], name="ABCD"),
#             columns=pd.Index(["x"], name="X"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="Y")
#         assert len(feature_embedding) == 4


# class TestFromNumpy:

#     def test_from_numpy_basic(self):
#         array = np.array([[1, 2], [3, 4]])
#         feature_embedding = FeatureEmbedding.from_numpy(array, name="NumPy")
#         expected_df = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index([0, 1], name="sample"),
#             columns=pd.Index([0, 1], name="feature"),
#         )
#         assert feature_embedding.name == "NumPy"
#         pd.testing.assert_frame_equal(feature_embedding.values, expected_df)


# class TestGetSampleFeatures:

#     def test_get_sample_features_basic_constructed_from_values(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         sample_features = feature_embedding.get_sample_features("s0")
#         assert sample_features.name == "s0"
#         expected_series = pd.Series(
#             [1, 2], index=pd.Index(["f0", "f1"], name="F"), name="s0"
#         )
#         pd.testing.assert_series_equal(sample_features.values, expected_series)

#     def test_get_sample_features_basic_constructed_from_rvs(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V])
#         sample_features = feature_embedding.get_sample_features("s0")
#         assert sample_features.name == "s0"
#         expected_series = pd.Series(
#             [1, 2], index=pd.Index(["U", "V"], name="feature"), name="s0"
#         )
#         pd.testing.assert_series_equal(sample_features.values, expected_series)

#     def test_get_sample_features_at_indexer_constructed_from_values(self):
#         values = pd.DataFrame(
#             [[10, 20], [30, 40], [50, 60]],
#             index=pd.Index(["a", "b", "c"], name="ABC"),
#             columns=pd.Index(["x", "y"], name="XY"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="Z")
#         sample_features = feature_embedding.get_sample_features_at[1]
#         assert sample_features.name == "b"
#         expected_series = pd.Series(
#             [30, 40], index=pd.Index(["x", "y"], name="XY"), name="b"
#         )
#         pd.testing.assert_series_equal(sample_features.values, expected_series)

#     def test_get_sample_features_at_indexer_constructed_from_rvs(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V])
#         sample_features = feature_embedding.get_sample_features_at[0]
#         assert sample_features.name == "s0"
#         expected_series = pd.Series(
#             [1, 2], index=pd.Index(["U", "V"], name="feature"), name="s0"
#         )
#         pd.testing.assert_series_equal(sample_features.values, expected_series)


# class TestGetEventFeatures:

#     def test_get_event_features_basic_constructed_from_values(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4], [5, 6]],
#             index=pd.Index(["s0", "s1", "s2"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         event_features = feature_embedding.get_event_features(["s0", "s2"], name="E")
#         assert event_features.domain.name == "E"
#         assert set(event_features.domain.values) == {"s0", "s2"}
#         expected_df = pd.DataFrame(
#             [[1, 2], [5, 6]],
#             index=pd.Index(["s0", "s2"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         pd.testing.assert_frame_equal(event_features.values, expected_df)

#     def test_get_event_features_basic_constructed_from_rvs(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V])
#         event_features = feature_embedding.get_event_features(["s0", "s2"], name="E")
#         assert event_features.domain.name == "E"
#         assert set(event_features.domain.values) == {"s0", "s2"}
#         expected_df = pd.DataFrame(
#             [[1, 2], [5, 6]],
#             index=pd.Index(["s0", "s2"], name="sample"),
#             columns=pd.Index(["U", "V"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(event_features.values, expected_df)

#     def test_get_event_features_default_name(self):
#         values = pd.DataFrame(
#             [[100], [200]],
#             index=pd.Index(["a", "b"], name="AB"),
#             columns=pd.Index(["x"], name="X"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="Y")
#         event_features = feature_embedding.get_event_features(["a"])
#         assert event_features.domain.name == "A"


# class TestGetEventFeaturesAt:

#     @pytest.fixture
#     def embedding_from_rvs(self):
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVariable(
#             outputs={"omega0": 1, "omega1": 3, "omega2": 5}, domain=Omega, name="X"
#         )
#         Y = RandomVariable(
#             outputs={"omega0": 2, "omega1": 4, "omega2": 6}, domain=Omega, name="Y"
#         )
#         return FeatureEmbedding(random_variables=[X, Y])

#     @pytest.fixture
#     def embedding_from_values(self):
#         df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
#         return FeatureEmbedding(values=df)

#     def test_get_event_features_at_int(self, embedding_from_rvs, embedding_from_values):
#         features_from_rvs = embedding_from_rvs.get_event_features_at[0]
#         features_from_values = embedding_from_values.get_event_features_at[0]
#         expected_df = pd.DataFrame(
#             data=[[1, 2]],
#             index=pd.Index(["omega0"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         expected_df_from_values = pd.DataFrame(data=[[1, 2]])
#         pd.testing.assert_frame_equal(features_from_rvs.values, expected_df)
#         pd.testing.assert_frame_equal(
#             features_from_values.values, expected_df_from_values
#         )
#         assert features_from_rvs.domain.name == "A"
#         assert features_from_values.domain.name == "A"

#     def test_get_event_features_at_int_and_custom_names(
#         self, embedding_from_rvs, embedding_from_values
#     ):
#         features_from_rvs = embedding_from_rvs.get_event_features_at[0, "B"]
#         features_from_values = embedding_from_values.get_event_features_at[0, "B"]
#         expected_df = pd.DataFrame(
#             data=[[1, 2]],
#             index=pd.Index(["omega0"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         expected_df_from_values = pd.DataFrame(data=[[1, 2]])
#         pd.testing.assert_frame_equal(features_from_rvs.values, expected_df)
#         pd.testing.assert_frame_equal(
#             features_from_values.values, expected_df_from_values
#         )
#         assert features_from_rvs.domain.name == "B"
#         assert features_from_values.domain.name == "B"

#     def test_get_event_features_at_list(
#         self, embedding_from_rvs, embedding_from_values
#     ):
#         features_from_rvs = embedding_from_rvs.get_event_features_at[[0, 1]]
#         features_from_values = embedding_from_values.get_event_features_at[[0, 1]]
#         expected_df = pd.DataFrame(
#             data=[[1, 2], [3, 4]],
#             index=pd.Index(["omega0", "omega1"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         expected_df_from_values = pd.DataFrame(data=[[1, 2], [3, 4]])
#         pd.testing.assert_frame_equal(features_from_rvs.values, expected_df)
#         pd.testing.assert_frame_equal(
#             features_from_values.values, expected_df_from_values
#         )
#         assert features_from_rvs.domain.name == "A"
#         assert features_from_values.domain.name == "A"

#     def test_get_event_features_at_list_and_custom_names(
#         self, embedding_from_rvs, embedding_from_values
#     ):
#         features_from_rvs = embedding_from_rvs.get_event_features_at[[0, 1], "B"]
#         features_from_values = embedding_from_values.get_event_features_at[[0, 1], "B"]
#         expected_df = pd.DataFrame(
#             data=[[1, 2], [3, 4]],
#             index=pd.Index(["omega0", "omega1"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         expected_df_from_values = pd.DataFrame(data=[[1, 2], [3, 4]])
#         pd.testing.assert_frame_equal(features_from_rvs.values, expected_df)
#         pd.testing.assert_frame_equal(
#             features_from_values.values, expected_df_from_values
#         )
#         assert features_from_rvs.domain.name == "B"
#         assert features_from_values.domain.name == "B"

#     def test_get_event_features_at_slice(
#         self, embedding_from_rvs, embedding_from_values
#     ):
#         features_from_rvs = embedding_from_rvs.get_event_features_at[:2]
#         features_from_values = embedding_from_values.get_event_features_at[:2]
#         expected_df = pd.DataFrame(
#             data=[[1, 2], [3, 4]],
#             index=pd.Index(["omega0", "omega1"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         expected_df_from_values = pd.DataFrame(data=[[1, 2], [3, 4]])
#         pd.testing.assert_frame_equal(features_from_rvs.values, expected_df)
#         pd.testing.assert_frame_equal(
#             features_from_values.values, expected_df_from_values
#         )
#         assert features_from_rvs.domain.name == "A"
#         assert features_from_values.domain.name == "A"

#     def test_get_event_features_at_slice_and_custom_names(
#         self, embedding_from_rvs, embedding_from_values
#     ):
#         features_from_rvs = embedding_from_rvs.get_event_features_at[:2, "B"]
#         features_from_values = embedding_from_values.get_event_features_at[:2, "B"]
#         expected_df = pd.DataFrame(
#             data=[[1, 2], [3, 4]],
#             index=pd.Index(["omega0", "omega1"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         expected_df_from_values = pd.DataFrame(data=[[1, 2], [3, 4]])
#         pd.testing.assert_frame_equal(features_from_rvs.values, expected_df)
#         pd.testing.assert_frame_equal(
#             features_from_values.values, expected_df_from_values
#         )
#         assert features_from_rvs.domain.name == "B"
#         assert features_from_values.domain.name == "B"


# class TestCall:

#     def test_call_on_sample_index(self):
#         Omega = SampleSpace(["s0", "s1", "s2"])
#         X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
#         Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
#         embedding = FeatureEmbedding(random_variables=[X, Y])
#         spf = embedding("s0")
#         expected_series = pd.Series(
#             [1, 2], index=pd.Index(["X", "Y"], name="feature"), name="s0"
#         )
#         pd.testing.assert_series_equal(spf.values, expected_series)

#         df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
#         embedding = FeatureEmbedding(values=df)
#         spf = embedding(0)
#         expected_series = pd.Series([1, 2], name=0)
#         pd.testing.assert_series_equal(spf.values, expected_series)

#     def test_call_on_list(self):
#         Omega = SampleSpace(["s0", "s1", "s2"])
#         X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
#         Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
#         embedding = FeatureEmbedding(random_variables=[X, Y])
#         event_embedding = embedding(["s0", "s1"])
#         expected_df = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(event_embedding.values, expected_df)

#         df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
#         embedding = FeatureEmbedding(values=df)
#         event_embedding = embedding([0, 1])
#         expected_df = pd.DataFrame([[1, 2], [3, 4]])
#         pd.testing.assert_frame_equal(event_embedding.values, expected_df)

#     def test_call_on_event(self):
#         Omega = SampleSpace(["s0", "s1", "s2"])
#         X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
#         Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
#         embedding = FeatureEmbedding(random_variables=[X, Y])
#         B = Omega.get_event(["s0", "s1"], name="B")
#         event_embedding = embedding(B)
#         expected_df = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="sample"),
#             columns=pd.Index(["X", "Y"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(event_embedding.values, expected_df)
#         assert event_embedding.domain.name == "B"

#         df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
#         embedding = FeatureEmbedding(values=df)
#         domain = embedding.domain
#         B = domain.get_event([0, 1], name="B")
#         event_embedding = embedding(B)
#         expected_df = pd.DataFrame([[1, 2], [3, 4]])
#         pd.testing.assert_frame_equal(event_embedding.values, expected_df)
#         assert event_embedding.domain.name == "B"


# class TestGetFeatureRV:

#     def test_get_feature_rv_basic_constructed_from_values(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4], [5, 6]],
#             index=pd.Index(["s0", "s1", "s2"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         rv = feature_embedding.get_feature_rv("f0")
#         assert rv.name == "f0"
#         expected_values = pd.Series(
#             [1, 3, 5], index=pd.Index(["s0", "s1", "s2"], name="sample"), name="f0"
#         )
#         pd.testing.assert_series_equal(rv.values, expected_values)

#     def test_get_feature_rv_basic_constructed_from_rvs(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V])
#         rv = feature_embedding.get_feature_rv("U")
#         assert rv.name == "U"
#         expected_values = pd.Series(
#             [1, 3, 5], index=pd.Index(["s0", "s1", "s2"], name="sample"), name="U"
#         )
#         pd.testing.assert_series_equal(rv.values, expected_values)

#     def test_get_feature_rv_with_custom_feature_index(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="V")
#         feature_index = FeatureIndex(["A", "B"])
#         feature_embedding = FeatureEmbedding(
#             random_variables=[U, V], feature_index=feature_index
#         )
#         rv = feature_embedding.get_feature_rv("A")
#         assert rv.name == "A"
#         expected_values = pd.Series(
#             [1, 3, 5], index=pd.Index(["s0", "s1", "s2"], name="sample"), name="A"
#         )
#         pd.testing.assert_series_equal(rv.values, expected_values)


# class TestGetSubEmbedding:

#     def test_get_sub_embedding_single_column(self):
#         values = pd.DataFrame(
#             [[1, 2, 3], [4, 5, 6]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1", "f2"], name="F"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         sub_embedding = feature_embedding.get_sub_embedding(["f1"])
#         assert sub_embedding.name == "X_sub"
#         assert sub_embedding.domain == feature_embedding.domain
#         expected_df = pd.DataFrame(
#             [[2], [5]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f1"], name="F"),
#         )
#         pd.testing.assert_frame_equal(sub_embedding.values, expected_df)

#     def test_get_sub_embedding_multiple_columns(self):
#         values = pd.DataFrame(
#             [[10, 20, 30], [40, 50, 60]],
#             index=pd.Index(["a", "b"], name="AB"),
#             columns=pd.Index(["c0", "c1", "c2"], name="C"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="Y")
#         sub_embedding = feature_embedding.get_sub_embedding(["c0", "c2"])
#         expected_df = pd.DataFrame(
#             [[10, 30], [40, 60]],
#             index=pd.Index(["a", "b"], name="AB"),
#             columns=pd.Index(["c0", "c2"], name="C"),
#         )
#         pd.testing.assert_frame_equal(sub_embedding.values, expected_df)

#     def test_get_sub_embedding_constructed_from_rvs(self):
#         domain = SampleSpace(["s0", "s1", "s2"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=domain, name="V")
#         W = RandomVariable(outputs={"s0": 7, "s1": 8, "s2": 9}, domain=domain, name="W")
#         feature_index = FeatureIndex(["A", "B", "C"])
#         feature_embedding = FeatureEmbedding(
#             random_variables=[U, V, W], feature_index=feature_index
#         )
#         sub_embedding = feature_embedding.get_sub_embedding(["A", "C"])
#         expected_df = pd.DataFrame(
#             [[1, 7], [3, 8], [5, 9]],
#             index=pd.Index(["s0", "s1", "s2"], name="sample"),
#             columns=pd.Index(["A", "C"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(sub_embedding.values, expected_df)


# class TestIterSampleFeatures:

#     def test_iter_sample_features(self):
#         values = pd.DataFrame(
#             [[100], [200]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0"], name="F"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         indices = []
#         for sample_index, sample_features in feature_embedding.iter_sample_features():
#             indices.append(sample_index)
#             assert sample_features.name == sample_index
#         assert indices == ["s0", "s1"]


# class TestApplyToFeatures:

#     def test_apply_to_features_sum(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4], [5, 6]],
#             index=pd.Index(["s0", "s1", "s2"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         result = feature_embedding.apply_to_features(lambda sf: sf.values.sum())
#         expected_series = pd.Series(
#             [3, 7, 11], index=pd.Index(["s0", "s1", "s2"], name="S")
#         )
#         pd.testing.assert_series_equal(result, expected_series)

#     def test_apply_to_features_max(self):
#         domain = SampleSpace(["a", "b"], name="AB")
#         U = RandomVariable(outputs={"a": 1, "b": 4}, domain=domain, name="U")
#         V = RandomVariable(outputs={"a": 3, "b": 2}, domain=domain, name="V")
#         feature_embedding = FeatureEmbedding(random_variables=[U, V], name="W")
#         result = feature_embedding.apply_to_features(lambda sf: sf.values.max())
#         expected_series = pd.Series([3, 4], index=pd.Index(["a", "b"], name="sample"))
#         pd.testing.assert_series_equal(result, expected_series)


# class TestEquality:

#     def test_equal_feature_embeddings(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         fe1 = FeatureEmbedding(values=values, name="X")
#         fe2 = FeatureEmbedding(values=values, name="X")
#         assert fe1 == fe2

#     def test_not_equal_different_values(self):
#         values1 = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         values2 = pd.DataFrame(
#             [[1, 2], [3, 5]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         fe1 = FeatureEmbedding(values=values1, name="X")
#         fe2 = FeatureEmbedding(values=values2, name="X")
#         assert fe1 != fe2

#     def test_not_equal_different_name(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         fe1 = FeatureEmbedding(values=values, name="X")
#         fe2 = FeatureEmbedding(values=values, name="Y")
#         assert fe1 != fe2

#     def test_not_equal_to_non_feature_embedding(self):
#         values = pd.DataFrame(
#             [[1]], index=pd.Index(["s0"], name="S"), columns=pd.Index(["f0"], name="F")
#         )
#         fe = FeatureEmbedding(values=values, name="X")
#         assert fe != "not a feature embedding"
#         assert fe != 42
#         assert fe is not None


# class TestSetters:

#     def test_name_setter(self):
#         values = pd.DataFrame(
#             [[1]], index=pd.Index(["s0"], name="S"), columns=pd.Index(["f0"], name="F")
#         )
#         feature_embedding = FeatureEmbedding(values=values, name="X")
#         feature_embedding.name = "NewName"
#         assert feature_embedding.name == "NewName"


# class TestValidation:

#     def test_random_variables_must_be_list(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         with pytest.raises(TypeError, match="random_variables must be a list"):
#             FeatureEmbedding(random_variables=U)

#     def test_random_variables_elements_must_be_random_variable_instances(self):
#         with pytest.raises(
#             TypeError,
#             match="All elements in random_variables must be instances of RandomVariable",
#         ):
#             FeatureEmbedding(random_variables=[1, 2, 3])

#     def test_random_variables_mixed_types_invalid(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         with pytest.raises(
#             TypeError,
#             match="All elements in random_variables must be instances of RandomVariable",
#         ):
#             FeatureEmbedding(random_variables=[U, "not_a_rv", 42])

#     def test_feature_index_must_be_index_instance(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         with pytest.raises(TypeError, match="feature_index must be an Index instance"):
#             FeatureEmbedding(random_variables=[U], feature_index="not_an_index")

#     def test_feature_index_must_be_index_instance_not_list(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         with pytest.raises(TypeError, match="feature_index must be an Index instance"):
#             FeatureEmbedding(random_variables=[U], feature_index=["f0", "f1"])

#     def test_values_must_be_dataframe(self):
#         with pytest.raises(TypeError, match="values must be a pandas DataFrame."):
#             FeatureEmbedding(values="not_a_dataframe")

#     def test_values_must_be_dataframe_not_series(self):
#         series = pd.Series([1, 2, 3])
#         with pytest.raises(TypeError, match="values must be a pandas DataFrame."):
#             FeatureEmbedding(values=series)

#     def test_feature_index_and_random_variables_length_mismatch(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         V = RandomVariable(outputs={"s0": 2, "s1": 4}, domain=domain, name="V")
#         feature_index = FeatureIndex(["f0"])
#         with pytest.raises(
#             ValueError,
#             match="feature_index and random_variables must have the same length",
#         ):
#             FeatureEmbedding(random_variables=[U, V], feature_index=feature_index)

#     def test_feature_index_and_random_variables_length_mismatch_opposite(self):
#         domain = SampleSpace(["s0", "s1"], name="S")
#         U = RandomVariable(outputs={"s0": 1, "s1": 3}, domain=domain, name="U")
#         feature_index = FeatureIndex(["f0", "f1", "f2"])
#         with pytest.raises(
#             ValueError,
#             match="feature_index and random_variables must have the same length",
#         ):
#             FeatureEmbedding(random_variables=[U], feature_index=feature_index)

#     def test_name_must_be_string(self):
#         values = pd.DataFrame([[1, 2], [3, 4]])
#         with pytest.raises(TypeError, match="name must be a string"):
#             FeatureEmbedding(values=values, name=123)

#     def test_name_must_be_string_not_none(self):
#         values = pd.DataFrame([[1, 2], [3, 4]])
#         with pytest.raises(TypeError, match="name must be a string"):
#             FeatureEmbedding(values=values, name=None)

#     def test_name_setter_must_be_string(self):
#         values = pd.DataFrame([[1, 2], [3, 4]])
#         fe = FeatureEmbedding(values=values, name="X")
#         with pytest.raises(TypeError, match="name must be a string"):
#             fe.name = 456

#     def test_get_sample_features_invalid_sample_index(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         fe = FeatureEmbedding(values=values, name="X")
#         with pytest.raises(ValueError, match="Sample index s999 not found in domain"):
#             fe.get_sample_features("s999")

#     def test_get_event_features_invalid_sample_index(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4]],
#             index=pd.Index(["s0", "s1"], name="S"),
#             columns=pd.Index(["f0", "f1"], name="F"),
#         )
#         fe = FeatureEmbedding(values=values, name="X")
#         with pytest.raises(
#             ValueError, match="Sample index invalid not found in sample_space"
#         ):
#             fe.get_event_features(["s0", "invalid"])

#     def test_get_event_features_partial_invalid_indices(self):
#         domain = SampleSpace(["a", "b", "c"], name="ABC")
#         U = RandomVariable(outputs={"a": 1, "b": 2, "c": 3}, domain=domain, name="U")
#         fe = FeatureEmbedding(random_variables=[U], name="Test")
#         with pytest.raises(
#             ValueError, match="Sample index z not found in sample_space"
#         ):
#             fe.get_event_features(["a", "b", "z"])

#     def test_get_event_features_at_index_out_of_bounds(self):
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVariable(
#             outputs={"omega0": 1, "omega1": 3, "omega2": 5}, domain=Omega, name="X"
#         )
#         Y = RandomVariable(
#             outputs={"omega0": 2, "omega1": 4, "omega2": 6}, domain=Omega, name="Y"
#         )
#         embedding = FeatureEmbedding(random_variables=[X, Y])
#         with pytest.raises(IndexError):
#             embedding.get_event_features_at[3]
