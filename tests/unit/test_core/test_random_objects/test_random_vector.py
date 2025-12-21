import pandas as pd
import pytest

from sigalg.core import Index, RandomVector, SampleSpace


class TestConstructor:

    @pytest.mark.parametrize(
        "domain_indices, outputs, dimension, name, expected_feature_indices",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
                2,
                "Y",
                ["Y0", "Y1"],
                id="2d_outputs",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 10, "omega1": 20, "omega2": 30},
                1,
                "Z",
                ["Z"],
                id="1d_outputs",
            ),
            pytest.param(
                [0, 1, 2, 3],
                {0: (1, 2), 1: (3, 4), 2: (5, 6), 3: (7, 8)},
                2,
                "default_name_flag",
                ["X0", "X1"],
                id="default_name",
            ),
            pytest.param(
                [0, 1, 2],
                {0: 100, 1: 200, 2: 300},
                1,
                None,
                [0],
                id="none_name",
            ),
            pytest.param(
                ["a", "b", "c"],
                {"a": (0.1, 0.2, 0.3), "b": (0.4, 0.5, 0.6), "c": (0.7, 0.8, 0.9)},
                3,
                42,
                [0, 1, 2],
                id="non_string_name",
            ),
        ],
    )
    def test_constructor(
        self, domain_indices, outputs, dimension, name, expected_feature_indices
    ):
        """Test RandomVector constructor with various outputs and domain indices."""
        domain = SampleSpace(indices=domain_indices, name="Omega")
        if name == "default_name_flag":
            rv = RandomVector(outputs=outputs, domain=domain)
            name = "X"
        else:
            rv = RandomVector(outputs=outputs, domain=domain, name=name)
        expected_feature_index = Index(
            indices=expected_feature_indices, name="feature_index", data_name="feature"
        )
        expected_data = pd.DataFrame.from_dict(
            data=outputs, orient="index", columns=expected_feature_index.data
        )
        expected_data.index.name = domain.data.name

        pd.testing.assert_frame_equal(rv.data, expected_data)
        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == name
        assert rv.feature_index == expected_feature_index


class TestFromPandas:

    @pytest.mark.parametrize(
        "df_data, index, columns, name",
        [
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                pd.Index(["a", "b", "c"], name="letters"),
                pd.Index(["black", "blue"], name="colors"),
                "Z",
                id="custom_indices",
            ),
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                None,
                None,
                "X",
                id="default_indices",
            ),
            pytest.param(
                [10, 20, 30],
                pd.Index([0, 1, 2], name="numbers"),
                None,
                "V",
                id="1d_custom_index",
            ),
        ],
    )
    def test_from_pandas(self, df_data, index, columns, name):
        """Test RandomVector.from_pandas method."""
        data = pd.DataFrame(data=df_data, index=index, columns=columns)
        rv = RandomVector.from_pandas(data=data, name=name)
        expected_domain = SampleSpace(
            indices=list(data.index), name="Omega", data_name=data.index.name
        )
        expected_feature_index = Index(
            indices=list(data.columns),
            name="feature_index",
            data_name=data.columns.name,
        )

        pd.testing.assert_frame_equal(rv.data, data)
        assert rv.domain == expected_domain
        assert rv.feature_index == expected_feature_index
        assert rv.name == name


class TestRange:

    @pytest.mark.parametrize(
        "outputs, name, domain_indices, expected_range_outputs, expected_range_name, expected_range_domain_indices, expected_range_feature_indices",
        [
            pytest.param(
                {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
                "X",
                ["omega0", "omega1", "omega2"],
                {"x0": (3, 4), "x1": (1, 2)},
                "range(X)",
                ["x0", "x1"],
                ["X0", "X1"],
                id="2d_random_vector_with_str_name",
            ),
            pytest.param(
                {"omega0": 10, "omega1": 20, "omega2": 10},
                "Y",
                ["omega0", "omega1", "omega2"],
                {"y0": 10, "y1": 20},
                "range(Y)",
                ["y0", "y1"],
                ["Y"],
                id="1d_random_vector_with_str_name",
            ),
            pytest.param(
                {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
                42,
                ["omega0", "omega1", "omega2"],
                {0: (3, 4), 1: (1, 2)},
                None,
                [0, 1],
                [0, 1],
                id="2d_random_vector_with_int_name",
            ),
            pytest.param(
                {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
                None,
                ["omega0", "omega1", "omega2"],
                {0: (3, 4), 1: (1, 2)},
                None,
                [0, 1],
                [0, 1],
                id="2d_random_vector_with_none_name",
            ),
        ],
    )
    def test_range(
        self,
        outputs,
        name,
        domain_indices,
        expected_range_outputs,
        expected_range_name,
        expected_range_domain_indices,
        expected_range_feature_indices,
    ):
        """Test range property of RandomVector."""
        domain = SampleSpace(indices=domain_indices, name="Omega")
        rv = RandomVector(outputs=outputs, domain=domain, name=name)
        expected_range_domain = SampleSpace(
            indices=expected_range_domain_indices,
            name=expected_range_name,
            data_name="output",
        )
        expected_range_data = pd.DataFrame.from_dict(
            data=expected_range_outputs,
            orient="index",
            columns=expected_range_feature_indices,
        )
        expected_range_data.index.name = "output"
        expected_range_data.columns.name = "feature"

        pd.testing.assert_frame_equal(rv.range.data, expected_range_data)
        assert rv.range.domain == expected_range_domain
        assert rv.range.name == expected_range_name

    def test_range_on_rv_constructed_from_pandas(self):
        """Test range property of RandomVector constructed from pandas DataFrame."""
        data = pd.DataFrame([(1, 2), (3, 4), (3, 4)])
        rv = RandomVector.from_pandas(data=data)
        expected_range_data = pd.DataFrame(
            data=[(3, 4), (1, 2)],
            index=pd.Index(["x0", "x1"], name="output"),
            columns=[0, 1],
        )
        expected_range_domain = SampleSpace(
            indices=["x0", "x1"], name="range(X)", data_name="output"
        )

        pd.testing.assert_frame_equal(rv.range.data, expected_range_data)
        assert rv.range.domain == expected_range_domain
        assert rv.range.name == "range(X)"


# class TestFeatureIndex:

#     def test_feature_index_property_with_construction_from_values(self):
#         """Test feature_index property of RandomVector constructed from values with custom indices."""
#         values = pd.DataFrame(
#             [(1, 2), (3, 4), (5, 6)],
#             index=pd.Index(["a", "b", "c"], name="letters"),
#             columns=pd.Index(["black", "blue"], name="colors"),
#         )
#         Z = RandomVector.from_pandas(data=values, name="Z")
#         expected_feature_index = FeatureIndex(
#             indices=["black", "blue"], values_name="colors"
#         )
#         assert Z.feature_index == expected_feature_index
#         new_feature_index = FeatureIndex(
#             indices=["red", "green"], values_name="new_colors"
#         )
#         Z.feature_index = new_feature_index
#         assert Z.feature_index == new_feature_index

#     def test_feature_index_property_with_construction_from_output(self):
#         """Test feature_index property of RandomVector constructed from outputs."""
#         outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
#         Omega = SampleSpace.generate_default(size=3)
#         Y = RandomVector(outputs=outputs, domain=Omega, name="Y")
#         expected_feature_index = FeatureIndex(
#             indices=["Y0", "Y1"], values_name="feature"
#         )
#         assert Y.feature_index == expected_feature_index
#         new_feature_index = FeatureIndex(
#             indices=["red", "green"], values_name="new_colors"
#         )
#         Y.feature_index = new_feature_index
#         assert Y.feature_index == new_feature_index

#     def test_feature_index_property_with_construction_from_values_basic(self):
#         """Test feature_index property of RandomVector constructed from values with default indices."""
#         values = pd.DataFrame([(1, 2), (3, 4), (5, 6)])
#         X = RandomVector.from_pandas(data=values)
#         expected_feature_index = FeatureIndex(indices=[0, 1], values_name=None)
#         assert X.feature_index == expected_feature_index
#         new_feature_index = FeatureIndex(
#             indices=["red", "green"], values_name="new_colors"
#         )
#         X.feature_index = new_feature_index
#         assert X.feature_index == new_feature_index

#     def test_feature_index_property_with_1d_random_vector(self):
#         """Test feature_index property of single-component RandomVector."""
#         values = pd.DataFrame([10, 20, 30])
#         V = RandomVector.from_pandas(data=values, name="V")
#         expected_feature_index = FeatureIndex(indices=[0], values_name=None)
#         assert V.feature_index == expected_feature_index
#         new_feature_index = FeatureIndex(indices=["numbers"], values_name="new_feature")
#         V.feature_index = new_feature_index
#         assert V.feature_index == new_feature_index


# class TestRange:

#     def test_range_constructed_from_outputs(self):
#         """Test range property of RandomVector constructed from outputs."""
#         Omega = SampleSpace.generate_default(size=3)
#         outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)}
#         X = RandomVector(outputs=outputs, domain=Omega, name="X")
#         expected_df = pd.DataFrame(
#             data=[(3, 4), (1, 2)],
#             index=pd.Index(["x0", "x1"], name="output"),
#             columns=pd.Index(["X0", "X1"], name="feature"),
#         )
#         expected_counts = pd.Series(data=[2, 1], index=expected_df.index, name="count")
#         pd.testing.assert_frame_equal(X.range.data, expected_df)
#         pd.testing.assert_series_equal(X.range_counts, expected_counts)
#         assert X.range.name == "range(X)"

#     def test_range_constructed_from_values_basic(self):
#         """Test range property of RandomVector constructed from values with default indices."""
#         values = pd.DataFrame([(1, 2), (3, 4), (3, 4)])
#         X = RandomVector.from_pandas(data=values)
#         expected_df = pd.DataFrame(
#             data=[(3, 4), (1, 2)],
#             index=pd.Index(["x0", "x1"], name="output"),
#         )
#         expected_counts = pd.Series(data=[2, 1], index=expected_df.index, name="count")
#         pd.testing.assert_frame_equal(X.range.data, expected_df)
#         pd.testing.assert_series_equal(X.range_counts, expected_counts)
#         assert X.range.name == "range(X)"

#     def test_range_from_values(self):
#         """Test range property of RandomVector constructed from values with custom indices."""
#         values = pd.DataFrame(
#             [(1, 2), (3, 4), (3, 4)],
#             index=pd.Index(["a", "b", "c"], name="letters"),
#             columns=pd.Index(["black", "blue"], name="colors"),
#         )
#         Y = RandomVector.from_pandas(data=values, name="Y")
#         expected_df = pd.DataFrame(
#             data=[(3, 4), (1, 2)],
#             index=pd.Index(["y0", "y1"], name="output"),
#             columns=pd.Index(["black", "blue"], name="colors"),
#         )
#         expected_counts = pd.Series(data=[2, 1], index=expected_df.index, name="count")
#         pd.testing.assert_frame_equal(Y.range.data, expected_df)
#         pd.testing.assert_series_equal(Y.range_counts, expected_counts)
#         assert Y.range.name == "range(Y)"


# class TestRangeCounts:

#     def test_range_counts_constructed_from_outputs(self):
#         """Test range_counts property of RandomVector constructed from outputs."""
#         Omega = SampleSpace.generate_default(size=3)
#         outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)}
#         X = RandomVector(outputs=outputs, domain=Omega, name="X")
#         expected_counts = pd.Series(
#             data=[2, 1], index=pd.Index(["x0", "x1"], name="output"), name="count"
#         )
#         pd.testing.assert_series_equal(X.range_counts, expected_counts)

#     def test_range_counts_constructed_from_values_basic(self):
#         """Test range_counts property of RandomVector constructed from values with default indices."""
#         values = pd.DataFrame([(1, 2), (3, 4), (3, 4)])
#         X = RandomVector.from_pandas(data=values)
#         expected_counts = pd.Series(
#             data=[2, 1], index=pd.Index(["x0", "x1"], name="output"), name="count"
#         )
#         pd.testing.assert_series_equal(X.range_counts, expected_counts)

#     def test_range_counts_from_values(self):
#         """Test range_counts property of RandomVector constructed from values with custom indices."""
#         values = pd.DataFrame(
#             [(1, 2), (3, 4), (3, 4)],
#             index=pd.Index(["a", "b", "c"], name="letters"),
#             columns=pd.Index(["black", "blue"], name="colors"),
#         )
#         Y = RandomVector.from_pandas(data=values, name="Y")
#         expected_counts = pd.Series(
#             data=[2, 1], index=pd.Index(["y0", "y1"], name="output"), name="count"
#         )
#         pd.testing.assert_series_equal(Y.range_counts, expected_counts)


# class TestCallMethod:

#     def test_call_method_on_sample_index(self):
#         """Test calling RandomVector on a single sample index."""
#         values = pd.DataFrame(
#             [(1, 2), (3, 4), (5, 6)],
#             index=pd.Index(["a", "b", "c"], name="letters"),
#             columns=pd.Index(["black", "blue"], name="colors"),
#         )
#         Y = RandomVector.from_pandas(data=values, name="Y")
#         expected_spf = SamplePointFeatures(values=values.loc["a"], name="a")
#         pd.testing.assert_series_equal(Y("a").data, expected_spf.values)

#     def test_call_method_on_sample_indices(self):
#         """Test calling RandomVector on a list of sample indices."""
#         Omega = SampleSpace.generate_default(size=3)
#         outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
#         X = RandomVector(outputs=outputs, domain=Omega, name="X")
#         expected_rv = RandomVector.from_pandas(
#             data=pd.DataFrame(
#                 [(1, 2), (5, 6)],
#                 index=pd.Index(["omega0", "omega2"], name="sample"),
#                 columns=pd.Index(["X0", "X1"], name="feature"),
#             ),
#             name="X_subset",
#         )
#         pd.testing.assert_frame_equal(
#             X(["omega0", "omega2"]).data, expected_rv.data
#         )
#         assert X(["omega0", "omega2"]).name == "X|event"

#     def test_call_method_on_event(self):
#         """Test calling RandomVector on an Event."""
#         Omega = SampleSpace.generate_default(size=3)
#         outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
#         X = RandomVector(outputs=outputs, domain=Omega, name="X")
#         event = Omega.get_event(["omega0", "omega2"])
#         expected_rv = RandomVector.from_pandas(
#             data=pd.DataFrame(
#                 [(1, 2), (5, 6)],
#                 index=pd.Index(["omega0", "omega2"], name="sample"),
#                 columns=pd.Index(["X0", "X1"], name="feature"),
#             ),
#             name="X|A",
#         )
#         pd.testing.assert_frame_equal(X(event).data, expected_rv.data)
#         assert X(event).name == "X|A"

#     def test_call_method_on_1d_random_vector(self):
#         """Test calling single-component RandomVector on various inputs."""
#         values = pd.DataFrame([10, 20, 30])
#         V = RandomVector.from_pandas(data=values, name="V")
#         expected_spf = SamplePointFeatures(values=values.loc[0], name=0)
#         pd.testing.assert_series_equal(V(0).data, expected_spf.values)
#         expected_rv_indices = RandomVector.from_pandas(
#             data=pd.DataFrame([10, 30], index=pd.Index([0, 2])),
#             name="V_subset",
#         )
#         pd.testing.assert_frame_equal(V([0, 2]).data, expected_rv_indices.data)
#         assert V([0, 2]).name == "V|event"
#         Omega = V.domain
#         event = Omega.get_event([0, 2], name="B")
#         expected_rv_event = RandomVector.from_pandas(
#             data=pd.DataFrame([10, 30], index=pd.Index([0, 2])),
#             name="V|B",
#         )
#         pd.testing.assert_frame_equal(V(event).data, expected_rv_event.data)
#         assert V(event).name == "V|B"


# class TestGetItem:

#     def test_getitem_on_int(self):
#         """Test indexing RandomVector with an integer."""
#         Omega = SampleSpace.generate_default(size=3)
#         outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
#         X = RandomVector(outputs=outputs, domain=Omega, name="X")
#         expected_spf = SamplePointFeatures(values=X.data.iloc[0], name="omega0")
#         pd.testing.assert_series_equal(X[0].data, expected_spf.values)

#     def test_getitem_on_slice(self):
#         """Test slicing RandomVector with a slice object."""
#         values = pd.DataFrame(
#             [(1, 2), (3, 4), (3, 4)],
#             index=pd.Index(["a", "b", "c"], name="letters"),
#             columns=pd.Index(["black", "blue"], name="colors"),
#         )
#         Y = RandomVector.from_pandas(data=values, name="Y")
#         expected_rv = RandomVector.from_pandas(
#             data=pd.DataFrame(
#                 [[1, 2], [3, 4]],
#                 index=pd.Index(["a", "b"], name="letters"),
#                 columns=pd.Index(["black", "blue"], name="colors"),
#             ),
#             name="Y|event",
#         )
#         pd.testing.assert_frame_equal(Y[:2].data, expected_rv.data)
#         assert Y[:2].name == "Y|event"

#     def test_getitem_on_list_of_ints(self):
#         values = pd.DataFrame(
#             [[1, 2], [3, 4], [5, 6]],
#             index=pd.Index(["a", "b", "c"], name="letters"),
#             columns=pd.Index(["black", "blue"], name="colors"),
#         )
#         Y = RandomVector.from_pandas(data=values, name="Y")
#         expected_rv = RandomVector.from_pandas(
#             data=pd.DataFrame(
#                 [[1, 2], [5, 6]],
#                 index=pd.Index(["a", "c"], name="letters"),
#                 columns=pd.Index(["black", "blue"], name="colors"),
#             ),
#             name="Y|event",
#         )
#         pd.testing.assert_frame_equal(Y[[0, 2]].data, expected_rv.data)
#         assert Y[[0, 2]].name == "Y|event"


# class TestGetComponents:

#     def test_get_components_with_single_index(self):
#         """Test the get_component method with a single index."""
#         Omega = SampleSpace.generate_default(size=3)
#         outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
#         X = RandomVector(outputs=outputs, domain=Omega, name="X")
#         X0 = RandomVariable(
#             outputs={"omega0": 1, "omega1": 3, "omega2": 5}, domain=Omega, name="X0"
#         )
#         X1 = RandomVariable(
#             outputs={"omega0": 2, "omega1": 4, "omega2": 6}, domain=Omega, name="X1"
#         )
#         assert X.get_components("X0") == X0
#         assert X.get_components("X1") == X1

#     def test_get_components_with_list(self):
#         """Test the get_components method with a list of indices."""
#         Omega = SampleSpace.generate_default(size=3)
#         outputs = {"omega0": (1, 2, 3), "omega1": (4, 5, 6), "omega2": (7, 8, 9)}
#         X = RandomVector(outputs=outputs, domain=Omega, name="X")
#         expected_rv = RandomVector(
#             outputs={"omega0": (1, 3), "omega1": (4, 6), "omega2": (7, 9)},
#             domain=Omega,
#             name="X_sub",
#         )
#         expected_rv.feature_index = FeatureIndex(["X0", "X2"])
#         components = X.get_components(["X0", "X2"])
#         assert components == expected_rv


# class TestArithmetic:

#     def test_add_two_random_vectors(self):
#         """Test adding two RandomVectors with same domain and feature_index."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="X",
#         )
#         Y = RandomVector(
#             outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
#             domain=Omega,
#             name="Y",
#         )
#         Z = X + Y
#         expected_values = pd.DataFrame(
#             [(11, 22), (33, 44), (55, 66)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X+Y)0", "(X+Y)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.data, expected_values)
#         assert Z.name == "(X+Y)"
#         assert Z.domain == Omega

#     def test_add_random_vector_and_scalar(self):
#         """Test adding a scalar to a RandomVector."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="X",
#         )
#         Z = X + 10
#         expected_values = pd.DataFrame(
#             [(11, 12), (13, 14), (15, 16)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X+10)0", "(X+10)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(X+10)"

#     def test_radd_scalar_and_random_vector(self):
#         """Test adding a RandomVector to a scalar (reverse add)."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="X",
#         )
#         Z = 10 + X
#         expected_values = pd.DataFrame(
#             [(11, 12), (13, 14), (15, 16)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X+10)0", "(X+10)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(X+10)"

#     def test_sub_two_random_vectors(self):
#         """Test subtracting two RandomVectors with same domain and feature_index."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
#             domain=Omega,
#             name="X",
#         )
#         Y = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="Y",
#         )
#         Z = X - Y
#         expected_values = pd.DataFrame(
#             [(9, 18), (27, 36), (45, 54)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X-Y)0", "(X-Y)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.data, expected_values)
#         assert Z.name == "(X-Y)"

#     def test_sub_random_vector_and_scalar(self):
#         """Test subtracting a scalar from a RandomVector."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
#             domain=Omega,
#             name="X",
#         )
#         Z = X - 5
#         expected_values = pd.DataFrame(
#             [(5, 15), (25, 35), (45, 55)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X-5)0", "(X-5)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(X-5)"

#     def test_rsub_scalar_and_random_vector(self):
#         """Test subtracting a RandomVector from a scalar (reverse sub)."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="X",
#         )
#         Z = 10 - X
#         expected_values = pd.DataFrame(
#             [(9, 8), (7, 6), (5, 4)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(10-X)0", "(10-X)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(10-X)"

#     def test_mul_two_random_vectors(self):
#         """Test multiplying two RandomVectors with same domain and feature_index."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (6, 7)},
#             domain=Omega,
#             name="X",
#         )
#         Y = RandomVector(
#             outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
#             domain=Omega,
#             name="Y",
#         )
#         Z = X * Y
#         expected_values = pd.DataFrame(
#             [(20, 60), (120, 200), (300, 420)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X*Y)0", "(X*Y)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.data, expected_values)
#         assert Z.name == "(X*Y)"

#     def test_mul_random_vector_and_scalar(self):
#         """Test multiplying a RandomVector by a scalar."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="X",
#         )
#         Z = X * 10
#         expected_values = pd.DataFrame(
#             [(10, 20), (30, 40), (50, 60)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X*10)0", "(X*10)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(X*10)"

#     def test_rmul_scalar_and_random_vector(self):
#         """Test multiplying a scalar by a RandomVector (reverse mul)."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="X",
#         )
#         Z = 10 * X
#         expected_values = pd.DataFrame(
#             [(10, 20), (30, 40), (50, 60)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X*10)0", "(X*10)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(X*10)"

#     def test_truediv_two_random_vectors(self):
#         """Test dividing two RandomVectors with same domain and feature_index."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (100, 200), "omega1": (300, 400), "omega2": (500, 600)},
#             domain=Omega,
#             name="X",
#         )
#         Y = RandomVector(
#             outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
#             domain=Omega,
#             name="Y",
#         )
#         Z = X / Y
#         expected_values = pd.DataFrame(
#             [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X/Y)0", "(X/Y)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.data, expected_values)
#         assert Z.name == "(X/Y)"

#     def test_truediv_random_vector_and_scalar(self):
#         """Test dividing a RandomVector by a scalar."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
#             domain=Omega,
#             name="X",
#         )
#         Z = X / 10
#         expected_values = pd.DataFrame(
#             [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X/10)0", "(X/10)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(X/10)"

#     def test_rtruediv_scalar_and_random_vector(self):
#         """Test dividing a scalar by a RandomVector (reverse div)."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (2, 4), "omega1": (5, 10), "omega2": (20, 25)},
#             domain=Omega,
#             name="X",
#         )
#         Z = 100 / X
#         expected_values = pd.DataFrame(
#             [(50.0, 25.0), (20.0, 10.0), (5.0, 4.0)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(100/X)0", "(100/X)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(100/X)"

#     def test_pow_two_random_vectors(self):
#         """Test exponentiating two RandomVectors with same domain and feature_index."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (6, 7)},
#             domain=Omega,
#             name="X",
#         )
#         Y = RandomVector(
#             outputs={"omega0": (2, 2), "omega1": (2, 2), "omega2": (2, 2)},
#             domain=Omega,
#             name="Y",
#         )
#         Z = X**Y
#         expected_values = pd.DataFrame(
#             [(4, 9), (16, 25), (36, 49)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X**Y)0", "(X**Y)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.data, expected_values)
#         assert Z.name == "(X**Y)"

#     def test_pow_random_vector_and_scalar(self):
#         """Test exponentiating a RandomVector by a scalar."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (6, 7)},
#             domain=Omega,
#             name="X",
#         )
#         Z = X**2
#         expected_values = pd.DataFrame(
#             [(4, 9), (16, 25), (36, 49)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(X**2)0", "(X**2)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(X**2)"

#     def test_rpow_scalar_and_random_vector(self):
#         """Test exponentiating a scalar by a RandomVector (reverse pow)."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (0, 1)},
#             domain=Omega,
#             name="X",
#         )
#         Z = 2**X
#         expected_values = pd.DataFrame(
#             [(4, 8), (16, 32), (1, 2)],
#             index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
#             columns=pd.Index(["(2**X)0", "(2**X)1"], name="feature"),
#         )
#         pd.testing.assert_frame_equal(Z.values, expected_values)
#         assert Z.name == "(2**X)"

#     def test_add_with_different_domains_raises_error(self):
#         """Test that adding RandomVectors with different domains raises ValueError."""
#         Omega1 = SampleSpace.generate_default(size=3, prefix="omega")
#         Omega2 = SampleSpace.generate_default(size=3, prefix="alpha")
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega1,
#             name="X",
#         )
#         Y = RandomVector(
#             outputs={"alpha0": (1, 2), "alpha1": (3, 4), "alpha2": (5, 6)},
#             domain=Omega2,
#             name="Y",
#         )
#         try:
#             Z = X + Y  # noqa: F841
#             raise AssertionError("Expected ValueError for different domains")
#         except ValueError as e:
#             assert "different domains" in str(e)

#     def test_add_with_non_random_vector_raises_error(self):
#         """Test that adding a non-RandomVector and non-scalar raises TypeError."""
#         Omega = SampleSpace.generate_default(size=3)
#         X = RandomVector(
#             outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
#             domain=Omega,
#             name="X",
#         )
#         try:
#             Z = X + "invalid"  # noqa: F841
#             raise AssertionError("Expected TypeError for invalid operand")
#         except TypeError as e:
#             assert "RandomVector or scalar" in str(e)
