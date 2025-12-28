import numpy as np
import pandas as pd
import pytest

from sigalg.core import (
    FeatureVector,
    Index,
    ProbabilityMeasure,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)


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
                id="2d_outputs_with_str_name",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 10, "omega1": 20, "omega2": 30},
                1,
                "Z",
                None,
                id="1d_outputs_with_str_name",
            ),
            pytest.param(
                [0, 1, 2, 3],
                {0: (1, 2), 1: (3, 4), 2: (5, 6), 3: (7, 8)},
                2,
                "default_name_flag",
                ["X0", "X1"],
                id="2d_outputs_with_default_name",
            ),
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 10, "omega1": 20, "omega2": 30},
                1,
                "default_name_flag",
                None,
                id="1d_outputs_with_default_name",
            ),
            pytest.param(
                [0, 1, 2],
                {0: (100, 150), 1: (200, 250), 2: (300, 350)},
                2,
                None,
                [0, 1],
                id="2d_outputs_with_none_name",
            ),
            pytest.param(
                [0, 1, 2],
                {0: 100, 1: 200, 2: 300},
                1,
                None,
                None,
                id="1d_outputs_with_none_name",
            ),
            pytest.param(
                ["a", "b", "c"],
                {"a": (0.1, 0.2), "b": (0.4, 0.5), "c": (0.7, 0.8)},
                2,
                42,
                [0, 1],
                id="2d_outputs_with_non_string_name",
            ),
            pytest.param(
                ["a", "b", "c"],
                {"a": 0.1, "b": 0.2, "c": 0.3},
                1,
                42,
                None,
                id="1d_outputs_with_non_string_name",
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

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == name

        if dimension == 1:
            expected_feature_index = None
            expected_data = pd.Series(data=outputs, index=domain.data, name=name)
            pd.testing.assert_series_equal(rv.data, expected_data)
        else:
            expected_feature_index = Index(
                indices=expected_feature_indices,
                name="feature_index",
                data_name="feature",
            )
            expected_data = pd.DataFrame.from_dict(
                data=outputs, orient="index", columns=expected_feature_index.data
            )
            expected_data.index.name = domain.data.name
            pd.testing.assert_frame_equal(rv.data, expected_data)

        assert rv.feature_index == expected_feature_index


class TestFromPandas:

    @pytest.mark.parametrize(
        "data, pandas_type, series_name, dimension, index_kwargs, columns_kwargs, name",
        [
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                "df",
                None,
                2,
                {"data": ["a", "b", "c"], "name": "letters"},
                {"data": ["black", "blue"], "name": "colors"},
                "Z",
                id="2d_df_custom_indices_with_str_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                None,
                1,
                {"data": ["a", "b", "c"], "name": "letters"},
                None,
                "Y",
                id="1d_series_custom_indices_with_str_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                None,
                1,
                {"data": ["a", "b", "c"], "name": "letters"},
                None,
                "Y",
                id="1d_df_custom_indices_with_str_name",
            ),
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                "df",
                None,
                2,
                None,
                None,
                "U",
                id="2d_df_default_indices_with_str_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                None,
                1,
                None,
                None,
                "U",
                id="1d_series_default_indices_with_str_name",
            ),
            pytest.param(
                [1, 2, 3],
                "df",
                None,
                1,
                None,
                None,
                "V",
                id="1d_df_default_indices_with_str_name",
            ),
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                "df",
                None,
                2,
                None,
                None,
                "default_name_flag",
                id="2d_df_default_indices_with_default_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                None,
                1,
                None,
                None,
                "default_name_flag",
                id="1d_series_default_indices_with_default_name",
            ),
            pytest.param(
                [1, 2, 3],
                "df",
                None,
                1,
                None,
                None,
                "default_name_flag",
                id="1d_df_default_indices_with_default_name",
            ),
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                "df",
                None,
                2,
                {"data": ["a", "b", "c"], "name": "letters"},
                {"data": ["black", "blue"], "name": "colors"},
                "default_name_flag",
                id="2d_df_custom_indices_with_default_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                None,
                1,
                {"data": ["a", "b", "c"], "name": "letters"},
                None,
                "default_name_flag",
                id="1d_series_custom_indices_with_default_name",
            ),
            pytest.param(
                [1, 2, 3],
                "df",
                None,
                1,
                {"data": ["a", "b", "c"], "name": "letters"},
                None,
                "default_name_flag",
                id="1d_df_custom_indices_with_default_name",
            ),
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                "df",
                None,
                2,
                None,
                None,
                None,
                id="2d_df_default_indices_with_none_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                None,
                1,
                None,
                None,
                None,
                id="1d_series_default_indices_with_none_name",
            ),
            pytest.param(
                [1, 2, 3],
                "df",
                None,
                1,
                None,
                None,
                None,
                id="1d_df_default_indices_with_none_name",
            ),
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                "df",
                None,
                2,
                None,
                None,
                42,
                id="2d_df_default_indices_with_int_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                None,
                1,
                None,
                None,
                42,
                id="1d_series_default_indices_with_int_name",
            ),
            pytest.param(
                [1, 2, 3],
                "df",
                None,
                1,
                None,
                None,
                42,
                id="1d_df_default_indices_with_int_name",
            ),
            pytest.param(
                [1, 2, 3],
                "series",
                "str_series_name",
                1,
                None,
                None,
                "U",
                id="1d_series_with_series_name",
            ),
        ],
    )
    def test_from_pandas(
        self,
        data,
        pandas_type,
        series_name,
        dimension,
        index_kwargs,
        columns_kwargs,
        name,
    ):
        """Test RandomVector.from_pandas method."""
        if name == "default_name_flag":
            name = "X"

        if index_kwargs is not None:
            index = pd.Index(**index_kwargs)
        else:
            index = None

        if columns_kwargs is not None:
            columns = pd.Index(**columns_kwargs)
        else:
            columns = None

        if pandas_type == "df":
            data = pd.DataFrame(data=data, index=index, columns=columns)
        else:
            data = pd.Series(data=data, index=index, name=series_name)

        if dimension == 1:
            expected_feature_index = None
        else:
            expected_feature_index = Index(
                indices=list(data.columns),
                name="feature_index",
                data_name=data.columns.name,
            )

        rv = RandomVector.from_pandas(data=data, name=name)
        expected_domain = SampleSpace(
            indices=list(data.index), name="Omega", data_name=data.index.name
        )

        assert rv.domain == expected_domain
        assert rv.feature_index == expected_feature_index
        assert rv.name == name

        if dimension == 1 and isinstance(data, pd.Series):
            pd.testing.assert_series_equal(rv.data, data)
        elif dimension == 1 and isinstance(data, pd.DataFrame):
            pd.testing.assert_series_equal(rv.data, data.iloc[:, 0])
        else:
            pd.testing.assert_frame_equal(rv.data, data)


class TestFromNumPy:

    def test_from_numpy(self):
        """Test RandomVector.from_numpy method."""
        arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
        arr_flat = np.array([10, 20, 30])
        arr_col = np.array([[10], [20], [30]])
        rv_2d = RandomVector.from_numpy(array=arr_2d, name="X")
        rv_flat = RandomVector.from_numpy(array=arr_flat, name="Y")
        rv_col = RandomVector.from_numpy(array=arr_col, name="Z")

        expected_domain = SampleSpace.from_pandas(
            data=pd.RangeIndex(start=0, stop=3), name="Omega"
        )

        expected_feature_index_2d = Index.from_pandas(
            data=pd.RangeIndex(start=0, stop=2), name="feature_index"
        )
        expected_feature_index_1d = None

        expected_data_2d = pd.DataFrame(data=arr_2d)
        expected_data_2d.index = expected_domain.data
        expected_data_2d.columns = expected_feature_index_2d.data

        expected_data_flat = pd.Series(
            data=arr_flat, index=expected_domain.data, name=0
        )
        expected_data_col = pd.Series(
            data=arr_col.flatten(), index=expected_domain.data, name=0
        )

        assert rv_2d.domain == expected_domain
        assert rv_flat.domain == expected_domain
        assert rv_col.domain == expected_domain

        assert rv_2d.feature_index == expected_feature_index_2d
        assert rv_flat.feature_index == expected_feature_index_1d
        assert rv_col.feature_index == expected_feature_index_1d

        assert rv_2d.name == "X"
        assert rv_flat.name == "Y"
        assert rv_col.name == "Z"

        pd.testing.assert_frame_equal(rv_2d.data, expected_data_2d)
        pd.testing.assert_series_equal(rv_flat.data, expected_data_flat)
        pd.testing.assert_series_equal(rv_col.data, expected_data_col)


class TestRangeAndRangeCounts:

    @pytest.mark.parametrize(
        "outputs, name, dimension, expected_range_outputs, expected_range_counts, expected_range_name, expected_rv_range_name, expected_range_feature_indices",
        [
            pytest.param(
                {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
                "X",
                2,
                {"x0": (1, 2), "x1": (3, 4)},
                {"x0": 1, "x1": 2},
                "range(X)",
                "X_range",
                ["X0", "X1"],
                id="2d_random_vector_with_str_name",
            ),
            pytest.param(
                {"omega0": 10, "omega1": 20, "omega2": 10},
                "Y",
                1,
                {"y0": 10, "y1": 20},
                {"y0": 2, "y1": 1},
                "range(Y)",
                "Y_range",
                None,
                id="1d_random_vector_with_str_name",
            ),
            pytest.param(
                {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
                42,
                2,
                {0: (1, 2), 1: (3, 4)},
                {0: 1, 1: 2},
                None,
                None,
                [0, 1],
                id="2d_random_vector_with_int_name",
            ),
            pytest.param(
                {"omega0": 1, "omega1": 1, "omega2": 2},
                42,
                1,
                {0: 1, 1: 2},
                {0: 2, 1: 1},
                None,
                None,
                None,
                id="1d_random_vector_with_int_name",
            ),
            pytest.param(
                {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
                None,
                2,
                {0: (1, 2), 1: (3, 4)},
                {0: 1, 1: 2},
                None,
                None,
                [0, 1],
                id="2d_random_vector_with_none_name",
            ),
            pytest.param(
                {"omega0": 1, "omega1": 1, "omega2": 2},
                None,
                1,
                {0: 1, 1: 2},
                {0: 2, 1: 1},
                None,
                None,
                None,
                id="1d_random_vector_with_none_name",
            ),
        ],
    )
    def test_range_and_range_counts(
        self,
        outputs,
        name,
        dimension,
        expected_range_outputs,
        expected_range_counts,
        expected_range_name,
        expected_rv_range_name,
        expected_range_feature_indices,
    ):
        """Test range property of RandomVector."""
        domain = SampleSpace(indices=outputs.keys(), name="Omega")
        rv = RandomVector(outputs=outputs, domain=domain, name=name)

        expected_range_domain = SampleSpace(
            indices=expected_range_outputs.keys(),
            name=expected_rv_range_name,
            data_name="output",
        )

        assert rv.range.domain == expected_range_domain
        assert rv.range.domain.name == expected_range_name
        assert rv.range.name == expected_rv_range_name

        expected_range_counts = pd.Series(data=expected_range_counts, name="count")
        expected_range_counts.index.name = "output"
        pd.testing.assert_series_equal(rv.range_counts, expected_range_counts)

        if dimension == 1:
            expected_range_data = pd.Series(data=expected_range_outputs, name=name)
            expected_feature_index = None
        else:
            expected_range_data = pd.DataFrame.from_dict(
                data=expected_range_outputs, orient="index"
            )
            expected_feature_index = Index(
                indices=expected_range_feature_indices,
                name="feature_index",
                data_name="feature",
            )
            expected_range_data.columns = expected_feature_index.data

        expected_range_data.index.name = "output"

        assert rv.feature_index == expected_feature_index

        if dimension == 1:
            pd.testing.assert_series_equal(rv.range.data, expected_range_data)
        else:
            pd.testing.assert_frame_equal(rv.range.data, expected_range_data)


class TestFeatureIndex:

    @pytest.fixture
    def domain(self):
        return SampleSpace.generate_default(size=3)

    @pytest.fixture
    def random_vector_2d(self, domain):
        outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
        return RandomVector(outputs=outputs, domain=domain, name="X")

    @pytest.fixture
    def random_vector_1d(self, domain):
        outputs = {"omega0": 10, "omega1": 20, "omega2": 30}
        return RandomVector(outputs=outputs, domain=domain, name="Y")

    def test_feature_index_property_of_2d_random_vector(self, random_vector_2d):
        """Test feature_index property of RandomVector."""
        expected_feature_index = Index(
            indices=["X0", "X1"], name="feature_index", data_name="feature"
        )

        assert random_vector_2d.feature_index == expected_feature_index
        assert random_vector_2d.feature_index.name == "feature_index"

    def test_feature_index_property_of_1d_random_vector(self, random_vector_1d):
        """Test feature_index property of 1D RandomVector."""
        assert random_vector_1d.feature_index is None

    def test_feature_index_setter_of_2d_random_vector(self, random_vector_2d):
        """Test setting feature_index of 2D RandomVector."""
        new_feature_index = Index(
            indices=["feature_a", "feature_b"],
            name="new_feature_index",
            data_name="new_feature",
        )
        random_vector_2d.feature_index = new_feature_index
        expected_data = pd.DataFrame(
            data=[(1, 2), (3, 4), (5, 6)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["feature_a", "feature_b"], name="new_feature"),
        )

        assert random_vector_2d.feature_index == new_feature_index
        assert random_vector_2d.feature_index.name == "new_feature_index"
        pd.testing.assert_frame_equal(random_vector_2d.data, expected_data)

    def test_feature_index_setter_of_1d_random_vector_raises(self, random_vector_1d):
        """Test that setting feature_index of 1D RandomVector raises an exception."""
        new_feature_index = Index(
            indices=["feature_a"],
            name="new_feature_index",
            data_name="new_feature",
        )
        with pytest.raises(ValueError):
            random_vector_1d.feature_index = new_feature_index

    def test_feature_index_setter_with_wrong_dimensions_raises(self, random_vector_2d):
        """Test that setting feature_index with wrong dimensions raises an exception."""
        new_feature_index = Index(
            indices=["feature_a", "feature_b", "feature_c"],
            name="new_feature_index",
            data_name="new_feature",
        )
        with pytest.raises(ValueError):
            random_vector_2d.feature_index = new_feature_index


class TestSigmaAlgebra:

    def test_sigma_algebra_property(self):
        """Test sigma_algebra property of RandomVector."""
        domain = SampleSpace.generate_default(size=3)
        outputs_2d = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
        outputs_1d = {"omega0": 10, "omega1": 20, "omega2": 30}
        rv_2d = RandomVector(outputs=outputs_2d, domain=domain, name="X")
        rv_1d = RandomVector(outputs=outputs_1d, domain=domain, name="Y")
        expected_sigma_algebra_2d = SigmaAlgebra(
            sample_id_to_atom_id=outputs_2d,
            sample_space=domain,
            name="sigma(X)",
        )
        expected_sigma_algebra_1d = SigmaAlgebra(
            sample_id_to_atom_id=outputs_1d,
            sample_space=domain,
            name="sigma(Y)",
        )

        assert rv_2d.sigma_algebra == expected_sigma_algebra_2d
        assert rv_1d.sigma_algebra == expected_sigma_algebra_1d


class TestIterFeatures:

    def test_iter_features_of_2d_random_vector(self):
        """Test iter_features method of 2D RandomVector."""
        domain = SampleSpace.generate_default(size=3)
        outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)}
        rv = RandomVector(outputs=outputs, domain=domain, name="X")

        expected_features = {
            "omega0": FeatureVector(
                data=pd.Series(
                    [1, 2],
                    index=pd.Index(["X0", "X1"], name="feature"),
                    name="omega0",
                )
            ),
            "omega1": FeatureVector(
                data=pd.Series(
                    [3, 4],
                    index=pd.Index(["X0", "X1"], name="feature"),
                    name="omega1",
                )
            ),
            "omega2": FeatureVector(
                data=pd.Series(
                    [5, 6],
                    index=pd.Index(["X0", "X1"], name="feature"),
                    name="omega2",
                )
            ),
        }

        for sample_idx, feature_vector in rv.iter_features():
            pd.testing.assert_series_equal(
                feature_vector.data, expected_features[sample_idx].data
            )

    def test_iter_features_of_1d_random_vector(self):
        """Test iter_features method of 1D RandomVector."""
        domain = SampleSpace.generate_default(size=3)
        outputs = {"omega0": 10, "omega1": 20, "omega2": 30}
        rv = RandomVector(outputs=outputs, domain=domain, name="Y")

        expected_features = {
            "omega0": 10,
            "omega1": 20,
            "omega2": 30,
        }

        for sample_idx, feature in rv.iter_features():
            assert feature == expected_features[sample_idx]


class TestPushforward:

    @pytest.fixture
    def X(self):
        domain = SampleSpace.generate_default(size=3)
        outputs = {"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)}
        return RandomVector(outputs=outputs, domain=domain, name="X")

    def test_pushforward_method_with_custom_measure(self, X):
        """Test pushforward method of RandomVector."""
        probabilities = {"omega0": 0.2, "omega1": 0.5, "omega2": 0.3}
        probability_measure = ProbabilityMeasure(
            probabilities=probabilities, sample_space=X.domain
        )
        P_X = X.pushforward(probability_measure).probability_measure

        expected_probability_measure = ProbabilityMeasure(
            probabilities={"x0": 0.2, "x1": 0.8},
            sample_space=X.range.domain,
            name="P_X",
        )
        assert P_X == expected_probability_measure
        assert P_X.name == "P_X"

    def test_pushforward_method_with_default_measure(self, X):
        """Test pushforward method of RandomVector with default (i.e, uniform) measure."""
        P_X = X.pushforward().probability_measure

        expected_probability_measure = ProbabilityMeasure(
            probabilities={"x0": 1 / 3, "x1": 2 / 3},
            sample_space=X.range.domain,
            name="P_X",
        )
        assert P_X == expected_probability_measure
        assert P_X.name == "P_X"


class TestAddProbabilityMeasureToDomain:

    def test_add_probability_measure_to_domain(self):
        """Test adding a ProbabilityMeasure to the domain of a RandomVector."""
        domain = SampleSpace.generate_default(size=4)
        outputs = {
            "omega0": (0, 0),
            "omega1": (0, 1),
            "omega2": (1, 0),
            "omega3": (1, 1),
        }
        X = RandomVector(outputs=outputs, domain=domain, name="X")

        def pmf(feature_vector):
            v0, v1 = feature_vector
            return 0.75**v0 * 0.25 ** (1 - v0) * 0.6**v1 * 0.4 ** (1 - v1)

        fps = X.add_probability_measure_to_domain(pmf=pmf)

        expected_probability_measure = ProbabilityMeasure(
            probabilities={
                "omega0": 0.25 * 0.4,
                "omega1": 0.25 * 0.6,
                "omega2": 0.75 * 0.4,
                "omega3": 0.75 * 0.6,
            },
            sample_space=domain,
        )

        assert fps.sample_space == domain
        assert fps.feature_embedding == X
        assert fps.probability_measure == expected_probability_measure


class TestCallMethod:

    @pytest.fixture
    def domain(self):
        return SampleSpace(indices=["s0", "s1", "s2"], name="S")

    @pytest.fixture
    def random_vector_2d(self, domain):
        outputs = {"s0": (1, 2), "s1": (3, 4), "s2": (5, 6)}
        return RandomVector(outputs=outputs, domain=domain, name="X")

    @pytest.fixture
    def random_vector_1d(self, domain):
        outputs = {"s0": 10, "s1": 20, "s2": 30}
        return RandomVector(outputs=outputs, domain=domain, name="Y")

    def test_call_method_on_sample_index(self, random_vector_2d, random_vector_1d):
        """Test calling RandomVector on a single sample index."""
        expected_2d_features = FeatureVector(
            data=pd.Series(
                [1, 2], index=pd.Index(["X0", "X1"], name="feature"), name="s0"
            )
        )
        expected_1d_features = 10
        actual_2d_features = random_vector_2d("s0")
        actual_1d_features = random_vector_1d("s0")

        assert isinstance(actual_2d_features, FeatureVector)
        pd.testing.assert_series_equal(
            actual_2d_features.data, expected_2d_features.data
        )
        assert actual_1d_features == expected_1d_features

    def test_call_method_on_sample_indices(self, random_vector_2d, random_vector_1d):
        """Test calling RandomVector on a list of sample indices."""
        expected_2d_rv = RandomVector.from_pandas(
            data=pd.DataFrame(
                [(1, 2), (5, 6)],
                index=pd.Index(["s0", "s2"], name="sample"),
                columns=pd.Index(["X0", "X1"], name="feature"),
            ),
            name="X|event",
        )
        expected_1d_rv = RandomVector.from_pandas(
            data=pd.Series(
                [10, 30],
                index=pd.Index(["s0", "s2"], name="sample"),
                name="Y",
            )
        )
        actual_2d_rv = random_vector_2d(["s0", "s2"])
        actual_1d_rv = random_vector_1d(["s0", "s2"])

        pd.testing.assert_frame_equal(actual_2d_rv.data, expected_2d_rv.data)
        pd.testing.assert_series_equal(actual_1d_rv.data, expected_1d_rv.data)
        assert actual_2d_rv.name == "X|event"
        assert actual_1d_rv.name == "Y|event"

    def test_call_method_on_event(self, random_vector_2d, random_vector_1d, domain):
        """Test calling RandomVector on an Event."""
        B = domain.get_event(["s0", "s2"], name="B")
        expected_2d_rv = RandomVector.from_pandas(
            data=pd.DataFrame(
                [(1, 2), (5, 6)],
                index=pd.Index(["s0", "s2"], name="sample"),
                columns=pd.Index(["X0", "X1"], name="feature"),
            ),
            name="X|B",
        )
        expected_1d_rv = RandomVector.from_pandas(
            data=pd.Series(
                [10, 30],
                index=pd.Index(["s0", "s2"], name="sample"),
                name="Y",
            ),
            name="Y|B",
        )
        actual_2d_rv = random_vector_2d(B)
        actual_1d_rv = random_vector_1d(B)

        pd.testing.assert_frame_equal(actual_2d_rv.data, expected_2d_rv.data)
        pd.testing.assert_series_equal(actual_1d_rv.data, expected_1d_rv.data)
        assert actual_2d_rv.name == "X|B"
        assert actual_1d_rv.name == "Y|B"

    def test_invalid_input_raises(self):
        """Test that invalid inputs raise appropriate exceptions."""
        outputs = {"s0": (1, 2), "s1": (3, 4), "s2": (5, 6)}
        domain = SampleSpace(indices=["s0", "s1", "s2"], name="Omega")
        X = RandomVector(outputs=outputs, domain=domain, name="X")

        with pytest.raises(TypeError):
            X({"s0": 1})
        with pytest.raises(KeyError):
            X(3.14)
        with pytest.raises(KeyError):
            X(["s0", "s3"])
        with pytest.raises(ValueError):
            other_domain = SampleSpace(indices=["t0", "t1", "t2"], name="Theta")
            A = other_domain.get_event(["t0", "t2"])
            X(A)


# class TestToRandomVariable:

#     def test_to_random_vector(self):
#         """Test conversion of RandomVariable to RandomVector."""
#         outputs = {"omega0": 1, "omega1": 2, "omega2": 3}
#         domain = SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega")
#         X = RandomVector(outputs=outputs, domain=domain, name="X")
#         random_variable = X.to_random_variable()
#         expected_data = pd.Series(data=[1, 2, 3], index=domain.data, name="X")

#         pd.testing.assert_series_equal(random_variable.data, expected_data)
#         assert random_variable.name == "X"


class TestArithmetic:

    def test_add_two_random_vectors(self):
        """Test adding two RandomVectors with same domain and feature_index."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="X",
        )
        Y = RandomVector(
            outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
            domain=Omega,
            name="Y",
        )
        Z = X + Y
        expected_data = pd.DataFrame(
            [(11, 22), (33, 44), (55, 66)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X+Y)0", "(X+Y)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_vector_and_scalar(self):
        """Test adding a scalar to a RandomVector."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="X",
        )
        Z = X + 10
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X+10)0", "(X+10)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_vector(self):
        """Test adding a RandomVector to a scalar (reverse add)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="X",
        )
        Z = 10 + X
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X+10)0", "(X+10)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+10)"

    def test_sub_two_random_vectors(self):
        """Test subtracting two RandomVectors with same domain and feature_index."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
            domain=Omega,
            name="X",
        )
        Y = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="Y",
        )
        Z = X - Y
        expected_values = pd.DataFrame(
            [(9, 18), (27, 36), (45, 54)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X-Y)0", "(X-Y)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_vector_and_scalar(self):
        """Test subtracting a scalar from a RandomVector."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
            domain=Omega,
            name="X",
        )
        Z = X - 5
        expected_data = pd.DataFrame(
            [(5, 15), (25, 35), (45, 55)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X-5)0", "(X-5)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_vector(self):
        """Test subtracting a RandomVector from a scalar (reverse sub)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="X",
        )
        Z = 10 - X
        expected_data = pd.DataFrame(
            [(9, 8), (7, 6), (5, 4)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(10-X)0", "(10-X)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10-X)"

    def test_mul_two_random_vectors(self):
        """Test multiplying two RandomVectors with same domain and feature_index."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (6, 7)},
            domain=Omega,
            name="X",
        )
        Y = RandomVector(
            outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
            domain=Omega,
            name="Y",
        )
        Z = X * Y
        expected_data = pd.DataFrame(
            [(20, 60), (120, 200), (300, 420)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X*Y)0", "(X*Y)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*Y)"

    def test_mul_random_vector_and_scalar(self):
        """Test multiplying a RandomVector by a scalar."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="X",
        )
        Z = X * 10
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X*10)0", "(X*10)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_vector(self):
        """Test multiplying a scalar by a RandomVector (reverse mul)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="X",
        )
        Z = 10 * X
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X*10)0", "(X*10)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*10)"

    def test_truediv_two_random_vectors(self):
        """Test dividing two RandomVectors with same domain and feature_index."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (100, 200), "omega1": (300, 400), "omega2": (500, 600)},
            domain=Omega,
            name="X",
        )
        Y = RandomVector(
            outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
            domain=Omega,
            name="Y",
        )
        Z = X / Y
        expected_data = pd.DataFrame(
            [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X/Y)0", "(X/Y)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/Y)"

    def test_truediv_random_vector_and_scalar(self):
        """Test dividing a RandomVector by a scalar."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (10, 20), "omega1": (30, 40), "omega2": (50, 60)},
            domain=Omega,
            name="X",
        )
        Z = X / 10
        expected_data = pd.DataFrame(
            [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X/10)0", "(X/10)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_vector(self):
        """Test dividing a scalar by a RandomVector (reverse div)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (2, 4), "omega1": (5, 10), "omega2": (20, 25)},
            domain=Omega,
            name="X",
        )
        Z = 100 / X
        expected_data = pd.DataFrame(
            [(50.0, 25.0), (20.0, 10.0), (5.0, 4.0)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(100/X)0", "(100/X)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(100/X)"

    def test_pow_two_random_vectors(self):
        """Test exponentiating two RandomVectors with same domain and feature_index."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (6, 7)},
            domain=Omega,
            name="X",
        )
        Y = RandomVector(
            outputs={"omega0": (2, 2), "omega1": (2, 2), "omega2": (2, 2)},
            domain=Omega,
            name="Y",
        )
        Z = X**Y
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X**Y)0", "(X**Y)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self):
        """Test exponentiating a RandomVector by a scalar."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (6, 7)},
            domain=Omega,
            name="X",
        )
        Z = X**2
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(X**2)0", "(X**2)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self):
        """Test exponentiating a scalar by a RandomVector (reverse pow)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (2, 3), "omega1": (4, 5), "omega2": (0, 1)},
            domain=Omega,
            name="X",
        )
        Z = 2**X
        expected_data = pd.DataFrame(
            [(4, 8), (16, 32), (1, 2)],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["(2**X)0", "(2**X)1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(2**X)"

    def test_add_with_different_domains_raises_error(self):
        """Test that adding RandomVectors with different domains raises ValueError."""
        Omega1 = SampleSpace.generate_default(size=3, prefix="omega")
        Omega2 = SampleSpace.generate_default(size=3, prefix="alpha")
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega1,
            name="X",
        )
        Y = RandomVector(
            outputs={"alpha0": (1, 2), "alpha1": (3, 4), "alpha2": (5, 6)},
            domain=Omega2,
            name="Y",
        )
        try:
            Z = X + Y  # noqa: F841
            raise AssertionError("Expected ValueError for different domains")
        except ValueError as e:
            assert "different domains" in str(e)

    def test_add_with_non_random_vector_raises_error(self):
        """Test that adding a non-RandomVector and non-scalar raises TypeError."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVector(
            outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (5, 6)},
            domain=Omega,
            name="X",
        )
        try:
            Z = X + "invalid"  # noqa: F841
            raise AssertionError("Expected TypeError for invalid operand")
        except TypeError as e:
            assert "RandomVector or scalar" in str(e)


class TestArithmeticWithRandomVariable:

    def test_add_two_random_variables(self):
        """Test adding two RandomVariables with same domain."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="X",
        )
        Y = RandomVariable(
            outputs={"omega0": 10, "omega1": 30, "omega2": 50},
            domain=Omega,
            name="Y",
        )
        Z = X + Y
        expected_values = pd.Series(
            [11, 33, 55],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X+Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_variable_and_scalar(self):
        """Test adding a scalar to a RandomVariable."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="X",
        )
        Z = X + 10
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X+10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_variable(self):
        """Test adding a RandomVariable to a scalar (reverse add)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="X",
        )
        Z = 10 + X
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X+10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+10)"

    def test_sub_two_random_variables(self):
        """Test subtracting two RandomVariables with same domain."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 10, "omega1": 30, "omega2": 50},
            domain=Omega,
            name="X",
        )
        Y = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="Y",
        )
        Z = X - Y
        expected_values = pd.Series(
            [9, 27, 45],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X-Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_variable_and_scalar(self):
        """Test subtracting a scalar from a RandomVariable."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 10, "omega1": 30, "omega2": 50},
            domain=Omega,
            name="X",
        )
        Z = X - 5
        expected_values = pd.Series(
            [5, 25, 45],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X-5)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_variable(self):
        """Test subtracting a RandomVariable from a scalar (reverse sub)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="X",
        )
        Z = 10 - X
        expected_values = pd.Series(
            [9, 7, 5],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(10-X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10-X)"

    def test_mul_two_random_variables(self):
        """Test multiplying two RandomVariables with same domain."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6},
            domain=Omega,
            name="X",
        )
        Y = RandomVariable(
            outputs={"omega0": 10, "omega1": 30, "omega2": 50},
            domain=Omega,
            name="Y",
        )
        Z = X * Y
        expected_values = pd.Series(
            [20, 120, 300],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X*Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*Y)"

    def test_mul_random_variable_and_scalar(self):
        """Test multiplying a RandomVariable by a scalar."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="X",
        )
        Z = X * 10
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X*10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_variable(self):
        """Test multiplying a scalar by a RandomVariable (reverse mul)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="X",
        )
        Z = 10 * X
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X*10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*10)"

    def test_truediv_two_random_variables(self):
        """Test dividing two RandomVariables with same domain."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 100, "omega1": 300, "omega2": 500},
            domain=Omega,
            name="X",
        )
        Y = RandomVariable(
            outputs={"omega0": 10, "omega1": 30, "omega2": 50},
            domain=Omega,
            name="Y",
        )
        Z = X / Y
        expected_values = pd.Series(
            [10.0, 10.0, 10.0],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X/Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/Y)"

    def test_truediv_random_variable_and_scalar(self):
        """Test dividing a RandomVariable by a scalar."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 10, "omega1": 30, "omega2": 50},
            domain=Omega,
            name="X",
        )
        Z = X / 10
        expected_values = pd.Series(
            [1.0, 3.0, 5.0],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X/10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_variable(self):
        """Test dividing a scalar by a RandomVariable (reverse div)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 2, "omega1": 5, "omega2": 20},
            domain=Omega,
            name="X",
        )
        Z = 100 / X
        expected_values = pd.Series(
            [50.0, 20.0, 5.0],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(100/X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(100/X)"

    def test_pow_two_random_variables(self):
        """Test exponentiating two RandomVariables with same domain."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6},
            domain=Omega,
            name="X",
        )
        Y = RandomVariable(
            outputs={"omega0": 2, "omega1": 2, "omega2": 2},
            domain=Omega,
            name="Y",
        )
        Z = X**Y
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X**Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**Y)"

    def test_pow_random_variable_and_scalar(self):
        """Test exponentiating a RandomVariable by a scalar."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6},
            domain=Omega,
            name="X",
        )
        Z = X**2
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(X**2)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_variable(self):
        """Test exponentiating a scalar by a RandomVariable (reverse pow)."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 0},
            domain=Omega,
            name="X",
        )
        Z = 2**X
        expected_values = pd.Series(
            [4, 16, 1],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            name="(2**X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(2**X)"

    def test_add_with_different_domains_raises_error(self):
        """Test that adding RandomVariables with different domains raises ValueError."""
        Omega1 = SampleSpace.generate_default(size=3, prefix="omega")
        Omega2 = SampleSpace.generate_default(size=3, prefix="alpha")
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega1,
            name="X",
        )
        Y = RandomVariable(
            outputs={"alpha0": 1, "alpha1": 3, "alpha2": 5},
            domain=Omega2,
            name="Y",
        )
        try:
            Z = X + Y  # noqa: F841
            raise AssertionError("Expected ValueError for different domains")
        except ValueError as e:
            assert "different domains" in str(e)

    def test_add_with_non_random_variable_raises_error(self):
        """Test that adding a non-RandomVariable and non-scalar raises TypeError."""
        Omega = SampleSpace.generate_default(size=3)
        X = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=Omega,
            name="X",
        )
        try:
            Z = X + "invalid"  # noqa: F841
            raise AssertionError("Expected TypeError for invalid operand")
        except TypeError as e:
            assert "RandomVector or scalar" in str(e)
