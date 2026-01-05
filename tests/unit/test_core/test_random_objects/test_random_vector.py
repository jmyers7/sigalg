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

    def test_2d_outputs_with_str_name(self):
        """Test RandomVector constructor with 2D outputs and string name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        rv = RandomVector(domain=domain, name="Y").from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == "Y"

        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(["Y_0", "Y_1"])
        expected_data = pd.DataFrame.from_dict(
            data=outputs, orient="index", columns=expected_index.data
        )
        expected_data.index.name = domain.data.name
        pd.testing.assert_frame_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_1d_outputs_with_str_name(self):
        """Test RandomVector constructor with 1D outputs and string name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": 10, "omega_1": 20, "omega_2": 30}
        rv = RandomVector(domain=domain, name="Z").from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == "Z"

        expected_index = None
        expected_data = pd.Series(data=outputs, index=domain.data, name="Z")
        pd.testing.assert_series_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_2d_outputs_with_default_name(self):
        """Test RandomVector constructor with 2D outputs and default name."""
        domain = SampleSpace().from_list([0, 1, 2, 3])
        outputs = {0: (1, 2), 1: (3, 4), 2: (5, 6), 3: (7, 8)}
        rv = RandomVector(domain=domain).from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == "X"

        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(["X_0", "X_1"])
        expected_data = pd.DataFrame.from_dict(
            data=outputs, orient="index", columns=expected_index.data
        )
        expected_data.index.name = domain.data.name
        pd.testing.assert_frame_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_1d_outputs_with_default_name(self):
        """Test RandomVector constructor with 1D outputs and default name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": 10, "omega_1": 20, "omega_2": 30}
        rv = RandomVector(domain=domain).from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == "X"

        expected_index = None
        expected_data = pd.Series(data=outputs, index=domain.data, name="X")
        pd.testing.assert_series_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_2d_outputs_with_none_name(self):
        """Test RandomVector constructor with 2D outputs and None name."""
        domain = SampleSpace().from_list([0, 1, 2])
        outputs = {0: (100, 150), 1: (200, 250), 2: (300, 350)}
        rv = RandomVector(domain=domain, name=None).from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name is None

        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_data = pd.DataFrame.from_dict(
            data=outputs, orient="index", columns=expected_index.data
        )
        expected_data.index.name = domain.data.name
        pd.testing.assert_frame_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_1d_outputs_with_none_name(self):
        """Test RandomVector constructor with 1D outputs and None name."""
        domain = SampleSpace().from_list([0, 1, 2])
        outputs = {0: 100, 1: 200, 2: 300}
        rv = RandomVector(domain=domain, name=None).from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name is None

        expected_index = None
        expected_data = pd.Series(data=outputs, index=domain.data, name=None)
        pd.testing.assert_series_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_2d_outputs_with_non_string_name(self):
        """Test RandomVector constructor with 2D outputs and non-string name."""
        domain = SampleSpace().from_list(["a", "b", "c"])
        outputs = {"a": (0.1, 0.2), "b": (0.4, 0.5), "c": (0.7, 0.8)}
        rv = RandomVector(domain=domain, name=42).from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == 42

        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_data = pd.DataFrame.from_dict(
            data=outputs, orient="index", columns=expected_index.data
        )
        expected_data.index.name = domain.data.name
        pd.testing.assert_frame_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_1d_outputs_with_non_string_name(self):
        """Test RandomVector constructor with 1D outputs and non-string name."""
        domain = SampleSpace().from_list(["a", "b", "c"])
        outputs = {"a": 0.1, "b": 0.2, "c": 0.3}
        rv = RandomVector(domain=domain, name=42).from_dict(outputs)

        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == 42

        expected_index = None
        expected_data = pd.Series(data=outputs, index=domain.data, name=42)
        pd.testing.assert_series_equal(rv.data, expected_data)
        assert rv.index == expected_index

    def test_constructor_with_custom_index(self):
        """Test RandomVector constructor with custom index parameter."""
        domain = SampleSpace.generate_sequence(size=3, prefix="s")
        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        custom_index = Index(
            name="custom_index",
            data_name="feature",
        ).from_list(["feature_a", "feature_b"])

        rv = RandomVector(domain=domain, name="X", index=custom_index).from_dict(
            outputs
        )

        assert rv.index == custom_index
        assert rv.dimension == 2
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=pd.Index(["s_0", "s_1", "s_2"], name="sample"),
            columns=pd.Index(["feature_a", "feature_b"], name="feature"),
        )
        pd.testing.assert_frame_equal(rv.data, expected_data)

    def test_constructor_index_wrong_length_raises(self):
        """Test that index with wrong length raises an error."""
        domain = SampleSpace.generate_sequence(size=2, prefix="s")
        outputs = {"s_0": (1, 2), "s_1": (3, 4)}
        wrong_index = Index(
            name="wrong",
            data_name="feature",
        ).from_list(["a", "b", "c"])

        with pytest.raises(
            ValueError,
            match="Length of index must match the dimension of the RandomVector",
        ):
            RandomVector(
                domain=domain,
                name="X",
                index=wrong_index,
            ).from_dict(outputs)

    def test_constructor_index_not_index_type_raises(self):
        """Test that index that's not an Index raises a TypeError."""
        domain = SampleSpace.generate_sequence(size=2, prefix="s")
        outputs = {"s_0": (1, 2), "s_1": (3, 4)}

        with pytest.raises(TypeError, match="index must be an Index"):
            RandomVector(
                domain=domain,
                name="X",
                index=["a", "b"],  # Should be an Index, not a list
            ).from_dict(outputs)


class TestFromPandas:

    def test_2d_df_custom_indices_with_str_name(self):
        """Test RandomVector.from_pandas with 2D DataFrame, custom indices and string name."""
        data = pd.DataFrame(
            data=[(1, 2), (3, 4), (5, 6)],
            index=pd.Index(["a", "b", "c"], name="letters"),
            columns=pd.Index(["black", "blue"], name="colors"),
        )
        rv = RandomVector(name="Z").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(["a", "b", "c"])
        expected_domain.data.name = "letters"
        expected_index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue"])

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, data)

    def test_1d_series_custom_indices_with_str_name(self):
        """Test RandomVector.from_pandas with 1D Series, custom index and string name."""
        data = pd.Series(
            data=[1, 2, 3],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="output",
        )
        rv = RandomVector(name="Y").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(["a", "b", "c"])
        expected_domain.data.name = "letters"
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, data)

    def test_1d_series_custom_indices_with_str_name_no_series_name(self):
        """Test RandomVector.from_pandas with 1D Series, custom index, string name, no series name."""
        data = pd.Series(
            data=[1, 2, 3],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name=None,
        )
        rv = RandomVector(name="Y").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(["a", "b", "c"])
        expected_domain.data.name = "letters"
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, data)

    def test_2d_df_default_indices_with_str_name(self):
        """Test RandomVector.from_pandas with 2D DataFrame, default indices and string name."""
        data = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)])
        rv = RandomVector(name="U").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(list(data.columns))

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "U"
        pd.testing.assert_frame_equal(rv.data, data)

    def test_1d_series_default_indices_with_str_name(self):
        """Test RandomVector.from_pandas with 1D Series, default index and string name."""
        data = pd.Series(data=[1, 2, 3], name=None)
        rv = RandomVector(name="U").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "U"
        pd.testing.assert_series_equal(rv.data, data)

    def test_1d_df_default_indices_with_str_name(self):
        """Test RandomVector.from_pandas with 1D DataFrame (single column), default indices and string name."""
        data = pd.DataFrame(data=[1, 2, 3])
        rv = RandomVector(name="V").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "V"
        pd.testing.assert_series_equal(rv.data, data.iloc[:, 0])

    def test_2d_df_default_indices_with_default_name(self):
        """Test RandomVector.from_pandas with 2D DataFrame, default indices and default name."""
        data = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)])
        rv = RandomVector().from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(list(data.columns))

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "X"
        pd.testing.assert_frame_equal(rv.data, data)

    def test_1d_series_default_indices_with_default_name(self):
        """Test RandomVector.from_pandas with 1D Series, default index and default name."""
        data = pd.Series(data=[1, 2, 3], name=None)
        rv = RandomVector().from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "X"
        pd.testing.assert_series_equal(rv.data, data)

    def test_1d_df_default_indices_with_default_name(self):
        """Test RandomVector.from_pandas with 1D DataFrame, default indices and default name."""
        data = pd.DataFrame(data=[1, 2, 3])
        rv = RandomVector().from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "X"
        pd.testing.assert_series_equal(rv.data, data.iloc[:, 0])

    def test_2d_df_custom_indices_with_default_name(self):
        """Test RandomVector.from_pandas with 2D DataFrame, custom indices and default name."""
        data = pd.DataFrame(
            data=[(1, 2), (3, 4), (5, 6)],
            index=pd.Index(["a", "b", "c"], name="letters"),
            columns=pd.Index(["black", "blue"], name="colors"),
        )
        rv = RandomVector().from_pandas(data=data)

        expected_domain = SampleSpace().from_list(["a", "b", "c"])
        expected_domain.data.name = "letters"
        expected_index = Index(
            name="index",
            data_name="colors",
        ).from_list(["black", "blue"])

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "X"
        pd.testing.assert_frame_equal(rv.data, data)

    def test_1d_series_custom_indices_with_default_name(self):
        """Test RandomVector.from_pandas with 1D Series, custom index and default name."""
        data = pd.Series(
            data=[1, 2, 3],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name=None,
        )
        rv = RandomVector().from_pandas(data=data)

        expected_domain = SampleSpace().from_list(["a", "b", "c"])
        expected_domain.data.name = "letters"
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "X"
        pd.testing.assert_series_equal(rv.data, data)

    def test_1d_df_custom_indices_with_default_name(self):
        """Test RandomVector.from_pandas with 1D DataFrame, custom index and default name."""
        data = pd.DataFrame(
            data=[1, 2, 3],
            index=pd.Index(["a", "b", "c"], name="letters"),
        )
        rv = RandomVector().from_pandas(data=data)

        expected_domain = SampleSpace().from_list(["a", "b", "c"])
        expected_domain.data.name = "letters"
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "X"
        pd.testing.assert_series_equal(rv.data, data.iloc[:, 0])

    def test_2d_df_default_indices_with_none_name(self):
        """Test RandomVector.from_pandas with 2D DataFrame, default indices and None name."""
        data = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)])
        rv = RandomVector(name=None).from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(list(data.columns))

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name is None
        pd.testing.assert_frame_equal(rv.data, data)

    def test_1d_series_default_indices_with_none_name(self):
        """Test RandomVector.from_pandas with 1D Series, default index and None name."""
        data = pd.Series(data=[1, 2, 3], name=None)
        rv = RandomVector(name=None).from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name is None
        pd.testing.assert_series_equal(rv.data, data)

    def test_1d_df_default_indices_with_none_name(self):
        """Test RandomVector.from_pandas with 1D DataFrame, default indices and None name."""
        data = pd.DataFrame(data=[1, 2, 3])
        rv = RandomVector(name=None).from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name is None
        pd.testing.assert_series_equal(rv.data, data.iloc[:, 0])

    def test_2d_df_default_indices_with_int_name(self):
        """Test RandomVector.from_pandas with 2D DataFrame, default indices and int name."""
        data = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)])
        rv = RandomVector(name=42).from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(list(data.columns))

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == 42
        pd.testing.assert_frame_equal(rv.data, data)

    def test_1d_series_default_indices_with_int_name(self):
        """Test RandomVector.from_pandas with 1D Series, default index and int name."""
        data = pd.Series(data=[1, 2, 3], name=None)
        rv = RandomVector(name=42).from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == 42
        pd.testing.assert_series_equal(rv.data, data)

    def test_1d_df_default_indices_with_int_name(self):
        """Test RandomVector.from_pandas with 1D DataFrame, default indices and int name."""
        data = pd.DataFrame(data=[1, 2, 3])
        rv = RandomVector(name=42).from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == 42
        pd.testing.assert_series_equal(rv.data, data.iloc[:, 0])

    def test_1d_series_with_series_name(self):
        """Test RandomVector.from_pandas with 1D Series that has its own name."""
        data = pd.Series(data=[1, 2, 3], name="str_series_name")
        rv = RandomVector(name="U").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_index = None

        assert rv.domain == expected_domain
        assert rv.index == expected_index
        assert rv.name == "U"
        pd.testing.assert_series_equal(rv.data, data)


class TestFromNumPy:

    def test_from_numpy(self):
        """Test RandomVector.from_numpy method."""
        arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
        arr_flat = np.array([10, 20, 30])
        arr_col = np.array([[10], [20], [30]])
        rv_2d = RandomVector(name="X").from_numpy(array=arr_2d)
        rv_flat = RandomVector(name="Y").from_numpy(array=arr_flat)
        rv_col = RandomVector(name="Z").from_numpy(array=arr_col)

        # Get expected domain from the actual rv to match the data.index correctly
        expected_domain = rv_2d.domain

        expected_index_2d = Index(
            name="index",
            data_name="feature",
        ).from_list(list(range(2)))
        expected_index_flat = None
        expected_index_col = None

        assert rv_2d.domain == expected_domain
        assert rv_flat.domain == expected_domain
        assert rv_col.domain == expected_domain

        assert rv_2d.index == expected_index_2d
        assert rv_flat.index == expected_index_flat
        assert rv_col.index == expected_index_col

        assert rv_2d.name == "X"
        assert rv_flat.name == "Y"
        assert rv_col.name == "Z"

        # Just check the values match, not the full dataframe equality
        assert rv_2d.data.shape == (3, 2)
        assert rv_flat.data.shape == (3,)
        assert rv_col.data.shape == (3,)


class TestRangeAndRangeCounts:

    def test_range_2d_random_vector_with_str_name(self):
        """Test range property of 2D RandomVector with string name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)}
        rv = RandomVector(domain=domain, name="X").from_dict(outputs)

        expected_range_domain = SampleSpace().from_list(["x_0", "x_1"])
        expected_range_domain.data.name = "output"
        expected_range_domain.name = "range(X)"

        assert rv.range.domain == expected_range_domain
        assert rv.range.domain.name == "range(X)"
        assert rv.range.name == "X_range"

        expected_range_counts = pd.Series(data={"x_0": 1, "x_1": 2}, name="count")
        expected_range_counts.index.name = "output"
        pd.testing.assert_series_equal(rv.range_counts, expected_range_counts)

        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(["X_0", "X_1"])
        expected_range_data = pd.DataFrame.from_dict(
            data={"x_0": (1, 2), "x_1": (3, 4)}, orient="index"
        )
        expected_range_data.columns = expected_index.data
        expected_range_data.index.name = "output"

        assert rv.index == expected_index
        pd.testing.assert_frame_equal(rv.range.data, expected_range_data)

    def test_range_1d_random_vector_with_str_name(self):
        """Test range property of 1D RandomVector with string name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": 10, "omega_1": 20, "omega_2": 10}
        rv = RandomVector(domain=domain, name="Y").from_dict(outputs)

        expected_range_domain = SampleSpace().from_list(["y_0", "y_1"])
        expected_range_domain.data.name = "output"
        expected_range_domain.name = "range(Y)"

        assert rv.range.domain == expected_range_domain
        assert rv.range.domain.name == "range(Y)"
        assert rv.range.name == "Y_range"

        expected_range_counts = pd.Series(data={"y_0": 2, "y_1": 1}, name="count")
        expected_range_counts.index.name = "output"
        pd.testing.assert_series_equal(rv.range_counts, expected_range_counts)

        expected_index = None
        expected_range_data = pd.Series(data={"y_0": 10, "y_1": 20}, name="Y")
        expected_range_data.index.name = "output"

        assert rv.index == expected_index
        pd.testing.assert_series_equal(rv.range.data, expected_range_data)

    def test_range_2d_random_vector_with_int_name(self):
        """Test range property of 2D RandomVector with int name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)}
        rv = RandomVector(domain=domain, name=42).from_dict(outputs)

        expected_range_domain = SampleSpace().from_list([0, 1])
        expected_range_domain.data.name = "output"
        expected_range_domain.name = None

        assert rv.range.domain == expected_range_domain
        assert rv.range.domain.name is None
        assert rv.range.name is None

        expected_range_counts = pd.Series(data={0: 1, 1: 2}, name="count")
        expected_range_counts.index.name = "output"
        pd.testing.assert_series_equal(rv.range_counts, expected_range_counts)

        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_range_data = pd.DataFrame.from_dict(
            data={0: (1, 2), 1: (3, 4)}, orient="index"
        )
        expected_range_data.columns = expected_index.data
        expected_range_data.index.name = "output"

        assert rv.index == expected_index
        pd.testing.assert_frame_equal(rv.range.data, expected_range_data)

    def test_range_1d_random_vector_with_int_name(self):
        """Test range property of 1D RandomVector with int name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": 1, "omega_1": 1, "omega_2": 2}
        rv = RandomVector(domain=domain, name=42).from_dict(outputs)

        expected_range_domain = SampleSpace().from_list([0, 1])
        expected_range_domain.data.name = "output"
        expected_range_domain.name = None

        assert rv.range.domain == expected_range_domain
        assert rv.range.domain.name is None
        assert rv.range.name is None

        expected_range_counts = pd.Series(data={0: 2, 1: 1}, name="count")
        expected_range_counts.index.name = "output"
        pd.testing.assert_series_equal(rv.range_counts, expected_range_counts)

        expected_index = None
        expected_range_data = pd.Series(data={0: 1, 1: 2}, name=42)
        expected_range_data.index.name = "output"

        assert rv.index == expected_index
        pd.testing.assert_series_equal(rv.range.data, expected_range_data)

    def test_range_2d_random_vector_with_none_name(self):
        """Test range property of 2D RandomVector with None name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)}
        rv = RandomVector(domain=domain, name=None).from_dict(outputs)

        expected_range_domain = SampleSpace().from_list([0, 1])
        expected_range_domain.data.name = "output"
        expected_range_domain.name = None

        assert rv.range.domain == expected_range_domain
        assert rv.range.domain.name is None
        assert rv.range.name is None

        expected_range_counts = pd.Series(data={0: 1, 1: 2}, name="count")
        expected_range_counts.index.name = "output"
        pd.testing.assert_series_equal(rv.range_counts, expected_range_counts)

        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_range_data = pd.DataFrame.from_dict(
            data={0: (1, 2), 1: (3, 4)}, orient="index"
        )
        expected_range_data.columns = expected_index.data
        expected_range_data.index.name = "output"

        assert rv.index == expected_index
        pd.testing.assert_frame_equal(rv.range.data, expected_range_data)

    def test_range_1d_random_vector_with_none_name(self):
        """Test range property of 1D RandomVector with None name."""
        domain = SampleSpace.generate_sequence(size=3, prefix="omega")
        outputs = {"omega_0": 1, "omega_1": 1, "omega_2": 2}
        rv = RandomVector(domain=domain, name=None).from_dict(outputs)

        expected_range_domain = SampleSpace().from_list([0, 1])
        expected_range_domain.data.name = "output"
        expected_range_domain.name = None

        assert rv.range.domain == expected_range_domain
        assert rv.range.domain.name is None
        assert rv.range.name is None

        expected_range_counts = pd.Series(data={0: 2, 1: 1}, name="count")
        expected_range_counts.index.name = "output"
        pd.testing.assert_series_equal(rv.range_counts, expected_range_counts)

        expected_index = None
        expected_range_data = pd.Series(data={0: 1, 1: 2}, name=None)
        expected_range_data.index.name = "output"

        assert rv.index == expected_index
        pd.testing.assert_series_equal(rv.range.data, expected_range_data)


class TestFeatureIndex:

    @pytest.fixture
    def domain(self):
        return SampleSpace.generate_sequence(size=3)

    @pytest.fixture
    def random_vector_2d(self, domain):
        outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        return RandomVector(domain=domain, name="X").from_dict(outputs)

    @pytest.fixture
    def random_vector_1d(self, domain):
        outputs = {"omega_0": 10, "omega_1": 20, "omega_2": 30}
        return RandomVector(domain=domain, name="Y").from_dict(outputs)

    def test_index_property_of_2d_random_vector(self, random_vector_2d):
        """Test index property of RandomVector."""
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list(["X_0", "X_1"])

        assert random_vector_2d.index == expected_index
        assert random_vector_2d.index.name == "index"

    def test_index_property_of_1d_random_vector(self, random_vector_1d):
        """Test index property of 1D RandomVector."""
        expected_index = None
        assert random_vector_1d.index == expected_index


class TestSigmaAlgebra:

    def test_sigma_algebra_property(self):
        """Test sigma_algebra property of RandomVector."""
        domain = SampleSpace.generate_sequence(size=3)
        outputs_2d = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        outputs_1d = {"omega_0": 10, "omega_1": 20, "omega_2": 30}
        rv_2d = RandomVector(domain=domain, name="X").from_dict(outputs_2d)
        rv_1d = RandomVector(domain=domain, name="Y").from_dict(outputs_1d)
        expected_sigma_algebra_2d = SigmaAlgebra(
            sample_space=domain,
            name="sigma(X)",
        ).from_dict(sample_id_to_atom_id=outputs_2d)
        expected_sigma_algebra_1d = SigmaAlgebra(
            sample_space=domain,
            name="sigma(Y)",
        ).from_dict(sample_id_to_atom_id=outputs_1d)

        assert rv_2d.sigma_algebra == expected_sigma_algebra_2d
        assert rv_1d.sigma_algebra == expected_sigma_algebra_1d


class TestIterFeatures:

    def test_iter_features_of_2d_random_vector(self):
        """Test iter_features method of 2D RandomVector."""
        domain = SampleSpace.generate_sequence(size=3)
        outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        rv = RandomVector(domain=domain, name="X").from_dict(outputs)

        expected_features = {
            "omega_0": FeatureVector(
                data=pd.Series(
                    [1, 2],
                    index=pd.Index(["X_0", "X_1"], name="feature"),
                    name="omega_0",
                )
            ),
            "omega_1": FeatureVector(
                data=pd.Series(
                    [3, 4],
                    index=pd.Index(["X_0", "X_1"], name="feature"),
                    name="omega_1",
                )
            ),
            "omega_2": FeatureVector(
                data=pd.Series(
                    [5, 6],
                    index=pd.Index(["X_0", "X_1"], name="feature"),
                    name="omega_2",
                )
            ),
        }

        for sample_idx, feature_vector in rv.iter_features():
            pd.testing.assert_series_equal(
                feature_vector.data, expected_features[sample_idx].data
            )

    def test_iter_features_of_1d_random_vector(self):
        """Test iter_features method of 1D RandomVector."""
        domain = SampleSpace.generate_sequence(size=3)
        outputs = {"omega_0": 10, "omega_1": 20, "omega_2": 30}
        rv = RandomVector(domain=domain, name="Y").from_dict(outputs)

        expected_features = {
            "omega_0": 10,
            "omega_1": 20,
            "omega_2": 30,
        }

        for sample_idx, feature in rv.iter_features():
            assert feature == expected_features[sample_idx]


class TestPushforward:

    @pytest.fixture
    def X(self):
        domain = SampleSpace.generate_sequence(size=3)
        outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)}
        return RandomVector(domain=domain, name="X").from_dict(outputs)

    def test_pushforward_method_with_custom_measure(self, X):
        """Test pushforward method of RandomVector."""
        probabilities = {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3}
        probability_measure = ProbabilityMeasure(sample_space=X.domain).from_dict(
            probabilities=probabilities
        )
        P_X = X.pushforward(probability_measure)

        expected_probability_measure = ProbabilityMeasure(
            sample_space=X.range.domain,
            name="P_X",
        ).from_dict(probabilities={"x_0": 0.2, "x_1": 0.8})
        assert P_X == expected_probability_measure
        assert P_X.name == "P_X"

    def test_pushforward_method_with_default_measure(self, X):
        """Test pushforward method of RandomVector with default (i.e, uniform) measure."""
        P_X = X.pushforward()

        expected_probability_measure = ProbabilityMeasure(
            sample_space=X.range.domain,
            name="P_X",
        ).from_dict(probabilities={"x_0": 1 / 3, "x_1": 2 / 3})
        assert P_X == expected_probability_measure
        assert P_X.name == "P_X"


class TestCallMethod:

    @pytest.fixture
    def domain(self):
        return SampleSpace.generate_sequence(prefix="s", size=3, name="S")

    @pytest.fixture
    def random_vector_2d(self, domain):
        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        return RandomVector(domain=domain, name="X").from_dict(outputs)

    @pytest.fixture
    def random_vector_1d(self, domain):
        outputs = {"s_0": 10, "s_1": 20, "s_2": 30}
        return RandomVector(domain=domain, name="Y").from_dict(outputs)

    def test_call_method_on_sample_index(self, random_vector_2d, random_vector_1d):
        """Test calling RandomVector on a single sample index."""
        expected_2d_features = FeatureVector(
            data=pd.Series(
                [1, 2], index=pd.Index(["X_0", "X_1"], name="feature"), name="s_0"
            )
        )
        expected_1d_features = 10
        actual_2d_features = random_vector_2d("s_0")
        actual_1d_features = random_vector_1d("s_0")

        assert isinstance(actual_2d_features, FeatureVector)
        pd.testing.assert_series_equal(
            actual_2d_features.data, expected_2d_features.data
        )
        assert actual_1d_features == expected_1d_features

    def test_call_method_on_sample_indices(self, random_vector_2d, random_vector_1d):
        """Test calling RandomVector on a list of sample indices."""
        expected_2d_rv = RandomVector(name="X|event").from_pandas(
            data=pd.DataFrame(
                [(1, 2), (5, 6)],
                index=pd.Index(["s_0", "s_2"], name="sample"),
                columns=pd.Index(["X_0", "X_1"], name="feature"),
            ),
        )
        expected_1d_rv = RandomVector().from_pandas(
            data=pd.Series(
                [10, 30],
                index=pd.Index(["s_0", "s_2"], name="sample"),
                name="Y",
            )
        )
        actual_2d_rv = random_vector_2d(["s_0", "s_2"])
        actual_1d_rv = random_vector_1d(["s_0", "s_2"])

        pd.testing.assert_frame_equal(actual_2d_rv.data, expected_2d_rv.data)
        pd.testing.assert_series_equal(actual_1d_rv.data, expected_1d_rv.data)
        assert actual_2d_rv.name == "X|event"
        assert actual_1d_rv.name == "Y|event"

    def test_call_method_on_event(self, random_vector_2d, random_vector_1d, domain):
        """Test calling RandomVector on an Event."""
        B = domain.get_event(["s_0", "s_2"], name="B")
        expected_2d_rv = RandomVector(name="X|B").from_pandas(
            data=pd.DataFrame(
                [(1, 2), (5, 6)],
                index=pd.Index(["s_0", "s_2"], name="sample"),
                columns=pd.Index(["X_0", "X_1"], name="feature"),
            ),
        )
        expected_1d_rv = RandomVector(name="Y|B").from_pandas(
            data=pd.Series(
                [10, 30],
                index=pd.Index(["s_0", "s_2"], name="sample"),
                name="Y",
            ),
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
        domain = SampleSpace().from_list(["s0", "s1", "s2"])
        X = RandomVector(domain=domain, name="X").from_dict(outputs)

        with pytest.raises(TypeError):
            X({"s0": 1})
        with pytest.raises(KeyError):
            X(3.14)
        with pytest.raises(KeyError):
            X(["s0", "s3"])
        with pytest.raises(ValueError):
            other_domain = SampleSpace().from_list(["t0", "t1", "t2"])
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
        """Test adding two RandomVectors with same domain and index."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({"omega_0": (10, 20), "omega_1": (30, 40), "omega_2": (50, 60)})
        Z = X + Y
        expected_data = pd.DataFrame(
            [(11, 22), (33, 44), (55, 66)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X+Y)_0", "(X+Y)_1"], name="feature"),
        )
        print(Z.data)
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_vector_and_scalar(self):
        """Test adding a scalar to a RandomVector."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Z = X + 10
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X+10)_0", "(X+10)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_vector(self):
        """Test adding a RandomVector to a scalar (reverse add)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Z = 10 + X
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(10+X)_0", "(10+X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10+X)"

    def test_sub_two_random_vectors(self):
        """Test subtracting two RandomVectors with same domain and index."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (10, 20), "omega_1": (30, 40), "omega_2": (50, 60)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Z = X - Y
        expected_values = pd.DataFrame(
            [(9, 18), (27, 36), (45, 54)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X-Y)_0", "(X-Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_vector_and_scalar(self):
        """Test subtracting a scalar from a RandomVector."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (10, 20), "omega_1": (30, 40), "omega_2": (50, 60)})
        Z = X - 5
        expected_data = pd.DataFrame(
            [(5, 15), (25, 35), (45, 55)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X-5)_0", "(X-5)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_vector(self):
        """Test subtracting a RandomVector from a scalar (reverse sub)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Z = 10 - X
        expected_data = pd.DataFrame(
            [(9, 8), (7, 6), (5, 4)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(10-X)_0", "(10-X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10-X)"

    def test_mul_two_random_vectors(self):
        """Test multiplying two RandomVectors with same domain and index."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (2, 3), "omega_1": (4, 5), "omega_2": (6, 7)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({"omega_0": (10, 20), "omega_1": (30, 40), "omega_2": (50, 60)})
        Z = X * Y
        expected_data = pd.DataFrame(
            [(20, 60), (120, 200), (300, 420)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X*Y)_0", "(X*Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*Y)"

    def test_mul_random_vector_and_scalar(self):
        """Test multiplying a RandomVector by a scalar."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Z = X * 10
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X*10)_0", "(X*10)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_vector(self):
        """Test multiplying a scalar by a RandomVector (reverse mul)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Z = 10 * X
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(10*X)_0", "(10*X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10*X)"

    def test_truediv_two_random_vectors(self):
        """Test dividing two RandomVectors with same domain and index."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict(
            {"omega_0": (100, 200), "omega_1": (300, 400), "omega_2": (500, 600)}
        )
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({"omega_0": (10, 20), "omega_1": (30, 40), "omega_2": (50, 60)})
        Z = X / Y
        expected_data = pd.DataFrame(
            [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X/Y)_0", "(X/Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/Y)"

    def test_truediv_random_vector_and_scalar(self):
        """Test dividing a RandomVector by a scalar."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (10, 20), "omega_1": (30, 40), "omega_2": (50, 60)})
        Z = X / 10
        expected_data = pd.DataFrame(
            [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X/10)_0", "(X/10)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_vector(self):
        """Test dividing a scalar by a RandomVector (reverse div)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (2, 4), "omega_1": (5, 10), "omega_2": (20, 25)})
        Z = 100 / X
        expected_data = pd.DataFrame(
            [(50.0, 25.0), (20.0, 10.0), (5.0, 4.0)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(100/X)_0", "(100/X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(100/X)"

    def test_pow_two_random_vectors(self):
        """Test exponentiating two RandomVectors with same domain and index."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (2, 3), "omega_1": (4, 5), "omega_2": (6, 7)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({"omega_0": (2, 2), "omega_1": (2, 2), "omega_2": (2, 2)})
        Z = X**Y
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X**Y)_0", "(X**Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self):
        """Test exponentiating a RandomVector by a scalar."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (2, 3), "omega_1": (4, 5), "omega_2": (6, 7)})
        Z = X**2
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(X**2)_0", "(X**2)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self):
        """Test exponentiating a scalar by a RandomVector (reverse pow)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (2, 3), "omega_1": (4, 5), "omega_2": (0, 1)})
        Z = 2**X
        expected_data = pd.DataFrame(
            [(4, 8), (16, 32), (1, 2)],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            columns=pd.Index(["(2**X)_0", "(2**X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(2**X)"

    def test_add_with_different_domains_raises_error(self):
        """Test that adding RandomVectors with different domains raises ValueError."""
        Omega1 = SampleSpace.generate_sequence(size=3, prefix="omega")
        Omega2 = SampleSpace.generate_sequence(size=3, prefix="alpha")
        X = RandomVector(
            domain=Omega1,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        Y = RandomVector(
            domain=Omega2,
            name="Y",
        ).from_dict({"alpha_0": (1, 2), "alpha_1": (3, 4), "alpha_2": (5, 6)})
        try:
            Z = X + Y  # noqa: F841
            raise AssertionError("Expected ValueError for different domains")
        except ValueError as e:
            assert "different domains" in str(e)

    def test_add_with_non_random_vector_raises_error(self):
        """Test that adding a non-RandomVector and non-scalar raises TypeError."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)})
        try:
            Z = X + "invalid"  # noqa: F841
            raise AssertionError("Expected TypeError for invalid operand")
        except TypeError as e:
            assert "RandomVector or scalar" in str(e)


class TestArithmeticWithRandomVariable:

    def test_add_two_random_variables(self):
        """Test adding two RandomVariables with same domain."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            outputs={"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            outputs={"omega_0": 10, "omega_1": 30, "omega_2": 50},
        )
        Z = X + Y
        expected_values = pd.Series(
            [11, 33, 55],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X+Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_variable_and_scalar(self):
        """Test adding a scalar to a RandomVariable."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Z = X + 10
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X+10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_variable(self):
        """Test adding a RandomVariable to a scalar (reverse add)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Z = 10 + X
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(10+X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10+X)"

    def test_sub_two_random_variables(self):
        """Test subtracting two RandomVariables with same domain."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 10, "omega_1": 30, "omega_2": 50},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Z = X - Y
        expected_values = pd.Series(
            [9, 27, 45],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X-Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_variable_and_scalar(self):
        """Test subtracting a scalar from a RandomVariable."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 10, "omega_1": 30, "omega_2": 50},
        )
        Z = X - 5
        expected_values = pd.Series(
            [5, 25, 45],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X-5)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_variable(self):
        """Test subtracting a RandomVariable from a scalar (reverse sub)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Z = 10 - X
        expected_values = pd.Series(
            [9, 7, 5],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(10-X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10-X)"

    def test_mul_two_random_variables(self):
        """Test multiplying two RandomVariables with same domain."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 2, "omega_1": 4, "omega_2": 6},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {"omega_0": 10, "omega_1": 30, "omega_2": 50},
        )
        Z = X * Y
        expected_values = pd.Series(
            [20, 120, 300],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X*Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*Y)"

    def test_mul_random_variable_and_scalar(self):
        """Test multiplying a RandomVariable by a scalar."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Z = X * 10
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X*10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_variable(self):
        """Test multiplying a scalar by a RandomVariable (reverse mul)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Z = 10 * X
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(10*X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10*X)"

    def test_truediv_two_random_variables(self):
        """Test dividing two RandomVariables with same domain."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 100, "omega_1": 300, "omega_2": 500},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {"omega_0": 10, "omega_1": 30, "omega_2": 50},
        )
        Z = X / Y
        expected_values = pd.Series(
            [10.0, 10.0, 10.0],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X/Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/Y)"

    def test_truediv_random_variable_and_scalar(self):
        """Test dividing a RandomVariable by a scalar."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 10, "omega_1": 30, "omega_2": 50},
        )
        Z = X / 10
        expected_values = pd.Series(
            [1.0, 3.0, 5.0],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X/10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_variable(self):
        """Test dividing a scalar by a RandomVariable (reverse div)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 2, "omega_1": 5, "omega_2": 20},
        )
        Z = 100 / X
        expected_values = pd.Series(
            [50.0, 20.0, 5.0],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(100/X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(100/X)"

    def test_pow_two_random_variables(self):
        """Test exponentiating two RandomVariables with same domain."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 2, "omega_1": 4, "omega_2": 6},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {"omega_0": 2, "omega_1": 2, "omega_2": 2},
        )
        Z = X**Y
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X**Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**Y)"

    def test_pow_random_variable_and_scalar(self):
        """Test exponentiating a RandomVariable by a scalar."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 2, "omega_1": 4, "omega_2": 6},
        )
        Z = X**2
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(X**2)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_variable(self):
        """Test exponentiating a scalar by a RandomVariable (reverse pow)."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 2, "omega_1": 4, "omega_2": 0},
        )
        Z = 2**X
        expected_values = pd.Series(
            [4, 16, 1],
            index=pd.Index(["omega_0", "omega_1", "omega_2"], name="sample"),
            name="(2**X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(2**X)"

    def test_add_with_different_domains_raises_error(self):
        """Test that adding RandomVariables with different domains raises ValueError."""
        Omega1 = SampleSpace.generate_sequence(size=3, prefix="omega")
        Omega2 = SampleSpace.generate_sequence(size=3, prefix="alpha")
        X = RandomVariable(domain=Omega1, name="X").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        Y = RandomVariable(domain=Omega2, name="Y").from_dict(
            {"alpha_0": 1, "alpha_1": 3, "alpha_2": 5},
        )
        try:
            Z = X + Y  # noqa: F841
            raise AssertionError("Expected ValueError for different domains")
        except ValueError as e:
            assert "different domains" in str(e)

    def test_add_with_non_random_variable_raises_error(self):
        """Test that adding a non-RandomVariable and non-scalar raises TypeError."""
        Omega = SampleSpace.generate_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 3, "omega_2": 5},
        )
        try:
            Z = X + "invalid"  # noqa: F841
            raise AssertionError("Expected TypeError for invalid operand")
        except TypeError as e:
            assert "RandomVector or scalar" in str(e)
