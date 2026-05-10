import numpy as np
import pandas as pd
import pytest

from sigalg.core import (
    FeatureVector,
    Index,
    ProbabilityMeasure,
    ProbabilitySpace,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.8,
                1: 0.2,
            }
        )

    @pytest.fixture
    def index(self):
        return Index().from_sequence(size=2)

    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters."""
        X = RandomVector()
        prob_space = ProbabilitySpace()

        assert X.point_outputs is None
        assert X.atom_outputs is None
        assert X.data is None
        assert X.atom_data is None
        assert X.components is None
        assert X.index is None
        assert X.generated_sig_alg is None
        assert X.prob_space == prob_space
        assert X.domain is None
        assert X.sig_alg is None
        assert X.prob_measure is None
        assert X.range is None

    def test_constructor_with_custom_parameters(self, Omega, F, P, index):
        """Test the constructor with custom parameters."""
        Y = RandomVector(domain=Omega, sig_alg=F, prob_measure=P, index=index, name="Y")
        prob_space = ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)

        assert Y.point_outputs is None
        assert Y.atom_outputs is None
        assert Y.data is None
        assert Y.atom_data is None
        assert Y.components is None
        assert Y.index == index
        assert Y.generated_sig_alg is None
        assert Y.prob_space == prob_space
        assert Y.domain == Omega
        assert Y.sig_alg == F
        assert Y.prob_measure == P
        assert Y.range is None


class TestFromDict:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.8,
                1: 0.2,
            }
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return ProbabilitySpace(sample_space=Omega, sig_alg=F, prob_measure=P)

    @pytest.fixture
    def point_outputs_2d(self):
        return {0: (1, 2), 1: (1, 2), 2: (3, 4)}

    @pytest.fixture
    def atom_outputs_2d(self):
        return {0: (1, 2), 1: (3, 4)}

    @pytest.fixture
    def point_outputs_1d(self):
        return {0: 1, 1: 1, 2: 2}

    @pytest.fixture
    def atom_outputs_1d(self):
        return {0: 1, 1: 2}

    def test_2d_point_outputs(self, prob_space, point_outputs_2d, atom_outputs_2d):
        """Test RandomVector constructor with 2D point outputs."""
        X = RandomVector(*prob_space).from_dict(point_outputs_2d, type="point")
        expected_index = Index(name="index", data_name="feature").from_list(
            ["X_0", "X_1"]
        )
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_1d_point_outputs(self, prob_space, point_outputs_1d, atom_outputs_1d):
        """Test RandomVector constructor with 1D point outputs."""
        Y = RandomVector(*prob_space, name="Y").from_dict(
            point_outputs_1d, type="point"
        )
        expected_data = pd.Series(
            data=[1, 1, 2],
            index=prob_space.sample_space.data,
            name="Y",
        )
        expected_atom_data = pd.Series(
            data=[1, 2],
            index=pd.Index([0, 1], name="atom ID"),
            name="Y",
        )

        assert Y.point_outputs == point_outputs_1d
        assert Y.atom_outputs == atom_outputs_1d
        pd.testing.assert_series_equal(Y.data, expected_data)
        pd.testing.assert_series_equal(Y.atom_data, expected_atom_data)

    def test_2d_atom_outputs(self, prob_space, point_outputs_2d, atom_outputs_2d):
        """Test RandomVector constructor with 2D atom outputs."""
        X = RandomVector(*prob_space).from_dict(atom_outputs_2d, type="atom")
        expected_index = Index(name="index", data_name="feature").from_list(
            ["X_0", "X_1"]
        )
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_1d_atom_outputs(self, prob_space, point_outputs_1d, atom_outputs_1d):
        """Test RandomVector constructor with 1D atom outputs."""
        Y = RandomVector(*prob_space, name="Y").from_dict(atom_outputs_1d, type="atom")
        expected_data = pd.Series(
            data=[1, 1, 2],
            index=prob_space.sample_space.data,
            name="Y",
        )
        expected_atom_data = pd.Series(
            data=[1, 2],
            index=pd.Index([0, 1], name="atom ID"),
            name="Y",
        )

        assert Y.point_outputs == point_outputs_1d
        assert Y.atom_outputs == atom_outputs_1d
        pd.testing.assert_series_equal(Y.data, expected_data)
        pd.testing.assert_series_equal(Y.atom_data, expected_atom_data)
        assert Y.index is None

    def test_dicts_out_of_order_with_2d_points(self, prob_space):
        """Test that from_dict correctly handles 2d-dicts with keys out of order."""
        point_outputs = {2: (3, 4), 0: (1, 2), 1: (1, 2)}
        X = RandomVector(*prob_space).from_dict(point_outputs, type="point")
        expected_atom_outputs = {0: (1, 2), 1: (3, 4)}
        expected_index = Index(name="index", data_name="feature").from_list(
            ["X_0", "X_1"]
        )
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == point_outputs
        assert X.atom_outputs == expected_atom_outputs
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_dicts_out_of_order_with_1d_points(self, prob_space):
        """Test that from_dict correctly handles 1d-dicts with keys out of order."""
        point_outputs = {2: 2, 0: 1, 1: 1}
        X = RandomVector(*prob_space).from_dict(point_outputs, type="point")
        expected_atom_outputs = {0: 1, 1: 2}
        expected_data = pd.Series(
            [1, 1, 2], index=prob_space.sample_space.data, name="X"
        )
        expected_atom_data = pd.Series(
            [1, 2], index=pd.Index([0, 1], name="atom ID"), name="X"
        )

        assert X.point_outputs == point_outputs
        assert X.atom_outputs == expected_atom_outputs
        pd.testing.assert_series_equal(X.data, expected_data)
        pd.testing.assert_series_equal(X.atom_data, expected_atom_data)
        assert X.index is None

    def test_dicts_out_of_order_with_2d_atoms(self, prob_space):
        """Test that from_dict correctly handles 2d-dicts with keys out of order."""
        atom_outputs = {1: (3, 4), 0: (1, 2)}
        X = RandomVector(*prob_space).from_dict(atom_outputs, type="atom")
        expected_point_outputs = {0: (1, 2), 1: (1, 2), 2: (3, 4)}
        expected_index = Index(name="index", data_name="feature").from_list(
            ["X_0", "X_1"]
        )
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == expected_point_outputs
        assert X.atom_outputs == atom_outputs
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_dicts_out_of_order_with_1d_atoms(self, prob_space):
        """Test that from_dict correctly handles 1d-dicts with keys out of order."""
        atom_outputs = {1: 2, 0: 1}
        X = RandomVector(*prob_space).from_dict(atom_outputs, type="atom")
        expected_point_outputs = {0: 1, 1: 1, 2: 2}
        expected_data = pd.Series(
            [1, 1, 2], index=prob_space.sample_space.data, name="X"
        )
        expected_atom_data = pd.Series(
            [1, 2], index=pd.Index([0, 1], name="atom ID"), name="X"
        )

        assert X.point_outputs == expected_point_outputs
        assert X.atom_outputs == atom_outputs
        pd.testing.assert_series_equal(X.data, expected_data)
        pd.testing.assert_series_equal(X.atom_data, expected_atom_data)
        assert X.index is None

    def test_2d_point_outputs_with_none_name(
        self, prob_space, point_outputs_2d, atom_outputs_2d
    ):
        """Test RandomVector constructor with 2D point outputs and None name."""
        X = RandomVector(*prob_space, name=None).from_dict(
            point_outputs_2d, type="point"
        )
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_2d_atom_outputs_with_none_name(
        self, prob_space, point_outputs_2d, atom_outputs_2d
    ):
        """Test RandomVector constructor with 2D atom outputs and None name."""
        X = RandomVector(*prob_space, name=None).from_dict(atom_outputs_2d, type="atom")
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_1d_point_outputs_with_none_name(
        self, prob_space, point_outputs_1d, atom_outputs_1d
    ):
        """Test RandomVector constructor with 1D point outputs and None name."""
        Y = RandomVector(*prob_space, name=None).from_dict(
            point_outputs_1d, type="point"
        )
        expected_data = pd.Series(
            data=[1, 1, 2],
            index=prob_space.sample_space.data,
            name=None,
        )
        expected_atom_data = pd.Series(
            data=[1, 2],
            index=pd.Index([0, 1], name="atom ID"),
            name=None,
        )

        assert Y.point_outputs == point_outputs_1d
        assert Y.atom_outputs == atom_outputs_1d
        pd.testing.assert_series_equal(Y.data, expected_data)
        pd.testing.assert_series_equal(Y.atom_data, expected_atom_data)
        assert Y.index is None

    def test_1d_atom_outputs_with_none_name(
        self, prob_space, point_outputs_1d, atom_outputs_1d
    ):
        """Test RandomVector constructor with 1D atom outputs and None name."""
        Y = RandomVector(*prob_space, name=None).from_dict(atom_outputs_1d, type="atom")
        expected_data = pd.Series(
            data=[1, 1, 2],
            index=prob_space.sample_space.data,
            name=None,
        )
        expected_atom_data = pd.Series(
            data=[1, 2],
            index=pd.Index([0, 1], name="atom ID"),
            name=None,
        )

        assert Y.point_outputs == point_outputs_1d
        assert Y.atom_outputs == atom_outputs_1d
        pd.testing.assert_series_equal(Y.data, expected_data)
        pd.testing.assert_series_equal(Y.atom_data, expected_atom_data)
        assert Y.index is None

    def test_2d_point_outputs_with_int_name(
        self, prob_space, point_outputs_2d, atom_outputs_2d
    ):
        """Test RandomVector constructor with 2D point outputs and integer name."""
        X = RandomVector(*prob_space, name=42).from_dict(point_outputs_2d, type="point")
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_2d_atom_outputs_with_int_name(
        self, prob_space, point_outputs_2d, atom_outputs_2d
    ):
        """Test RandomVector constructor with 2D atom outputs and integer name."""
        X = RandomVector(*prob_space, name=42).from_dict(atom_outputs_2d, type="atom")
        expected_index = Index(
            name="index",
            data_name="feature",
        ).from_list([0, 1])
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=expected_index.data,
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=expected_index.data,
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == expected_index

    def test_1d_point_outputs_with_int_name(
        self, prob_space, point_outputs_1d, atom_outputs_1d
    ):
        """Test RandomVector constructor with 1D point outputs and integer name."""
        Y = RandomVector(*prob_space, name=42).from_dict(point_outputs_1d, type="point")
        expected_data = pd.Series(
            data=[1, 1, 2],
            index=prob_space.sample_space.data,
            name=42,
        )
        expected_atom_data = pd.Series(
            data=[1, 2],
            index=pd.Index([0, 1], name="atom ID"),
            name=42,
        )

        assert Y.point_outputs == point_outputs_1d
        assert Y.atom_outputs == atom_outputs_1d
        pd.testing.assert_series_equal(Y.data, expected_data)
        pd.testing.assert_series_equal(Y.atom_data, expected_atom_data)
        assert Y.index is None

    def test_1d_atom_outputs_with_int_name(
        self, prob_space, point_outputs_1d, atom_outputs_1d
    ):
        """Test RandomVector constructor with 1D atom outputs and integer name."""
        Y = RandomVector(*prob_space, name=42).from_dict(atom_outputs_1d, type="atom")
        expected_data = pd.Series(
            data=[1, 1, 2],
            index=prob_space.sample_space.data,
            name=42,
        )
        expected_atom_data = pd.Series(
            data=[1, 2],
            index=pd.Index([0, 1], name="atom ID"),
            name=42,
        )

        assert Y.point_outputs == point_outputs_1d
        assert Y.atom_outputs == atom_outputs_1d
        pd.testing.assert_series_equal(Y.data, expected_data)
        pd.testing.assert_series_equal(Y.atom_data, expected_atom_data)
        assert Y.index is None

    def test_2d_points_with_custom_index(
        self, prob_space, point_outputs_2d, atom_outputs_2d
    ):
        """Test RandomVector.from_dict with custom index parameter."""
        custom_index = Index(
            name="custom_index",
            data_name="feature",
        ).from_list(["feature_a", "feature_b"])
        X = RandomVector(*prob_space, index=custom_index).from_dict(
            point_outputs_2d, type="point"
        )
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=pd.Index(["feature_a", "feature_b"], name="feature"),
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=pd.Index(["feature_a", "feature_b"], name="feature"),
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == custom_index

    def test_2d_atoms_with_custom_index(
        self, prob_space, point_outputs_2d, atom_outputs_2d
    ):
        """Test RandomVector.from_dict with custom index parameter."""
        custom_index = Index(
            name="custom_index",
            data_name="feature",
        ).from_list(["feature_a", "feature_b"])
        X = RandomVector(*prob_space, index=custom_index).from_dict(
            atom_outputs_2d, type="atom"
        )
        expected_data = pd.DataFrame(
            [(1, 2), (1, 2), (3, 4)],
            index=prob_space.sample_space.data,
            columns=pd.Index(["feature_a", "feature_b"], name="feature"),
        )
        expected_atom_data = pd.DataFrame(
            [(1, 2), (3, 4)],
            index=pd.Index([0, 1], name="atom ID"),
            columns=pd.Index(["feature_a", "feature_b"], name="feature"),
        )

        assert X.point_outputs == point_outputs_2d
        assert X.atom_outputs == atom_outputs_2d
        pd.testing.assert_frame_equal(X.data, expected_data)
        pd.testing.assert_frame_equal(X.atom_data, expected_atom_data)
        assert X.index == custom_index

    def test_from_dict_with_non_measurable_random_vector_raises(self, prob_space):
        """Test from_dict with non-measurable random vector raises."""
        with pytest.raises(ValueError, match="not measureable"):
            RandomVector(*prob_space).from_dict(
                {
                    0: (1, 2),
                    1: (3, 4),
                    2: (3, 4),
                },
                type="point",
            )


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
            name="Y",
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

    def test_from_pandas_sets_default_probability_measure(self):
        """Test that from_pandas sets a default uniform probability measure."""
        data = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)])
        rv = RandomVector(name="W").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_prob_measure = ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(expected_domain)
        )

        assert rv.prob_measure == expected_prob_measure

    def test_from_pandas_sets_default_sigma_algebra(self):
        """Test that from_pandas sets a default power set sigma algebra."""
        data = pd.DataFrame(data=[(1, 2), (3, 4), (5, 6)])
        rv = RandomVector(name="V").from_pandas(data=data)

        expected_domain = SampleSpace().from_list(list(data.index))
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space=expected_domain)

        assert rv.sig_alg == expected_sigma_algebra


class TestFromNumPy:
    def test_from_numpy(self):
        """Test RandomVector.from_numpy method."""
        arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
        arr_flat = np.array([10, 20, 30])
        arr_col = np.array([[10], [20], [30]])
        rv_2d = RandomVector(name="X").from_numpy(array=arr_2d)
        rv_flat = RandomVector(name="Y").from_numpy(array=arr_flat)
        rv_col = RandomVector(name="Z").from_numpy(array=arr_col)

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

        assert rv_2d.data.shape == (3, 2)
        assert rv_flat.data.shape == (3,)
        assert rv_col.data.shape == (3,)

    def test_from_numpy_sets_default_probability_measure(self):
        """Test that from_numpy sets a default uniform probability measure."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        rv = RandomVector(name="W").from_numpy(array=arr)

        expected_domain = SampleSpace().from_sequence(size=3)
        expected_prob_measure = ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(expected_domain)
        )

        assert rv.prob_measure == expected_prob_measure

    def test_from_numpy_sets_default_sigma_algebra(self):
        """Test that from_numpy sets a default power set sigma algebra."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        rv = RandomVector(name="V").from_numpy(array=arr)

        expected_domain = SampleSpace().from_sequence(size=3)
        expected_sigma_algebra = SigmaAlgebra.power_set(sample_space=expected_domain)

        assert rv.sig_alg == expected_sigma_algebra


class TestFromConstant:
    def test_from_constant_2d(self):
        """Test the from_constant method with a 2-dimensional output."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega).from_constant(constant=(1, 2))
        expected_index = Index(data_name="feature").from_sequence(size=2, prefix="X")
        expected_data = pd.DataFrame(
            [(1, 2)] * 3, index=Omega.data, columns=expected_index.data
        )

        pd.testing.assert_frame_equal(X.data, expected_data)

    def test_from_constant_1d(self):
        """Test the from_constant method with a 1-dimensional output."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega).from_constant(constant=2)
        expected_data = pd.Series(
            [
                2,
            ]
            * 3,
            index=Omega.data,
            name="X",
        )

        pd.testing.assert_series_equal(X.data, expected_data)


# --------------------- test properties --------------------- #


class TestPointOutputs:
    pass


class TestAtomOutputs:
    pass


class TestData:
    pass


class TestAtomData:
    pass


class TestComponents:
    pass


class TestIndex:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def random_vector_2d(self, Omega):
        outputs = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        return RandomVector(domain=Omega, name="X").from_dict(outputs)

    @pytest.fixture
    def random_vector_1d(self, Omega):
        outputs = {0: 10, 1: 20, 2: 30}
        return RandomVector(domain=Omega, name="Y").from_dict(outputs)

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


class TestGeneratedSigAlg:
    def test_generated_sigma_algebra_property(self):
        """Test generated_sigma_algebra property of RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        outputs_2d = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        outputs_1d = {0: 10, 1: 20, 2: 30}
        X = RandomVector(domain=Omega, name="X").from_dict(outputs_2d)
        Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_1d)
        expected_sigma_algebra_2d = SigmaAlgebra(
            sample_space=Omega,
            name="sigma(X)",
        ).from_dict(sample_id_to_atom_id=outputs_2d)
        expected_sigma_algebra_1d = SigmaAlgebra(
            sample_space=Omega,
            name="sigma(Y)",
        ).from_dict(sample_id_to_atom_id=outputs_1d)

        assert X.generated_sig_alg == expected_sigma_algebra_2d
        assert Y.generated_sig_alg == expected_sigma_algebra_1d


class TestProbSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.8,
                1: 0.2,
            }
        )

    @pytest.fixture
    def point_outputs(self):
        return {
            0: (1, 2),
            1: (1, 2),
            2: (3, 4),
        }

    def test_prob_space_with_defaults(self, Omega, point_outputs):
        """Test that default probability space has power-set sigma-algebra and uniform probability measure."""
        X = RandomVector(domain=Omega).from_dict(point_outputs)
        prob_space = ProbabilitySpace(sample_space=Omega)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == SigmaAlgebra.power_set(sample_space=Omega)
        assert X.prob_space.prob_measure == ProbabilityMeasure.uniform(
            sig_alg=SigmaAlgebra.power_set(sample_space=Omega)
        )

    def test_prob_space_with_custom_prob_measure(self, Omega, P, point_outputs):
        """Test constructor with custom probability measure sets sigma-algebra to the sigma-algebra of the probability measure."""
        X = RandomVector(domain=Omega, prob_measure=P).from_dict(point_outputs)
        prob_space = ProbabilitySpace(Omega, prob_measure=P)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == P.sig_alg
        assert X.prob_space.prob_measure == P

    def test_prob_space_with_custom_sigma_algebra(self, Omega, F, point_outputs):
        """Test constructor with custom sigma-algebra sets the probability measure to uniform over the sigma-algebra."""
        X = RandomVector(domain=Omega, sig_alg=F).from_dict(point_outputs)
        prob_space = ProbabilitySpace(Omega, F)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.prob_measure == ProbabilityMeasure.uniform(sig_alg=F)

    def test_prob_space_with_all_components(self, Omega, F, P, point_outputs):
        """Test constructor with all components."""
        prob_space = ProbabilitySpace(Omega, F, P)
        X = RandomVector(*prob_space).from_dict(point_outputs)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.prob_measure == P


class TestDomain:
    pass


class TestSigAlg:
    pass


class TestProbMeasure:
    pass


class TestRange:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict(
            {
                0: 0,
                1: 0,
                2: 1,
                3: 2,
            }
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(sig_alg=F).from_dict(
            {
                0: 0.2,
                1: 0.1,
                2: 0.7,
            }
        )

    @pytest.fixture
    def point_outputs_2d(self):
        return {
            0: (1, 2),
            1: (1, 2),
            2: (3, 4),
            3: (3, 4),
        }

    @pytest.fixture
    def point_outputs_1d(self):
        return {
            0: 4,
            1: 4,
            2: 5,
            3: 6,
        }

    def test_range_2d_random_vector_with_str_name(self, Omega, F, P, point_outputs_2d):
        """Test range property of 2D RandomVector with string name."""
        X = RandomVector(Omega, F, P).from_dict(point_outputs_2d)
        expected_pushforward = ProbabilityMeasure(name="P_X").from_dict(
            {
                (1, 2): 0.2,
                (3, 4): 0.8,
            },
            type="point",
        )
        expected_range = ProbabilitySpace(prob_measure=expected_pushforward)

        assert X.range == expected_range

    def test_range_1d_random_vector_with_str_name(self, Omega, F, P, point_outputs_1d):
        """Test range property of 1D RandomVector with string name."""
        X = RandomVector(Omega, F, P, name="X").from_dict(point_outputs_1d)
        expected_pushforward = ProbabilityMeasure(name="P_X").from_dict(
            {
                4: 0.2,
                5: 0.1,
                6: 0.7,
            },
            type="point",
        )
        expected_range = ProbabilitySpace(prob_measure=expected_pushforward)

        assert X.range == expected_range

    def test_range_2d_random_vector_with_int_name(self, Omega, F, P, point_outputs_2d):
        """Test range property of 2D RandomVector with int name."""
        X = RandomVector(Omega, F, P, name=42).from_dict(point_outputs_2d)
        expected_pushforward = ProbabilityMeasure(name="P_42").from_dict(
            {
                (1, 2): 0.2,
                (3, 4): 0.8,
            },
            type="point",
        )
        expected_range = ProbabilitySpace(prob_measure=expected_pushforward)

        assert X.range == expected_range

    def test_range_1d_random_vector_with_int_name(self, Omega, F, P, point_outputs_1d):
        """Test range property of 1D RandomVector with int name."""
        X = RandomVector(Omega, F, P, name=42).from_dict(point_outputs_1d)
        expected_pushforward = ProbabilityMeasure(name="P_42").from_dict(
            {
                4: 0.2,
                5: 0.1,
                6: 0.7,
            },
            type="point",
        )
        expected_range = ProbabilitySpace(prob_measure=expected_pushforward)

        assert X.range == expected_range

    def test_range_2d_random_vector_with_none_name(self, Omega, F, P, point_outputs_2d):
        """Test range property of 2D RandomVector with None name."""
        X = RandomVector(Omega, F, P, name=None).from_dict(point_outputs_2d)
        expected_pushforward = ProbabilityMeasure(name="pushforward").from_dict(
            {
                (1, 2): 0.2,
                (3, 4): 0.8,
            },
            type="point",
        )
        expected_range = ProbabilitySpace(prob_measure=expected_pushforward)

        assert X.range == expected_range

    def test_range_1d_random_vector_with_none_name(self, Omega, F, P, point_outputs_1d):
        """Test range property of 1D RandomVector with None name."""
        X = RandomVector(Omega, F, P, name=None).from_dict(point_outputs_1d)
        expected_pushforward = ProbabilityMeasure(name="pushforward").from_dict(
            {
                4: 0.2,
                5: 0.1,
                6: 0.7,
            },
            type="point",
        )
        expected_range = ProbabilitySpace(prob_measure=expected_pushforward)

        assert X.range == expected_range


# --------------------- test probability methods --------------------- #


class TestIsMeasurable:
    pass


# --------------------- test data access methods --------------------- #


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace(name="S").from_sequence(prefix="s", size=3)

    @pytest.fixture
    def random_vector_2d(self, Omega):
        outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        return RandomVector(domain=Omega, name="X").from_dict(outputs)

    @pytest.fixture
    def random_vector_1d(self, Omega):
        outputs = {"s_0": 10, "s_1": 20, "s_2": 30}
        return RandomVector(domain=Omega, name="Y").from_dict(outputs)

    def test_call_method_on_sample_index(self, random_vector_2d, random_vector_1d):
        """Test calling RandomVector on a single sample index."""
        expected_2d_features = FeatureVector().from_pandas(
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

    def test_call_method_on_event(self, random_vector_2d, random_vector_1d, Omega):
        """Test calling RandomVector on an Event."""
        F = SigmaAlgebra.power_set(Omega)
        B = F.get_event(["s_0", "s_2"], name="B")
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
            other_F = SigmaAlgebra.power_set(other_domain)
            A = other_F.get_event(["t0", "t2"])
            X(A)


class TestGetComponentRV:
    pass


class TestGetSubVector:
    pass


class TestIterFeatures:
    def test_iter_features_of_2d_random_vector(self):
        """Test iter_features method of 2D RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        outputs = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        X = RandomVector(domain=Omega, name="X").from_dict(outputs)

        expected_features = {
            0: FeatureVector().from_pandas(
                data=pd.Series(
                    [1, 2],
                    index=pd.Index(["X_0", "X_1"], name="feature"),
                    name=0,
                )
            ),
            1: FeatureVector().from_pandas(
                data=pd.Series(
                    [3, 4],
                    index=pd.Index(["X_0", "X_1"], name="feature"),
                    name=1,
                )
            ),
            2: FeatureVector().from_pandas(
                data=pd.Series(
                    [5, 6],
                    index=pd.Index(["X_0", "X_1"], name="feature"),
                    name=2,
                )
            ),
        }

        for sample_idx, feature_vector in X.iter_features():
            pd.testing.assert_series_equal(
                feature_vector.data, expected_features[sample_idx].data
            )

    def test_iter_features_of_1d_random_vector(self):
        """Test iter_features method of 1D RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        outputs = {0: 10, 1: 20, 2: 30}
        Y = RandomVector(domain=Omega, name="Y").from_dict(outputs)

        expected_features = {
            0: 10,
            1: 20,
            2: 30,
        }

        for sample_idx, feature in Y.iter_features():
            assert feature == expected_features[sample_idx]


class TestApplyToFeatures:
    pass


# --------------------- equality --------------------- #


class TestEquality:
    pass


# --------------------- arithmetic --------------------- #


class TestArithmetic:
    def test_add_two_random_vectors(self):
        """Test adding two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X + Y
        expected_data = pd.DataFrame(
            [(11, 22), (33, 44), (55, 66)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X+Y)_0", "(X+Y)_1"], name="feature"),
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_vector_and_scalar(self):
        """Test adding a scalar to a RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = X + 10
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X+10)_0", "(X+10)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_vector(self):
        """Test adding a RandomVector to a scalar (reverse add)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = 10 + X
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(10+X)_0", "(10+X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10+X)"

    def test_sub_two_random_vectors(self):
        """Test subtracting two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = X - Y
        expected_values = pd.DataFrame(
            [(9, 18), (27, 36), (45, 54)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X-Y)_0", "(X-Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_vector_and_scalar(self):
        """Test subtracting a scalar from a RandomVector."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X - 5
        expected_data = pd.DataFrame(
            [(5, 15), (25, 35), (45, 55)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X-5)_0", "(X-5)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_vector(self):
        """Test subtracting a RandomVector from a scalar (reverse sub)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = 10 - X
        expected_data = pd.DataFrame(
            [(9, 8), (7, 6), (5, 4)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(10-X)_0", "(10-X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10-X)"

    def test_mul_two_random_vectors(self):
        """Test multiplying two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X * Y
        expected_data = pd.DataFrame(
            [(20, 60), (120, 200), (300, 420)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X*Y)_0", "(X*Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*Y)"

    def test_mul_random_vector_and_scalar(self):
        """Test multiplying a RandomVector by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = X * 10
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X*10)_0", "(X*10)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_vector(self):
        """Test multiplying a scalar by a RandomVector (reverse mul)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})
        Z = 10 * X
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(10*X)_0", "(10*X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(10*X)"

    def test_truediv_two_random_vectors(self):
        """Test dividing two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (100, 200), 1: (300, 400), 2: (500, 600)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X / Y
        expected_data = pd.DataFrame(
            [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X/Y)_0", "(X/Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/Y)"

    def test_truediv_random_vector_and_scalar(self):
        """Test dividing a RandomVector by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (10, 20), 1: (30, 40), 2: (50, 60)})
        Z = X / 10
        expected_data = pd.DataFrame(
            [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X/10)_0", "(X/10)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_vector(self):
        """Test dividing a scalar by a RandomVector (reverse div)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (2, 4), 1: (5, 10), 2: (20, 25)})
        Z = 100 / X
        expected_data = pd.DataFrame(
            [(50.0, 25.0), (20.0, 10.0), (5.0, 4.0)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(100/X)_0", "(100/X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(100/X)"

    def test_pow_two_random_vectors(self):
        """Test exponentiating two RandomVectors with same domain and index."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Y = RandomVector(
            domain=Omega,
            name="Y",
        ).from_dict({0: (2, 2), 1: (2, 2), 2: (2, 2)})
        Z = X**Y
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X**Y)_0", "(X**Y)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self):
        """Test exponentiating a RandomVector by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Z = X**2
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X**2)_0", "(X**2)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self):
        """Test exponentiating a scalar by a RandomVector (reverse pow)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (2, 3), 1: (4, 5), 2: (0, 1)})
        Z = 2**X
        expected_data = pd.DataFrame(
            [(4, 8), (16, 32), (1, 2)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(2**X)_0", "(2**X)_1"], name="feature"),
        )
        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(2**X)"

    def test_add_with_different_probability_spaces_raises_error(self):
        """Test that adding RandomVectors with different probability spaces raises ValueError."""
        Omega1 = SampleSpace().from_list(["a", "b", "c"])
        Omega2 = SampleSpace().from_list(["x", "y", "z"])
        X = RandomVector(
            domain=Omega1,
            name="X",
        ).from_dict({"a": (1, 2), "b": (3, 4), "c": (5, 6)})
        Y = RandomVector(
            domain=Omega2,
            name="Y",
        ).from_dict({"x": (1, 2), "y": (3, 4), "z": (5, 6)})

        with pytest.raises(ValueError, match="different probability spaces"):
            Z = X + Y  # noqa: F841

    def test_add_with_non_random_vector_raises_error(self):
        """Test that adding a non-RandomVector and non-scalar raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(
            domain=Omega,
            name="X",
        ).from_dict({0: (1, 2), 1: (3, 4), 2: (5, 6)})

        with pytest.raises(TypeError):
            Z = X + "invalid"  # noqa: F841


class TestArithmeticWithRandomVariable:
    def test_add_two_random_variables(self):
        """Test adding two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            outputs={0: 1, 1: 3, 2: 5},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            outputs={0: 10, 1: 30, 2: 50},
        )
        Z = X + Y
        expected_values = pd.Series(
            [11, 33, 55],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X+Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+Y)"
        assert Z.domain == Omega

    def test_add_random_variable_and_scalar(self):
        """Test adding a scalar to a RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = X + 10
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X+10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_variable(self):
        """Test adding a RandomVariable to a scalar (reverse add)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = 10 + X
        expected_values = pd.Series(
            [11, 13, 15],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(10+X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10+X)"

    def test_sub_two_random_variables(self):
        """Test subtracting two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = X - Y
        expected_values = pd.Series(
            [9, 27, 45],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X-Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-Y)"

    def test_sub_random_variable_and_scalar(self):
        """Test subtracting a scalar from a RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X - 5
        expected_values = pd.Series(
            [5, 25, 45],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X-5)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_variable(self):
        """Test subtracting a RandomVariable from a scalar (reverse sub)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = 10 - X
        expected_values = pd.Series(
            [9, 7, 5],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(10-X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10-X)"

    def test_mul_two_random_variables(self):
        """Test multiplying two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 6},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X * Y
        expected_values = pd.Series(
            [20, 120, 300],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X*Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*Y)"

    def test_mul_random_variable_and_scalar(self):
        """Test multiplying a RandomVariable by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = X * 10
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X*10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_variable(self):
        """Test multiplying a scalar by a RandomVariable (reverse mul)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )
        Z = 10 * X
        expected_values = pd.Series(
            [10, 30, 50],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(10*X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(10*X)"

    def test_truediv_two_random_variables(self):
        """Test dividing two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 100, 1: 300, 2: 500},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X / Y
        expected_values = pd.Series(
            [10.0, 10.0, 10.0],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X/Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/Y)"

    def test_truediv_random_variable_and_scalar(self):
        """Test dividing a RandomVariable by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 10, 1: 30, 2: 50},
        )
        Z = X / 10
        expected_values = pd.Series(
            [1.0, 3.0, 5.0],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X/10)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X/10)"

    def test_rtruediv_scalar_and_random_variable(self):
        """Test dividing a scalar by a RandomVariable (reverse div)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 2, 1: 5, 2: 20},
        )
        Z = 100 / X
        expected_values = pd.Series(
            [50.0, 20.0, 5.0],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(100/X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(100/X)"

    def test_pow_two_random_variables(self):
        """Test exponentiating two RandomVariables with same domain."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 6},
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {0: 2, 1: 2, 2: 2},
        )
        Z = X**Y
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X**Y)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**Y)"

    def test_pow_random_variable_and_scalar(self):
        """Test exponentiating a RandomVariable by a scalar."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 6},
        )
        Z = X**2
        expected_values = pd.Series(
            [4, 16, 36],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X**2)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_variable(self):
        """Test exponentiating a scalar by a RandomVariable (reverse pow)."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 2, 1: 4, 2: 0},
        )
        Z = 2**X
        expected_values = pd.Series(
            [4, 16, 1],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(2**X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "(2**X)"

    def test_add_with_different_probability_spaces_raises_error(self):
        """Test that adding RandomVariables with different probability spaces raises ValueError."""
        Omega1 = SampleSpace().from_list(["a", "b", "c"])
        Omega2 = SampleSpace().from_list(["x", "y", "z"])
        X = RandomVariable(domain=Omega1, name="X").from_dict(
            {"a": 1, "b": 3, "c": 5},
        )
        Y = RandomVariable(domain=Omega2, name="Y").from_dict(
            {"x": 1, "y": 3, "z": 5},
        )

        with pytest.raises(ValueError, match="different probability spaces"):
            Z = X + Y  # noqa: F841

    def test_add_with_non_random_variable_raises_error(self):
        """Test that adding a non-RandomVariable and non-scalar raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {0: 1, 1: 3, 2: 5},
        )

        with pytest.raises(TypeError):
            Z = X + "invalid"  # noqa: F841


# --------------------- comparison --------------------- #


class TestComparisonOperators:
    def test_lt_two_random_vectors(self):
        """Test less than comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (1, 2), 1: (2, 3), 2: (3, 4)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
        )
        result = X < Y
        expected_data = pd.DataFrame(
            [[False, True], [False, True], [False, False]],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X < Y)_0", "(X < Y)_1"], name="feature"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X < Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_le_two_random_vectors(self):
        """Test less than or equal comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (1, 2), 1: (2, 3), 2: (3, 4)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {0: (1, 3), 1: (2, 4), 2: (3, 4)}
        )
        result = X <= Y
        expected_data = pd.DataFrame(
            [[True, True], [True, True], [True, True]],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X <= Y)_0", "(X <= Y)_1"], name="feature"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X <= Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_gt_two_random_vectors(self):
        """Test greater than comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (5, 6), 1: (3, 4), 2: (1, 2)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {0: (3, 5), 1: (3, 3), 2: (2, 3)}
        )
        result = X > Y
        expected_data = pd.DataFrame(
            [[True, True], [False, True], [False, False]],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X > Y)_0", "(X > Y)_1"], name="feature"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X > Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_ge_two_random_vectors(self):
        """Test greater than or equal comparison of two RandomVectors."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (5, 6), 1: (3, 4), 2: (1, 2)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {0: (5, 5), 1: (3, 4), 2: (2, 3)}
        )
        result = X >= Y
        expected_data = pd.DataFrame(
            [[True, True], [True, True], [False, False]],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index(["(X >= Y)_0", "(X >= Y)_1"], name="feature"),
        )

        assert isinstance(result, RandomVector)
        assert result.name == "(X >= Y)"
        assert result.domain == Omega
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_lt_random_variables(self):
        """Test less than comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 2, 1: 2, 2: 1})
        result = X < Y
        expected_data = pd.Series(
            [True, False, False],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X < Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X < Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_le_random_variables(self):
        """Test less than or equal comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 2, 1: 2, 2: 1})
        result = X <= Y
        expected_data = pd.Series(
            [True, True, False],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X <= Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X <= Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_gt_random_variables(self):
        """Test greater than comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 5, 1: 3, 2: 1})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 2, 1: 3, 2: 2})
        result = X > Y
        expected_data = pd.Series(
            [True, False, False],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X > Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X > Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_ge_random_variables(self):
        """Test greater than or equal comparison of two RandomVariables."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVariable(domain=Omega, name="X").from_dict({0: 5, 1: 3, 2: 1})
        Y = RandomVariable(domain=Omega, name="Y").from_dict({0: 2, 1: 3, 2: 2})
        result = X >= Y
        expected_data = pd.Series(
            [True, True, False],
            index=pd.Index([0, 1, 2], name="sample"),
            name="(X >= Y)",
        )

        assert isinstance(result, RandomVariable)
        assert result.name == "(X >= Y)"
        assert result.domain == Omega
        pd.testing.assert_series_equal(result.data, expected_data)

    def test_lt_random_vector_and_scalar(self):
        """Test less than comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(domain=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X < 5, 5 > X]
        expected_data = pd.DataFrame(
            [[True, True], [True, False]],
            index=Omega.data,
            columns=pd.Index(["(X < 5)_0", "(X < 5)_1"], name="feature"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X < 5)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_le_random_vector_and_scalar(self):
        """Test less than or equal comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(domain=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X <= 3, 3 >= X]
        expected_data = pd.DataFrame(
            [[True, True], [True, False]],
            index=Omega.data,
            columns=pd.Index(["(X <= 3)_0", "(X <= 3)_1"], name="feature"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X <= 3)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_gt_random_vector_and_scalar(self):
        """Test greater than comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(domain=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X > 2, 2 < X]
        expected_data = pd.DataFrame(
            [[False, False], [True, True]],
            index=Omega.data,
            columns=pd.Index(["(X > 2)_0", "(X > 2)_1"], name="feature"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X > 2)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_ge_random_vector_and_scalar(self):
        """Test greater than or equal comparison of a RandomVector and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVector(domain=Omega).from_dict(
            {
                0: (1, 2),
                1: (3, 5),
            }
        )
        results = [X >= 2, 2 <= X]
        expected_data = pd.DataFrame(
            [[False, True], [True, True]],
            index=Omega.data,
            columns=pd.Index(["(X >= 2)_0", "(X >= 2)_1"], name="feature"),
        )

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X >= 2)"
            assert result.domain == Omega
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_lt_random_variable_and_scalar(self):
        """Test less than comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X < 3, 3 > X]
        expected_data = pd.Series([True, False], index=Omega.data, name="(X < 3)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X < 3)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_le_random_variable_and_scalar(self):
        """Test less than or equal comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X <= 3, 3 >= X]
        expected_data = pd.Series([True, True], index=Omega.data, name="(X <= 3)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X <= 3)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_gt_random_variable_and_scalar(self):
        """Test greater than comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X > 1, 1 < X]
        expected_data = pd.Series([False, True], index=Omega.data, name="(X > 1)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X > 1)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_ge_random_variable_and_scalar(self):
        """Test greater than or equal comparison of a RandomVariable and scalar."""
        Omega = SampleSpace().from_sequence(size=2)
        X = RandomVariable(domain=Omega).from_dict(
            {
                0: 1,
                1: 3,
            }
        )
        results = [X >= 1, 1 <= X]
        expected_data = pd.Series([True, True], index=Omega.data, name="(X >= 1)")

        for result in results:
            assert isinstance(result, RandomVector)
            assert result.name == "(X >= 1)"
            assert result.domain == Omega
            pd.testing.assert_series_equal(result.data, expected_data)

    def test_lt_with_different_domains_raises(self):
        """Test that comparing RandomVectors with different domains raises ValueError."""
        Omega1 = SampleSpace().from_list(["a", "b", "c"])
        Omega2 = SampleSpace().from_list(["x", "y", "z"])
        X = RandomVector(domain=Omega1, name="X").from_dict(
            {"a": (1, 2), "b": (3, 4), "c": (5, 6)}
        )
        Y = RandomVector(domain=Omega2, name="Y").from_dict(
            {"x": (1, 2), "y": (3, 4), "z": (5, 6)}
        )

        with pytest.raises(ValueError, match="must have the same domain"):
            _ = X < Y

    def test_lt_with_different_dimensions_raises(self):
        """Test that comparing RandomVectors with different dimensions raises ValueError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {0: (1, 2, 3), 1: (3, 4, 5), 2: (5, 6, 7)}
        )

        with pytest.raises(ValueError, match="must have the same dimension"):
            _ = X < Y

    def test_lt_with_non_random_vector_raises(self):
        """Test that comparing RandomVector with non-RandomVector raises TypeError."""
        Omega = SampleSpace().from_sequence(size=3)
        X = RandomVector(domain=Omega, name="X").from_dict(
            {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        )

        with pytest.raises(TypeError, match="must be a RandomVector"):
            _ = X < "not a random vector"


class TestBooleanMethods:
    def test_all_returns_true_when_all_true(self):
        """Test that all() returns True when all values are True."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(domain=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {"omega_0": (0, 1), "omega_1": (1, 2), "omega_2": (2, 3)}
        )
        result = X > Y

        assert result.all() is True

    def test_all_returns_false_when_some_false(self):
        """Test that all() returns False when some values are False."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(domain=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        assert result.all() is False

    def test_any_returns_true_when_some_true(self):
        """Test that any() returns True when at least one value is True."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(domain=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        assert result.any() is True

    def test_any_returns_false_when_all_false(self):
        """Test that any() returns False when all values are False."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(domain=Omega, name="X").from_dict(
            {"omega_0": (5, 6), "omega_1": (7, 8), "omega_2": (9, 10)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        )
        result = X < Y

        assert result.any() is False

    def test_all_with_random_variable(self):
        """Test all() method with RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 2, "omega_2": 3}
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {"omega_0": 0, "omega_1": 1, "omega_2": 2}
        )
        result = X > Y

        assert result.all() is True

    def test_any_with_random_variable(self):
        """Test any() method with RandomVariable."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVariable(domain=Omega, name="X").from_dict(
            {"omega_0": 1, "omega_1": 2, "omega_2": 3}
        )
        Y = RandomVariable(domain=Omega, name="Y").from_dict(
            {"omega_0": 2, "omega_1": 2, "omega_2": 1}
        )
        result = X < Y

        assert result.any() is True

    def test_bool_raises_value_error(self):
        """Test that __bool__() raises ValueError to prevent ambiguous boolean conversion."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(domain=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        with pytest.raises(
            ValueError, match="truth value of a RandomVector is ambiguous"
        ):
            bool(result)

    def test_bool_in_if_statement_raises(self):
        """Test that using RandomVector in if statement raises ValueError."""
        Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        X = RandomVector(domain=Omega, name="X").from_dict(
            {"omega_0": (1, 2), "omega_1": (2, 3), "omega_2": (3, 4)}
        )
        Y = RandomVector(domain=Omega, name="Y").from_dict(
            {"omega_0": (-2, 3), "omega_1": (1, 4), "omega_2": (-2, 1)}
        )
        result = X < Y

        with pytest.raises(
            ValueError, match="truth value of a RandomVector is ambiguous"
        ):
            if result:
                pass
