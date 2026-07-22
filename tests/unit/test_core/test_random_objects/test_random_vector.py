import pandas as pd
import pytest

from sigalg.core import (
    Index,
    ProbabilityMeasure,
    MeasureSpace,
    RandomVariable,
    RandomVector,
    SampleSpace,
    SigmaAlgebra,
)

# --------------------- test constructors --------------------- #


class TestConstructor:
    @pytest.fixture
    def dict_2d(self):
        return {0: (1, 2), 1: (3, 4), 2: (5, 6)}

    @pytest.fixture
    def dict_1d(self):
        return {0: 10, 1: 20, 2: 30}

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            data=[(1, 2), (3, 4), (5, 6)],
            index=pd.Index(["a", "b", "c"], name="letter"),
            columns=pd.Index(["black", "blue"], name="color"),
        )

    @pytest.fixture
    def series(self):
        return pd.Series(
            data=[1, 2, 3],
            index=pd.Index(["a", "b", "c"], name="letter"),
            name="Y",
        )

    def test_constructor_no_parameters(self):
        """Test the constructor with no parameters."""
        X = RandomVector()
        prob_space = MeasureSpace()

        assert X.data is None
        assert X.atom_data is None
        assert X.components is None
        assert X.index is None
        assert X.generated_sig_alg is None
        assert X.prob_space == prob_space
        assert X.sample_space is None
        assert X.sig_alg is None
        assert X.prob_measure is None
        assert X.range is None

    def test_2d_with_no_provided_domain_no_provided_index(self, dict_2d):
        """Test from dict with no provided prob space."""
        rv = RandomVector(mapping=dict_2d, name="Z")
        expected_sample_space = SampleSpace.from_sequence(size=3)
        expected_index = Index.from_sequence(size=2)
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=pd.Index([0, 1, 2], name="sample"),
            columns=pd.Index([0, 1], name="index"),
        )

        assert rv.sample_space == expected_sample_space
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, expected_data)

    def test_2d_with_provided_aligned_domain_no_provided_index(self, dict_2d):
        """Test from dict with a provided aligned domain, but no provided index."""
        Omega = SampleSpace().from_sequence(size=3)
        rv = RandomVector(sample_space=Omega, mapping=dict_2d, name="Z")
        expected_index = Index.from_sequence(size=2)
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=Omega.data,
            columns=expected_index.data,
        )

        assert rv.sample_space == Omega
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, expected_data)

    def test_2d_with_no_provided_domain_provided_correct_length_index(self, dict_2d):
        """Test from dict with no provided domain, but a provided correct-length index."""
        index = Index(["A", "B"])
        rv = RandomVector(index=index, mapping=dict_2d, name="Z")
        expected_sample_space = SampleSpace.from_sequence(size=3)
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=expected_sample_space.data,
            columns=index.data,
        )

        assert rv.sample_space == expected_sample_space
        assert rv.index == index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, expected_data)

    def test_2d_with_provided_aligned_domain_provided_correct_length_index(
        self, dict_2d
    ):
        """Test from dict with both a provided aligned domain and correct-length index."""
        Omega = SampleSpace.from_sequence(size=3)
        index = Index(["A", "B"])
        rv = RandomVector(sample_space=Omega, index=index, mapping=dict_2d, name="Z")
        expected_data = pd.DataFrame(
            [(1, 2), (3, 4), (5, 6)],
            index=Omega.data,
            columns=index.data,
        )

        assert rv.sample_space == Omega
        assert rv.index == index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, expected_data)

    def test_1d_with_no_provided_domain(self, dict_1d):
        """Test from dict with no provided domain at construction for 1D output."""
        rv = RandomVector(mapping=dict_1d, name="Y")
        expected_sample_space = SampleSpace.from_sequence(size=3)
        expected_index = None
        expected_data = pd.Series(
            [10, 20, 30],
            index=expected_sample_space.data,
            name="Y",
        )

        assert rv.sample_space == expected_sample_space
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, expected_data)

    def test_1d_with_provided_aligned_domain(self, dict_1d):
        """Test from dict with provided aligned domain at construction for 1D output."""
        Omega = SampleSpace.from_sequence(size=3)
        rv = RandomVector(sample_space=Omega, mapping=dict_1d, name="Y")
        expected_index = None
        expected_data = pd.Series(
            [10, 20, 30],
            index=Omega.data,
            name="Y",
        )

        assert rv.sample_space == Omega
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, expected_data)

    def test_measurability(self):
        """Test from dict raises error for non-measurable mapping."""
        Omega = SampleSpace.from_sequence(size=3)
        F = SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 1,
                2: 1,
            },
        )
        mapping = {
            0: (1, 2),
            1: (3, 4),
            2: (5, 6),
        }

        with pytest.raises(
            ValueError,
            match="Random vector Z is not measurable",
        ):
            RandomVector(sample_space=Omega, sig_alg=F, mapping=mapping, name="Z")

    def test_2d_df_with_no_provided_domain_index(self, df):
        """Test from pandas with no provided domain and index at construction."""
        rv = RandomVector(mapping=df, name="Z")
        expected_sample_space = SampleSpace(["a", "b", "c"], variable_names=["letter"])
        expected_index = Index(["black", "blue"], variable_names=["color"])

        assert rv.sample_space == expected_sample_space
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    def test_2d_df_with_provided_aligned_domain_no_provided_index(self, df):
        """Test from pandas with a provided aligned domain, but no provided index."""
        Omega = SampleSpace(["a", "b", "c"], variable_names=["letter"])
        rv = RandomVector(sample_space=Omega, mapping=df, name="Z")
        expected_index = Index(["black", "blue"], variable_names=["color"])

        assert rv.sample_space == Omega
        assert rv.index == expected_index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    def test_2d_df_with_no_provided_domain_provided_aligned_index(self, df):
        """Test from pandas with a no provided domain, but a provided aligned index."""
        index = Index(["black", "blue"], variable_names=["color"])
        rv = RandomVector(index=index, mapping=df, name="Z")
        expected_domain = SampleSpace(["a", "b", "c"], variable_names=["letter"])

        assert rv.sample_space == expected_domain
        assert rv.index == index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    def test_2d_df_with_provided_aligned_domain_provided_aligned_index(self, df):
        """Test from pandas with both a provided aligned domain and index."""
        Omega = SampleSpace(["a", "b", "c"], variable_names=["letter"])
        index = Index(["black", "blue"], variable_names=["color"])
        rv = RandomVector(sample_space=Omega, index=index, mapping=df, name="Z")

        assert rv.sample_space == Omega
        assert rv.index == index
        assert rv.name == "Z"
        pd.testing.assert_frame_equal(rv.data, df)

    def test_1d_series_with_no_provided_domain(self, series):
        """Test from pandas with no provided domain at construction."""
        rv = RandomVector(name="Y", mapping=series)
        expected_domain = SampleSpace(["a", "b", "c"], variable_names=["letter"])
        expected_index = None

        assert rv.sample_space == expected_domain
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, series)

    def test_1d_series_with_provided_aligned_domain(self, series):
        """Test from pandas with provided aligned domain at construction."""
        Omega = SampleSpace(["a", "b", "c"], variable_names=["letter"])
        rv = RandomVector(sample_space=Omega, mapping=series, name="Y")
        expected_index = None

        assert rv.sample_space == Omega
        assert rv.index == expected_index
        assert rv.name == "Y"
        pd.testing.assert_series_equal(rv.data, series)


class TestFromConstant:
    def test_from_constant_2d(self):
        """Test the from_constant method with a 2-dimensional output."""
        Omega = SampleSpace.from_sequence(size=3)
        X = RandomVector.from_constant(sample_space=Omega, constant=(1, 2))
        expected_index = Index.from_sequence(size=2)
        expected_data = pd.DataFrame(
            [(1, 2)] * 3, index=Omega.data, columns=expected_index.data
        )

        pd.testing.assert_frame_equal(X.data, expected_data)

    def test_from_constant_1d(self):
        """Test the from_constant method with a 1-dimensional output."""
        Omega = SampleSpace.from_sequence(size=3)
        X = RandomVector.from_constant(sample_space=Omega, constant=2)
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


class TestGeneratedSigAlg:
    def test_generated_sigma_algebra_property(self):
        """Test generated_sigma_algebra property of RandomVector."""
        Omega = SampleSpace.from_sequence(size=3)
        outputs_2d = {0: (1, 2), 1: (3, 4), 2: (5, 6)}
        outputs_1d = {0: 10, 1: 20, 2: 30}
        X = RandomVector(sample_space=Omega, mapping=outputs_2d)
        Y = RandomVector(sample_space=Omega, mapping=outputs_1d, name="Y")
        expected_sigma_algebra_2d = SigmaAlgebra(
            sample_space=Omega,
            mapping=outputs_2d,
            name="sigma(X)",
        )
        expected_sigma_algebra_1d = SigmaAlgebra(
            sample_space=Omega,
            mapping=outputs_1d,
            name="sigma(Y)",
        )

        assert X.generated_sig_alg == expected_sigma_algebra_2d
        assert Y.generated_sig_alg == expected_sigma_algebra_1d


class TestProbSpace:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.8,
                1: 0.2,
            },
        )

    @pytest.fixture
    def mapping(self):
        return {
            0: (1, 2),
            1: (1, 2),
            2: (3, 4),
        }

    def test_prob_space_with_defaults(self, Omega, mapping):
        """Test that default probability space has power-set sigma-algebra and uniform probability measure."""
        X = RandomVector(sample_space=Omega, mapping=mapping)
        prob_space = MeasureSpace(sample_space=Omega)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == SigmaAlgebra.power_set(Omega)
        assert X.prob_space.prob_measure == ProbabilityMeasure.uniform(
            sample_space=Omega
        )

    def test_prob_space_with_custom_prob_measure(self, Omega, P, mapping):
        """Test constructor with custom probability measure sets sigma-algebra to the sigma-algebra of the probability measure."""
        X = RandomVector(sample_space=Omega, prob_measure=P, mapping=mapping)
        prob_space = MeasureSpace(Omega, prob_measure=P)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == P.sig_alg
        assert X.prob_space.prob_measure == P

    def test_prob_space_with_custom_sigma_algebra(self, Omega, F, mapping):
        """Test constructor with custom sigma-algebra sets the probability measure to uniform over the sigma-algebra."""
        X = RandomVector(sample_space=Omega, sig_alg=F, mapping=mapping)
        prob_space = MeasureSpace(Omega, F)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.prob_measure == ProbabilityMeasure.uniform(sig_alg=F)

    def test_prob_space_with_all_components(self, Omega, F, P, mapping):
        """Test constructor with all components."""
        prob_space = MeasureSpace(Omega, F, P)
        X = RandomVector(*prob_space, mapping=mapping)

        assert X.prob_space == prob_space
        assert X.prob_space.sample_space == Omega
        assert X.prob_space.sig_alg == F
        assert X.prob_space.prob_measure == P


class TestRange:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=4)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 2,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.1,
                2: 0.7,
            },
        )

    @pytest.fixture
    def mapping_2d(self):
        return {
            0: (1, 2),
            1: (1, 2),
            2: (3, 4),
            3: (3, 4),
        }

    @pytest.fixture
    def mapping_1d(self):
        return {
            0: 4,
            1: 4,
            2: 5,
            3: 6,
        }

    def test_range_2d_random_vector(self, Omega, F, P, mapping_2d):
        """Test range property of 2D RandomVector."""
        X = RandomVector(Omega, F, P, mapping=mapping_2d)
        expected_sample_space = SampleSpace(
            [(1, 2), (3, 4)], name="X_range", variable_names=["X_0", "X_1"]
        )
        expected_sig_alg = SigmaAlgebra.power_set(expected_sample_space)
        prob_measure = ProbabilityMeasure(
            sig_alg=expected_sig_alg,
            name="P_X",
            mapping={
                (1, 2): 0.2,
                (3, 4): 0.8,
            },
        )
        expected_range = MeasureSpace(
            sig_alg=expected_sig_alg, prob_measure=prob_measure
        )

        assert X.range == expected_range

    def test_range_1d_random_vector(self, Omega, F, P, mapping_1d):
        """Test range property of 1D RandomVector."""
        X = RandomVector(Omega, F, P, mapping=mapping_1d)
        expected_sample_space = SampleSpace(
            [4, 5, 6], name="X_range", variable_names=["X"]
        )
        expected_sig_alg = SigmaAlgebra.power_set(expected_sample_space)
        expected_prob_measure = ProbabilityMeasure(
            sig_alg=expected_sig_alg,
            name="P_X",
            mapping={
                4: 0.2,
                5: 0.1,
                6: 0.7,
            },
        )
        expected_range = MeasureSpace(
            sig_alg=expected_sig_alg, prob_measure=expected_prob_measure
        )

        assert X.range == expected_range


# --------------------- test data access methods --------------------- #


class TestCallMethod:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=6)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra(
            sample_space=Omega,
            mapping={
                0: 0,
                1: 0,
                2: 1,
                3: 1,
                4: 2,
                5: 2,
            },
        )

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.3,
                1: 0.2,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return MeasureSpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (1, 2),
                2: (3, 4),
                3: (3, 4),
                4: (5, 6),
                5: (5, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 1,
                2: 3,
                3: 3,
                4: 5,
                5: 5,
            },
        )

    def test_call_method_on_sample_points(self, Omega, X, Y):
        """Test calling on sample points."""
        for sample_point in Omega:
            pd.testing.assert_series_equal(X(sample_point), X.data.loc[sample_point])
            assert Y(sample_point) == Y.data.loc[sample_point]

    def test_call_method_on_atoms(self, F, X, Y):
        """Test calling on atoms."""
        for atom_id, atom in F.atom_id_to_event.items():
            pd.testing.assert_series_equal(X(atom), X.atom_data.loc[atom_id])
            assert Y(atom) == Y.atom_data.loc[atom_id]


# --------------------- arithmetic --------------------- #


class TestArithmetic:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return MeasureSpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 4),
                2: (5, 6),
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (10, 20),
                1: (30, 40),
                2: (50, 60),
            },
        )

    def test_add_two_random_vectors(self, X, Y):
        """Test adding two RandomVectors."""
        Z = X + Y
        expected_data = pd.DataFrame(
            [(11, 22), (33, 44), (55, 66)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X+Y)"

    def test_add_random_vector_and_scalar(self, X):
        """Test adding a scalar to a RandomVector."""
        Z = X + 10
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_vector(self, X):
        """Test adding a RandomVector to a scalar (reverse add)."""
        Z = 10 + X
        expected_data = pd.DataFrame(
            [(11, 12), (13, 14), (15, 16)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10+X)"

    def test_sub_two_random_vectors(self, X, Y):
        """Test subtracting two RandomVectors."""
        Z = X - Y
        expected_values = pd.DataFrame(
            [(-9, -18), (-27, -36), (-45, -54)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_values)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-Y)"

    def test_sub_random_vector_and_scalar(self, X):
        """Test subtracting a scalar from a RandomVector."""
        Z = X - 5
        expected_data = pd.DataFrame(
            [(-4, -3), (-2, -1), (0, 1)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_vector(self, X):
        """Test subtracting a RandomVector from a scalar (reverse sub)."""
        Z = 5 - X
        expected_data = pd.DataFrame(
            [(4, 3), (2, 1), (0, -1)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(5-X)"

    def test_mul_two_random_vectors(self, X, Y):
        """Test multiplying two RandomVectors."""
        Z = X * Y
        expected_data = pd.DataFrame(
            [(10, 40), (90, 160), (250, 360)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*Y)"

    def test_mul_random_vector_and_scalar(self, X):
        """Test multiplying a RandomVector by a scalar."""
        Z = X * 10
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_vector(self, X):
        """Test multiplying a scalar by a RandomVector (reverse mul)."""
        Z = 10 * X
        expected_data = pd.DataFrame(
            [(10, 20), (30, 40), (50, 60)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10*X)"

    def test_truediv_two_random_vectors(self, X, Y):
        """Test dividing two RandomVectors."""
        Z = Y / X
        expected_data = pd.DataFrame(
            [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(Y/X)"

    def test_truediv_random_vector_and_scalar(self, Y):
        """Test dividing a RandomVector by a scalar."""
        Z = Y / 10
        expected_data = pd.DataFrame(
            [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            index=Y.sample_space.data,
            columns=Y.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(Y/10)"

    def test_rtruediv_scalar_and_random_vector(self, Y):
        """Test dividing a scalar by a RandomVector (reverse div)."""
        Z = 10 / Y
        expected_data = pd.DataFrame(
            [(1.0, 1 / 2), (1 / 3, 1 / 4), (1 / 5, 1 / 6)],
            index=Y.sample_space.data,
            columns=Y.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(10/Y)"

    def test_pow_two_random_vectors(self, prob_space):
        """Test exponentiating two RandomVectors."""
        X = RandomVector(*prob_space, mapping={0: (2, 3), 1: (4, 5), 2: (6, 7)})
        Y = RandomVector(
            *prob_space, name="Y", mapping={0: (2, 2), 1: (2, 2), 2: (2, 2)}
        )
        Z = X**Y
        expected_data = pd.DataFrame(
            [(4, 9), (16, 25), (36, 49)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self, X):
        """Test exponentiating a RandomVector by a scalar."""
        Z = X**2
        expected_data = pd.DataFrame(
            [(1, 4), (9, 16), (25, 36)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self, X):
        """Test exponentiating a scalar by a RandomVector (reverse pow)."""
        Z = 2**X
        expected_data = pd.DataFrame(
            [(2, 4), (8, 16), (32, 64)],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(2**X)"


class TestArithmeticWithRandomVariable:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return MeasureSpace(Omega, F, P)

    @pytest.fixture
    def X(self, prob_space):
        return RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 3,
                2: 5,
            },
        )

    @pytest.fixture
    def Y(self, prob_space):
        return RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 10,
                1: 30,
                2: 50,
            },
        )

    def test_add_two_random_variable(self, X, Y):
        """Test adding two RandomVariable."""
        Z = X + Y
        expected_data = pd.Series(
            [
                11,
                33,
                55,
            ],
            index=X.sample_space.data,
            name="(X+Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X+Y)"

    def test_add_random_variable_and_scalar(self, X):
        """Test adding a scalar to a RandomVariable."""
        Z = X + 10
        expected_data = pd.Series(
            [
                11,
                13,
                15,
            ],
            index=X.sample_space.data,
            name="(X+10)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X+10)"

    def test_radd_scalar_and_random_variable(self, X):
        """Test adding a RandomVariable to a scalar (reverse add)."""
        Z = 10 + X
        expected_data = pd.Series(
            [
                11,
                13,
                15,
            ],
            index=X.sample_space.data,
            name="(10+X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10+X)"

    def test_sub_two_random_variables(self, X, Y):
        """Test subtracting two RandomVariables."""
        Z = X - Y
        expected_values = pd.Series(
            [
                -9,
                -27,
                -45,
            ],
            index=X.sample_space.data,
            name="(X-Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-Y)"

    def test_sub_random_variable_and_scalar(self, X):
        """Test subtracting a scalar from a RandomVariable."""
        Z = X - 5
        expected_data = pd.Series(
            [
                -4,
                -2,
                0,
            ],
            index=X.sample_space.data,
            name="(X-5)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X-5)"

    def test_rsub_scalar_and_random_variable(self, X):
        """Test subtracting a RandomVariable from a scalar (reverse sub)."""
        Z = 5 - X
        expected_data = pd.Series(
            [
                4,
                2,
                0,
            ],
            index=X.sample_space.data,
            name="(5-X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(5-X)"

    def test_mul_two_random_variables(self, X, Y):
        """Test multiplying two RandomVariables."""
        Z = X * Y
        expected_data = pd.Series(
            [
                10,
                90,
                250,
            ],
            index=X.sample_space.data,
            name="(X*Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*Y)"

    def test_mul_random_variable_and_scalar(self, X):
        """Test multiplying a RandomVariable by a scalar."""
        Z = X * 10
        expected_data = pd.Series(
            [
                10,
                30,
                50,
            ],
            index=X.sample_space.data,
            name="(X*10)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X*10)"

    def test_rmul_scalar_and_random_variable(self, X):
        """Test multiplying a scalar by a RandomVariable (reverse mul)."""
        Z = 10 * X
        expected_data = pd.Series(
            [
                10,
                30,
                50,
            ],
            index=X.sample_space.data,
            name="(10*X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(10*X)"

    def test_truediv_two_random_variables(self, X, Y):
        """Test dividing two RandomVariables."""
        Z = Y / X
        expected_data = pd.Series(
            [
                10.0,
                10.0,
                10.0,
            ],
            index=X.sample_space.data,
            name="(Y/X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(Y/X)"

    def test_truediv_random_variable_and_scalar(self, Y):
        """Test dividing a RandomVariable by a scalar."""
        Z = Y / 10
        expected_data = pd.Series(
            [
                1.0,
                3.0,
                5.0,
            ],
            index=Y.sample_space.data,
            name="(Y/10)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(Y/10)"

    def test_rtruediv_scalar_and_random_variable(self, Y):
        """Test dividing a scalar by a RandomVariable (reverse div)."""
        Z = 10 / Y
        expected_data = pd.Series(
            [
                1.0,
                1 / 3,
                1 / 5,
            ],
            index=Y.sample_space.data,
            name="(10/Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == Y.prob_space
        assert Z.name == "(10/Y)"

    def test_pow_two_random_variables(self, prob_space):
        """Test exponentiating two RandomVariable."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 4,
                2: 6,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 2,
            },
        )
        Z = X**Y
        expected_data = pd.Series(
            [
                4,
                16,
                36,
            ],
            index=X.sample_space.data,
            name="(X**Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**Y)"

    def test_pow_random_vector_and_scalar(self, X):
        """Test exponentiating a RandomVariable by a scalar."""
        Z = X**2
        expected_data = pd.Series(
            [
                1,
                9,
                25,
            ],
            index=X.sample_space.data,
            name="(X**2)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(X**2)"

    def test_rpow_scalar_and_random_vector(self, X):
        """Test exponentiating a scalar by a RandomVariable (reverse pow)."""
        Z = 2**X
        expected_data = pd.Series(
            [
                2,
                8,
                32,
            ],
            index=X.sample_space.data,
            name="(2**X)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.prob_space == X.prob_space
        assert Z.name == "(2**X)"


# --------------------- comparison --------------------- #


class TestComparisonOperators:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return MeasureSpace(Omega, F, P)

    def test_lt_two_random_vectors(self, prob_space):
        """Test less than comparison of two RandomVectors."""
        X = RandomVector(*prob_space, mapping={0: (1, 2), 1: (2, 3), 2: (3, 4)})
        Y = RandomVector(
            *prob_space, name="Y", mapping={0: (-2, 3), 1: (1, 4), 2: (-2, 1)}
        )
        Z = X < Y
        expected_data = pd.DataFrame(
            [[False, True], [False, True], [False, False]],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X < Y)"
        assert Z.prob_space == X.prob_space

    def test_le_two_random_vectors(self, prob_space):
        """Test less than or equal comparison of two RandomVectors."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (-1, 3),
                1: (2, 4),
                2: (3, 4),
            },
        )
        Z = X <= Y
        expected_data = pd.DataFrame(
            [[False, True], [True, True], [True, True]],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X <= Y)"
        assert Z.prob_space == X.prob_space

    def test_gt_two_random_vectors(self, prob_space):
        """Test greater than comparison of two RandomVectors."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (-1, 3),
                1: (2, 4),
                2: (3, 4),
            },
        )
        Z = X > Y
        expected_data = pd.DataFrame(
            [[True, False], [True, True], [False, False]],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X > Y)"
        assert Z.prob_space == X.prob_space

    def test_ge_two_random_vectors(self, prob_space):
        """Test greater than or equal comparison of two RandomVectors."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (-1, 3),
                1: (2, 4),
                2: (3, 4),
            },
        )
        Z = X >= Y
        expected_data = pd.DataFrame(
            [[True, False], [True, True], [True, True]],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        pd.testing.assert_frame_equal(Z.data, expected_data)
        assert Z.name == "(X >= Y)"
        assert Z.prob_space == X.prob_space

    def test_lt_random_variables(self, prob_space):
        """Test less than comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X < Y
        expected_data = pd.Series(
            [True, False, False],
            index=X.sample_space.data,
            name="(X < Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X < Y)"
        assert Z.prob_space == X.prob_space

    def test_le_random_variables(self, prob_space):
        """Test less than or equal comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X <= Y
        expected_data = pd.Series(
            [True, True, False],
            index=X.sample_space.data,
            name="(X <= Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X <= Y)"
        assert Z.prob_space == X.prob_space

    def test_gt_random_variables(self, prob_space):
        """Test greater than comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X > Y
        expected_data = pd.Series(
            [False, False, True],
            index=X.sample_space.data,
            name="(X > Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X > Y)"
        assert Z.prob_space == X.prob_space

    def test_ge_random_variables(self, prob_space):
        """Test greater than or equal comparison of two RandomVariables."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 2,
                1: 2,
                2: 1,
            },
        )
        Z = X >= Y
        expected_data = pd.Series(
            [False, True, True],
            index=X.sample_space.data,
            name="(X >= Y)",
        )

        pd.testing.assert_series_equal(Z.data, expected_data)
        assert Z.name == "(X >= Y)"
        assert Z.prob_space == X.prob_space

    def test_lt_random_vector_and_scalar(self, prob_space):
        """Test less than comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 5),
            },
        )
        results = [X < 5, 5 > X]
        expected_data = pd.DataFrame(
            [[True, True], [True, False], [True, False]],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X < 5)"
            assert result.prob_space == X.prob_space

    def test_le_random_vector_and_scalar(self, prob_space):
        """Test less than or equal comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 5),
            },
        )
        results = [X <= 5, 5 >= X]
        expected_data = pd.DataFrame(
            [[True, True], [True, True], [True, True]],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X <= 5)"
            assert result.prob_space == X.prob_space

    def test_gt_random_vector_and_scalar(self, prob_space):
        """Test greater than comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 6),
            },
        )
        results = [X > 5, 5 < X]
        expected_data = pd.DataFrame(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X > 5)"
            assert result.prob_space == X.prob_space

    def test_ge_random_vector_and_scalar(self, prob_space):
        """Test greater than or equal comparison of a RandomVector and scalar."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (3, 5),
                2: (4, 6),
            },
        )
        results = [X >= 5, 5 <= X]
        expected_data = pd.DataFrame(
            [
                [False, False],
                [False, True],
                [False, True],
            ],
            index=X.sample_space.data,
            columns=X.index.data,
        )

        for result in results:
            pd.testing.assert_frame_equal(result.data, expected_data)
            assert result.name == "(X >= 5)"
            assert result.prob_space == X.prob_space

    def test_lt_random_variable_and_scalar(self, prob_space):
        """Test less than comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X < 5, 5 > X]
        expected_data = pd.Series(
            [
                True,
                False,
                False,
            ],
            index=X.sample_space.data,
            name="(X < 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X < 5)"
            assert result.prob_space == X.prob_space

    def test_le_random_variable_and_scalar(self, prob_space):
        """Test less than or equal comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X <= 5, 5 >= X]
        expected_data = pd.Series(
            [
                True,
                True,
                False,
            ],
            index=X.sample_space.data,
            name="(X <= 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X <= 5)"
            assert result.prob_space == X.prob_space

    def test_gt_random_variable_and_scalar(self, prob_space):
        """Test greater than comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X > 5, 5 < X]
        expected_data = pd.Series(
            [
                False,
                False,
                True,
            ],
            index=X.sample_space.data,
            name="(X > 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X > 5)"
            assert result.prob_space == X.prob_space

    def test_ge_random_variable_and_scalar(self, prob_space):
        """Test greater than or equal comparison of a RandomVariable and scalar."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 2,
                1: 5,
                2: 6,
            },
        )
        results = [X >= 5, 5 <= X]
        expected_data = pd.Series(
            [
                False,
                True,
                True,
            ],
            index=X.sample_space.data,
            name="(X >= 5)",
        )

        for result in results:
            pd.testing.assert_series_equal(result.data, expected_data)
            assert result.name == "(X >= 5)"
            assert result.prob_space == X.prob_space


class TestBooleanMethods:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=3)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def P(self, F):
        return ProbabilityMeasure(
            sig_alg=F,
            mapping={
                0: 0.2,
                1: 0.3,
                2: 0.5,
            },
        )

    @pytest.fixture
    def prob_space(self, Omega, F, P):
        return MeasureSpace(Omega, F, P)

    def test_all_returns_true_when_all_true(self, prob_space):
        """Test that all() returns True when all values are True."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (0, 1),
                1: (1, 2),
                2: (2, 3),
            },
        )
        Z = X > Y

        assert Z.all() is True

    def test_all_returns_false_when_some_false(self, prob_space):
        """Test that all() returns False when some values are False."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (1, 1),
                1: (1, 2),
                2: (2, 3),
            },
        )
        Z = X > Y

        assert Z.all() is False

    def test_any_returns_true_when_some_true(self, prob_space):
        """Test that any() returns True when at least one value is True."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 6),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 5),
            },
        )
        Z = X > Y

        assert Z.any() is True

    def test_any_returns_false_when_all_false(self, prob_space):
        """Test that any() returns False when all values are False."""
        X = RandomVector(
            *prob_space,
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Y = RandomVector(
            *prob_space,
            name="Y",
            mapping={
                0: (1, 2),
                1: (2, 3),
                2: (3, 4),
            },
        )
        Z = X > Y

        assert Z.any() is False

    def test_all_with_random_variable(self, prob_space):
        """Test all() method with RandomVariable."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 0,
                1: 1,
                2: 2,
            },
        )
        Z = X > Y

        assert Z.all() is True

    def test_any_with_random_variable(self, prob_space):
        """Test any() method with RandomVariable."""
        X = RandomVariable(
            *prob_space,
            mapping={
                0: 1,
                1: 2,
                2: 3,
            },
        )
        Y = RandomVariable(
            *prob_space,
            name="Y",
            mapping={
                0: 1,
                1: 2,
                2: 2,
            },
        )
        Z = X > Y

        assert Z.any() is True
