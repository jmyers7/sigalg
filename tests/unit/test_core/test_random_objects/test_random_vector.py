import pandas as pd
import pytest

from sigalg.core import FeatureIndex, RandomVariable, RandomVector, SampleSpace


class TestConstructor:

    @pytest.fixture
    def Omega(self):
        return SampleSpace.generate_default(size=3)

    @pytest.fixture
    def values_basic(self):
        return pd.DataFrame([[1, 2], [3, 4], [5, 6]])

    @pytest.fixture
    def values_with_indices(self):
        return pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["a", "b", "c"], name="letters"),
            columns=pd.Index(["black", "blue"], name="colors"),
        )

    def test_construction_rvs(self, Omega):
        X1 = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5}, domain=Omega, name="X1"
        )
        X2 = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6}, domain=Omega, name="X2"
        )
        components = [X1, X2]
        X = RandomVector(components=components)
        assert X.components == components

    def test_construction_values_basic(self, values_basic):
        Y = RandomVector(values=values_basic, name="Y")
        pd.testing.assert_frame_equal(Y.values, values_basic)

    def test_construction_values_with_indices(self, values_with_indices):
        X = RandomVector(values=values_with_indices)
        pd.testing.assert_frame_equal(X.values, values_with_indices)


class TestValues:

    def test_values_from_rvs(self):
        Omega = SampleSpace.generate_default(size=3)
        X1 = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5}, domain=Omega, name="X1"
        )
        X2 = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6}, domain=Omega, name="X2"
        )
        components = [X1, X2]
        X = RandomVector(components=components)
        expected_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["omega0", "omega1", "omega2"], name="sample"),
            columns=pd.Index(["X1", "X2"], name="feature"),
        )
        pd.testing.assert_frame_equal(X.values, expected_df)


class TestComponents:

    def test_components_from_values_basic(self):
        values = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        X = RandomVector(values=values)
        sample_space = SampleSpace([0, 1, 2])
        X0 = RandomVariable(outputs={0: 1, 1: 3, 2: 5}, domain=sample_space, name=0)
        X1 = RandomVariable(outputs={0: 2, 1: 4, 2: 6}, domain=sample_space, name=1)
        expected_components = [X0, X1]
        assert X.components == expected_components

    def test_components_from_values(self):
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["a", "b", "c"], name="letters"),
            columns=pd.Index(["black", "blue"], name="colors"),
        )
        Y = RandomVector(values=values, name="Y")
        sample_space = SampleSpace(["a", "b", "c"])
        black = RandomVariable(
            outputs={"a": 1, "b": 3, "c": 5}, domain=sample_space, name="black"
        )
        blue = RandomVariable(
            outputs={"a": 2, "b": 4, "c": 6}, domain=sample_space, name="blue"
        )
        expected_components = [black, blue]
        assert Y.components == expected_components


class TestDomain:

    def test_domain_from_rvs(self):
        sample_space = SampleSpace.generate_default(size=3)
        X1 = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=sample_space,
            name="X1",
        )
        X2 = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6},
            domain=sample_space,
            name="X2",
        )
        components = [X1, X2]
        X = RandomVector(components=components)
        assert X.domain == sample_space
        assert X.domain.name == "Omega"
        assert X.components[0].domain == sample_space
        X.domain.name = "S"
        assert X.domain == sample_space
        assert X.domain.name == "S"
        assert X.components[0].domain == sample_space

    def test_domain_from_values_basic(self):
        values = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        X = RandomVector(values=values)
        expected_sample_space = SampleSpace([0, 1, 2], name="Omega")
        assert X.domain == expected_sample_space
        assert X.domain.name == "Omega"
        assert X.components[0].domain == expected_sample_space
        X.domain.name = "S"
        assert X.domain == expected_sample_space
        assert X.domain.name == "S"
        assert X.components[0].domain == expected_sample_space

    def test_domain_from_values(self):
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["a", "b", "c"], name="letters"),
            columns=pd.Index(["black", "blue"], name="colors"),
        )
        X = RandomVector(values=values)
        expected_sample_space = SampleSpace(["a", "b", "c"], name="Omega")
        assert X.domain == expected_sample_space
        assert X.domain.name == "Omega"
        assert X.components[0].domain == expected_sample_space
        X.domain.name = "S"
        assert X.domain == expected_sample_space
        assert X.domain.name == "S"
        assert X.components[0].domain == expected_sample_space


class TestFeatureIndex:

    def test_feature_index_from_rvs(self):
        sample_space = SampleSpace.generate_default(size=3)
        X1 = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 5},
            domain=sample_space,
            name="X1",
        )
        X2 = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 6},
            domain=sample_space,
            name="X2",
        )
        components = [X1, X2]
        X = RandomVector(components=components)
        expected_index = FeatureIndex(indices=["X1", "X2"], values_name="feature")
        pd.testing.assert_index_equal(X.feature_index.values, expected_index.values)
        new_feature_index = FeatureIndex(indices=["Y1", "Y2"], values_name="variables")
        X.feature_index = new_feature_index
        assert X1.name == "Y1"
        assert X2.name == "Y2"
        pd.testing.assert_index_equal(X.values.columns, new_feature_index.values)

    def test_domain_from_values_basic(self):
        values = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
        X = RandomVector(values=values)
        expected_index = FeatureIndex(indices=[0, 1], values_name=None)
        pd.testing.assert_index_equal(X.feature_index.values, expected_index.values)
        new_feature_index = FeatureIndex(indices=["Y1", "Y2"], values_name="variables")
        X.feature_index = new_feature_index
        Y1 = X.components[0]
        Y2 = X.components[1]
        assert Y1.name == "Y1"
        assert Y2.name == "Y2"
        pd.testing.assert_index_equal(X.values.columns, new_feature_index.values)

    def test_domain_from_values(self):
        values = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=pd.Index(["a", "b", "c"], name="letters"),
            columns=pd.Index(["black", "blue"], name="colors"),
        )
        X = RandomVector(values=values)
        expected_index = FeatureIndex(indices=["black", "blue"], values_name="colors")
        pd.testing.assert_index_equal(X.feature_index.values, expected_index.values)
        new_feature_index = FeatureIndex(indices=["Y1", "Y2"], values_name="variables")
        X.feature_index = new_feature_index
        Y1 = X.components[0]
        Y2 = X.components[1]
        assert Y1.name == "Y1"
        assert Y2.name == "Y2"
        pd.testing.assert_index_equal(X.values.columns, new_feature_index.values)


class TestRange:

    def test_range_from_rvs(self):
        sample_space = SampleSpace.generate_default(size=3)
        X1 = RandomVariable(
            outputs={"omega0": 1, "omega1": 3, "omega2": 3},
            domain=sample_space,
            name="X1",
        )
        X2 = RandomVariable(
            outputs={"omega0": 2, "omega1": 4, "omega2": 4},
            domain=sample_space,
            name="X2",
        )
        components = [X1, X2]
        X = RandomVector(components=components)
        expected_df = pd.DataFrame(
            data=[[1, 2], [3, 4]],
            index=pd.Index(["x0", "x1"], name="output"),
            columns=pd.Index(["X1", "X2"], name="feature"),
        )
        pd.testing.assert_frame_equal(X.range.values, expected_df)
        assert X.range.name == "range(X)"
