import pandas as pd
import pytest

from sigalg.core import Domain, MultivariateFunction

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test base constructor with no parameters."""
        f = MultivariateFunction()

        assert f.name == "f"
        assert f.data is None
        assert f.function is None
        assert f.parameter_list is None
        assert f.domain is None

    def test_constructor_with_all_parameters(self):
        """Test base constructor with all parameters."""
        domain = Domain().from_list([(0, 1), (1, 2)], data_name=["x", "y"])
        g = MultivariateFunction(domain=domain, name="g")

        assert g.name == "g"
        assert g.data is None
        assert g.function is None
        assert g.parameter_list is None
        assert g.domain is domain


class TestFromCallable:
    def test_with_two_variables(self):
        """Test from_callable with two variables."""

        def function(x, y):
            return x**2

        f = MultivariateFunction().from_callable(function)

        assert f.function is function
        assert f.num_parameters == 2
        assert f.parameter_list == ["x", "y"]
        assert f.data is None

    def test_with_one_variable(self):
        """Test from_callable with one variable."""

        def function(x):
            return x**2

        f = MultivariateFunction().from_callable(function)

        assert f.function is function
        assert f.num_parameters == 1
        assert f.parameter_list == ["x"]
        assert f.data is None

    def test_with_bivariate_lambda(self):
        """Test from_callable with a bivariate lambda function."""
        f = MultivariateFunction().from_callable(lambda x, y: x + y)

        assert f.function(1, 1) == 2
        assert f.function(0, 1) == 1
        assert f.function(1, 0) == 1
        assert f.num_parameters == 2
        assert f.parameter_list == ["x", "y"]
        assert f.data is None

    def test_with_univariate_lambda(self):
        """Test from_callable with a univariate lambda function."""
        f = MultivariateFunction().from_callable(lambda x: x**2)

        assert f.function(1) == 1
        assert f.function(2) == 4
        assert f.function(4) == 16
        assert f.num_parameters == 1
        assert f.parameter_list == ["x"]
        assert f.data is None

    def test_with_two_variables_and_domain(self):
        """Test from_callable with two variables and a domain."""
        domain = Domain().from_list([(1, 2), (2, 3)], data_name=["x", "y"])
        f = MultivariateFunction(domain=domain).from_callable(lambda x, y: x**2 + y)
        expected_data = pd.Series(
            [3, 7],
            index=domain.data,
            name="output",
        )

        assert f.num_parameters == 2
        assert f.parameter_list == ["x", "y"]
        pd.testing.assert_series_equal(f.data, expected_data)

    def test_with_one_variable_and_domain(self):
        """Test from_callable with one variable and a domain."""
        domain = Domain().from_list([1, 3], data_name=["x"])
        f = MultivariateFunction(domain=domain).from_callable(lambda x: x**2)
        expected_data = pd.Series(
            [1, 9],
            index=domain.data,
            name="output",
        )

        assert f.num_parameters == 1
        assert f.parameter_list == ["x"]
        pd.testing.assert_series_equal(f.data, expected_data)


class TestFromPandas:
    def test_with_two_variables(self):
        """Test from_pandas method with two variables."""
        data = pd.Series(
            [0, 1],
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["x", "y"]),
        )
        f = MultivariateFunction().from_pandas(data)

        assert f.data is data
        assert f.num_parameters == 2
        assert f.parameter_list == ["x", "y"]
        assert f.function is not None
        assert f.function(x=1, y=2) == 0
        assert f.function(x=2, y=3) == 1
        assert f.function(1, 2) == 0
        assert f.function(2, 3) == 1

    def test_with_one_variable(self):
        """Test from_pandas method with one variable."""
        data = pd.Series(
            [0, 1],
            index=pd.Index([1, 2], name="x"),
        )
        f = MultivariateFunction().from_pandas(data)

        assert f.data is data
        assert f.num_parameters == 1
        assert f.parameter_list == ["x"]
        assert f.function is not None
        assert f.function(x=1) == 0
        assert f.function(x=2) == 1
        assert f.function(1) == 0
        assert f.function(2) == 1


# --------------------- test properties --------------------- #


class TestDict:
    def test_dict_from_callable(self):
        """Test dict property on function constructed with from_callable."""
        D = Domain().from_list([(1, 2), (2, 3)], data_name=["x", "y"])
        f = MultivariateFunction(domain=D).from_callable(lambda x, y: x**2 + y)
        expected_dict = {
            (1, 2): 3,
            (2, 3): 7,
        }
        assert f.dict == expected_dict

    def test_dict_from_pandas(self):
        """Test dict property on function constructed with from_pandas."""
        data = pd.Series(
            [0, 1],
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["x", "y"]),
        )
        f = MultivariateFunction().from_pandas(data)
        expected_dict = {
            (1, 2): 0,
            (2, 3): 1,
        }
        assert f.dict == expected_dict


# --------------------- test data access --------------------- #


class TestCall:
    @pytest.fixture
    def f(self):
        return MultivariateFunction().from_callable(lambda x, y: x**2 + y)

    @pytest.fixture
    def g(self):
        return MultivariateFunction(name="g").from_callable(lambda x: x**2)

    def test_on_bivariate_function(self, f):
        """Test call method on bivariate function constructed with from_callable."""
        assert f(1, 4) == 5
        assert f(x=3, y=5) == 14

    def test_on_univariate_function(self, g):
        """Test call method on univariate function constructed with from_callable."""
        assert g(2) == 4
        assert g(x=5) == 25

    def test_currying_on_bivariate_function(self, f):
        """Testing currying on bivariate function."""
        assert f(x=1).name == "f(x=1)"
        assert f(x=1)(y=4) == 5
        assert f(x=1)(4) == 5
