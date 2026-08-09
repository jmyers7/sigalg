import inspect

import pandas as pd
import pytest
from sigalg.core import Domain, MultivariateFunction, SampleSpace

# --------------------- test constructors --------------------- #


class TestConstructor:
    def test_constructor_no_parameters(self):
        """Test base constructor with no parameters."""
        f = MultivariateFunction()

        assert f.name == "f"
        assert f.data is None
        assert f.function is None
        assert f.variable_names is None
        assert f.output_name == "output"
        assert f.domain is None

    def test_from_fun_with_two_variables(self):
        """Test from function with two variables."""

        def mapping(*, x, y):
            return x**2 + y

        f = MultivariateFunction(mapping=mapping)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in ["x", "y"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function is mapping
        assert f.num_variables == 2
        assert f.variable_names == ["x", "y"]
        assert f.output_name == "output"
        assert f.signature == expected_signature

    def test_from_fun_with_one_variable(self):
        """Test from function with one variable."""

        def mapping(*, x):
            return x**2

        f = MultivariateFunction(mapping=mapping)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in ["x"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function is mapping
        assert f.num_variables == 1
        assert f.variable_names == ["x"]
        assert f.output_name == "output"
        assert f.signature == expected_signature

    def test_from_fun_with_bivariate_lambda(self):
        """Test from function with a bivariate lambda function."""
        f = MultivariateFunction(mapping=lambda *, x, y: x + y)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in ["x", "y"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function(x=1, y=1) == 2
        assert f.function(x=0, y=1) == 1
        assert f.function(x=1, y=0) == 1
        assert f.num_variables == 2
        assert f.variable_names == ["x", "y"]
        assert f.output_name == "output"
        assert f.signature == expected_signature

    def test_from_fun_with_univariate_lambda(self):
        """Test from function with a univariate lambda function."""
        f = MultivariateFunction(mapping=lambda *, x: x**2)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in ["x"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function(x=1) == 1
        assert f.function(x=2) == 4
        assert f.function(x=4) == 16
        assert f.num_variables == 1
        assert f.variable_names == ["x"]
        assert f.output_name == "output"
        assert f.signature == expected_signature

    def test_from_series_with_two_variables(self):
        """Test from series with two variables."""
        mapping = pd.Series(
            [0, 1],
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["x", "y"]),
        )
        f = MultivariateFunction(mapping=mapping)
        expected_data = mapping.copy()
        expected_data.name = "output"

        pd.testing.assert_series_equal(f.data, expected_data)
        assert f.num_variables == 2
        assert f.variable_names == ["x", "y"]
        assert f.output_name == "output"
        assert f.function is not None
        assert f.function(x=1, y=2) == 0
        assert f.function(x=2, y=3) == 1

    def test_from_series_with_one_variable(self):
        """Test from series with one variable."""
        mapping = pd.Series(
            [0, 1],
            index=pd.Index([1, 2], name="x"),
        )
        g = MultivariateFunction(mapping=mapping, name="g")
        expected_data = mapping.copy()
        expected_data.name = "output"

        pd.testing.assert_series_equal(g.data, expected_data)
        assert g.num_variables == 1
        assert g.variable_names == ["x"]
        assert g.output_name == "output"
        assert g.function is not None
        assert g.function(x=1) == 0
        assert g.function(x=2) == 1


# --------------------- test properties --------------------- #


class TestData:
    def test_from_dict_with_no_domain_and_no_output_name(self):
        mapping = {(1, 2): 2, (3, 4): 4, (5, 6): 6}
        f = MultivariateFunction(mapping=mapping)
        expected_domain = Domain([(1, 2), (3, 4), (5, 6)])
        expected_data = pd.Series([2, 4, 6], index=expected_domain.data, name="output")

        pd.testing.assert_series_equal(f.data, expected_data)

    def test_from_dict_with_domain_and_output_name(self):
        D = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["x", "y"])
        mapping = {(1, 2): 2, (3, 4): 4, (5, 6): 6}
        f = MultivariateFunction(domain=D, mapping=mapping, output_name="num")
        expected_data = pd.Series([2, 4, 6], index=D.data, name="num")

        pd.testing.assert_series_equal(f.data, expected_data)

    def test_from_series_with_no_domain_and_no_output_name(self):
        mapping = pd.Series([2, 4, 6])
        f = MultivariateFunction(mapping=mapping)
        expected_domain = Domain([0, 1, 2])
        expected_data = pd.Series([2, 4, 6], index=expected_domain.data, name="output")

        pd.testing.assert_series_equal(f.data, expected_data)

    def test_from_series_with_domain_and_output_name(self):
        D = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["x", "y"])
        mapping = pd.Series([2, 4, 6], index=D.data)
        f = MultivariateFunction(domain=D, mapping=mapping, output_name="num")
        expected_data = pd.Series([2, 4, 6], index=D.data, name="num")

        pd.testing.assert_series_equal(f.data, expected_data)

    def test_misaligned_output_name_raises(self):
        D = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["x", "y"])
        mapping = pd.Series([2, 4, 6], index=D.data, name="output")

        with pytest.raises(
            ValueError, match="The name of the pd.Series must match the output name"
        ):
            MultivariateFunction(domain=D, mapping=mapping, output_name="num")

    def test_from_callable(self):
        D = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["x", "y"])
        f = MultivariateFunction(domain=D, mapping=lambda *, x, y: y, output_name="num")
        expected_data = pd.Series([2, 4, 6], index=D.data, name="num")

        pd.testing.assert_series_equal(f.data, expected_data)


class TestDict:
    def test_dict_from_callable(self):
        """Test dict property on function constructed with from a callable."""
        D = Domain([(1, 2), (2, 3)], variable_names=["x", "y"])
        f = MultivariateFunction(domain=D, mapping=lambda *, x, y: x**2 + y)
        expected_dict = {
            (1, 2): 3,
            (2, 3): 7,
        }
        assert f.dict == expected_dict

    def test_dict_from_pandas(self):
        """Test dict property on function constructed with from a series."""
        mapping = pd.Series(
            [0, 1],
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["x", "y"]),
        )
        f = MultivariateFunction(mapping=mapping)
        expected_dict = {
            (1, 2): 0,
            (2, 3): 1,
        }
        assert f.dict == expected_dict


# --------------------- test data access --------------------- #


class TestCall:
    def test_on_bivariate_function_with_explicit_arguments(self):
        """Test call method on bivariate function constructed from callable and explicit arguments."""
        f = MultivariateFunction(mapping=lambda *, x, y: x + y)

        assert f(x=1, y=2) == 3

    def test_on_univariate_function_with_explicit_arguments(self):
        """Test call method on univariate function constructed with from callable and explicit arguments."""
        f = MultivariateFunction(mapping=lambda *, x: x**2)

        assert f(x=2) == 4

    def test_on_bivariate_function_with_pandas_mapping(self):
        """Test call method on bivariate function constructed from a series."""
        mapping = pd.Series(
            [0, 1],
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["x", "y"]),
        )
        f = MultivariateFunction(mapping=mapping)

        assert f(x=1, y=2) == 0
        assert f(x=2, y=3) == 1

    def test_on_univariate_function_with_pandas_mapping(self):
        """Test call method on univariate function constructed from a series."""
        mapping = pd.Series(
            [0, 1],
            index=pd.Index([1, 2], name="x"),
        )
        f = MultivariateFunction(mapping=mapping)

        assert f(x=1) == 0
        assert f(x=2) == 1


# --------------------- test equality --------------------- #


class TestEquality:
    @pytest.fixture
    def Omega(self):
        return SampleSpace.from_sequence(size=2)

    @pytest.fixture
    def D(self):
        return Domain([(0, 1), (1, 2)], variable_names=["x", "y"])

    @pytest.fixture
    def D_reordered(self):
        """Same domain but with reordered arguments."""
        return Domain([(1, 0), (2, 1)], variable_names=["y", "x"])

    def test_equal_functions_same_order(self, D):
        """Test that functions with same domain and function are equal."""
        f = MultivariateFunction(domain=D, mapping=lambda *, x, y: x**2 + y**2)
        g = MultivariateFunction(
            domain=D, name="g", mapping=lambda *, x, y: x**2 + y**2
        )

        assert f == g

    def test_equal_functions_different_order(self, D, D_reordered):
        """Test that functions with reordered arguments but same values are equal."""
        f = MultivariateFunction(domain=D, mapping=lambda *, x, y: x**2 + y**2)
        g = MultivariateFunction(
            domain=D_reordered, name="g", mapping=lambda *, y, x: x**2 + y**2
        )

        assert f == g

    def test_equal_univariate_functions(self):
        """Test equality for univariate functions."""
        D = Domain([1, 2], variable_names=["x"])
        f = MultivariateFunction(domain=D, mapping=lambda *, x: x**2)
        g = MultivariateFunction(domain=D, name="g", mapping=lambda *, x: x**2)

        assert f == g

    def test_equal_functions_from_series(self):
        """Test equality for functions constructed from series."""
        data1 = pd.Series(
            [1, 5],
            index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)], names=["x", "y"]),
            name="output",
        )
        data2 = pd.Series(
            [1, 5],
            index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)], names=["x", "y"]),
            name="output",
        )
        f = MultivariateFunction(mapping=data1)
        g = MultivariateFunction(name="g", mapping=data2)

        assert f == g

    def test_different_values(self, D):
        """Test that functions with same domain but different values are not equal."""
        f = MultivariateFunction(domain=D, mapping=lambda *, x, y: x**2 + y**2)
        g = MultivariateFunction(domain=D, name="g", mapping=lambda *, x, y: x + y)

        assert f != g

    def test_different_domains_same_function(self):
        """Test that functions with different domains are not equal."""
        D1 = Domain([(0, 1), (1, 2)], variable_names=["x", "y"])
        D2 = Domain([(0, 1), (2, 3)], variable_names=["x", "y"])

        f = MultivariateFunction(domain=D1, mapping=lambda *, x, y: x**2 + y**2)
        g = MultivariateFunction(
            domain=D2, name="g", mapping=lambda *, x, y: x**2 + y**2
        )

        assert f != g

    def test_missing_domain_self_raises(self):
        """Test that comparing when self has no domain raises ValueError."""
        f = MultivariateFunction(mapping=lambda *, x, y: x + y)
        D = Domain([(0, 1), (1, 2)], variable_names=["x", "y"])
        g = MultivariateFunction(domain=D, name="g", mapping=lambda *, x, y: x + y)

        with pytest.raises(ValueError, match="domains are not defined"):
            f == g  # noqa: B015

    def test_missing_domain_other_raises(self, D):
        """Test that comparing when other has no domain raises ValueError."""
        f = MultivariateFunction(domain=D, mapping=lambda *, x, y: x + y)
        g = MultivariateFunction(name="g", mapping=lambda *, x, y: x + y)

        with pytest.raises(ValueError, match="domains are not defined"):
            f == g  # noqa: B015


# --------------------- test arithmetic methods --------------------- #


class TestArithmeticWithScalars:
    @pytest.fixture
    def D(self):
        return Domain([(0, 1), (1, 2), (2, 3)], variable_names=["x", "y"])

    @pytest.fixture
    def f(self, D):
        return MultivariateFunction(domain=D, mapping=lambda *, x, y: x**2 + 2 * y)

    def test_add_scalar_right(self, f, D):
        """Test f + scalar."""
        result = f + 1
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: (x**2 + 2 * y) + 1
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f + 1)"

    def test_add_scalar_left(self, f, D):
        """Test scalar + f."""
        result = 1 + f
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: 1 + (x**2 + 2 * y)
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(1 + f)"

    def test_subtract_scalar_right(self, f, D):
        """Test f - scalar."""
        result = f - 1
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: (x**2 + 2 * y) - 1
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f - 1)"

    def test_subtract_scalar_left(self, f, D):
        """Test scalar - f."""
        result = 1 - f
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: 1 - (x**2 + 2 * y)
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(1 - f)"

    def test_multiply_scalar_right(self, f, D):
        """Test f * scalar."""
        result = f * 2
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: (x**2 + 2 * y) * 2
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f * 2)"

    def test_multiply_scalar_left(self, f, D):
        """Test scalar * f."""
        result = 2 * f
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: 2 * (x**2 + 2 * y)
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(2 * f)"

    def test_divide_scalar_right(self, f, D):
        """Test f / scalar."""
        result = f / 2
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: (x**2 + 2 * y) / 2
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f / 2)"

    def test_divide_scalar_left(self, f, D):
        """Test scalar / f."""
        result = 2 / f
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: 2 / (x**2 + 2 * y)
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(2 / f)"

    def test_power_scalar_right(self, f, D):
        """Test f ** scalar."""
        result = f**2
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: (x**2 + 2 * y) ** 2
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f ** 2)"

    def test_power_scalar_left(self, f, D):
        """Test scalar ** f."""
        result = 2**f
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: 2 ** (x**2 + 2 * y)
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(2 ** f)"


class TestArithmeticFullyAlignedDomains:
    @pytest.fixture
    def D(self):
        return Domain([(0, 1), (1, 2), (2, 3)], variable_names=["x", "y"])

    @pytest.fixture
    def f(self, D):
        return MultivariateFunction(domain=D, mapping=lambda *, x, y: x**2 + 2 * y)

    @pytest.fixture
    def g(self, D):
        return MultivariateFunction(
            domain=D, name="g", mapping=lambda *, x, y: 2 * x - y
        )

    def test_add(self, f, g, D):
        """Test f + g for fully aligned domains."""
        result = f + g
        expected_result = MultivariateFunction(
            domain=D,
            mapping=lambda *, x, y: (x**2 + 2 * y) + (2 * x - y),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f + g)"

    def test_subtract(self, f, g, D):
        """Test f - g for fully aligned domains."""
        result = f - g
        expected_result = MultivariateFunction(
            domain=D,
            mapping=lambda *, x, y: (x**2 + 2 * y) - (2 * x - y),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f - g)"

    def test_multiply(self, f, g, D):
        """Test f * g for fully aligned domains."""
        result = f * g
        expected_result = MultivariateFunction(
            domain=D,
            mapping=lambda *, x, y: (x**2 + 2 * y) * (2 * x - y),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f * g)"

    def test_divide(self, f, g, D):
        """Test g / f for fully aligned domains."""
        result = g / f
        expected_result = MultivariateFunction(
            domain=D,
            mapping=lambda *, x, y: (2 * x - y) / (x**2 + 2 * y),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(g / f)"

    def test_power(self, f, g, D):
        """Test f ** g for fully aligned domains."""
        result = f**g
        expected_result = MultivariateFunction(
            domain=D,
            mapping=lambda *, x, y: (x**2 + 2 * y) ** (2 * x - y),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f ** g)"


class TestArithmeticPartiallyAlignedDomains:
    @pytest.fixture
    def D_f(self):
        return Domain(
            [(2, -1), (0, 1), (1, 2), (2, 3)], variable_names=["x", "y"], name="D_f"
        )

    @pytest.fixture
    def D_g(self):
        return Domain(
            [(1, 0), (2, 2), (2, 4), (3, -1), (4, 5)],
            variable_names=["y", "z"],
            name="D_g",
        )

    @pytest.fixture
    def f(self, D_f):
        return MultivariateFunction(
            domain=D_f, mapping=lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    @pytest.fixture
    def g(self, D_g):
        return MultivariateFunction(
            domain=D_g,
            name="g",
            mapping=lambda *, y, z: 2 * y - z,
            output_name="output",
        )

    def test_add(self, f, g):
        """Test f + g for partially aligned domains."""
        result = f + g
        expected_domain = Domain(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)],
            variable_names=["x", "y", "z"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z: (x**2 + 2 * y) + (2 * y - z),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f + g)"
        assert result.variable_names == ["x", "y", "z"]

    def test_subtract(self, f, g):
        """Test f - g for partially aligned domains."""
        result = f - g
        expected_domain = Domain(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)],
            variable_names=["x", "y", "z"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z: (x**2 + 2 * y) - (2 * y - z),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f - g)"
        assert result.variable_names == ["x", "y", "z"]

    def test_multiply(self, f, g):
        """Test f * g for partially aligned domains."""
        result = f * g
        expected_domain = Domain(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)],
            variable_names=["x", "y", "z"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z: (x**2 + 2 * y) * (2 * y - z),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f * g)"
        assert result.variable_names == ["x", "y", "z"]

    def test_divide(self, f, g):
        """Test g / f for partially aligned domains."""
        result = g / f
        expected_domain = Domain(
            [(1, 0, 0), (2, 2, 1), (2, 4, 1), (3, -1, 2)],
            variable_names=["y", "z", "x"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, y, z, x: (2 * y - z) / (x**2 + 2 * y),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(g / f)"
        assert result.variable_names == ["y", "z", "x"]

    def test_power(self, f, g):
        """Test f ** g for partially aligned domains."""
        result = f**g
        expected_domain = Domain(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)],
            variable_names=["x", "y", "z"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z: (x**2 + 2 * y) ** (2 * y - z),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f ** g)"
        assert result.variable_names == ["x", "y", "z"]


class TestArithmeticNonAlignedDomains:
    @pytest.fixture
    def D_f(self):
        return Domain([(0, 1), (1, 2), (2, 3)], variable_names=["x", "y"], name="D_f")

    @pytest.fixture
    def D_g(self):
        return Domain([(1, 0), (2, 2), (2, 4)], variable_names=["z", "w"], name="D_g")

    @pytest.fixture
    def f(self, D_f):
        return MultivariateFunction(
            domain=D_f, mapping=lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    @pytest.fixture
    def g(self, D_g):
        return MultivariateFunction(
            domain=D_g,
            name="g",
            mapping=lambda *, z, w: 2 * z - w,
            output_name="output",
        )

    def test_add(self, f, g):
        """Test f + g for non-aligned domains (cross product)."""
        result = f + g
        expected_domain = Domain(
            [
                (0, 1, 1, 0),
                (0, 1, 2, 2),
                (0, 1, 2, 4),
                (1, 2, 1, 0),
                (1, 2, 2, 2),
                (1, 2, 2, 4),
                (2, 3, 1, 0),
                (2, 3, 2, 2),
                (2, 3, 2, 4),
            ],
            variable_names=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z, w: (x**2 + 2 * y) + (2 * z - w),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f + g)"
        assert result.variable_names == ["x", "y", "z", "w"]
        assert len(result.domain.data) == 9

    def test_subtract(self, f, g):
        """Test f - g for non-aligned domains."""
        result = f - g
        expected_domain = Domain(
            [
                (0, 1, 1, 0),
                (0, 1, 2, 2),
                (0, 1, 2, 4),
                (1, 2, 1, 0),
                (1, 2, 2, 2),
                (1, 2, 2, 4),
                (2, 3, 1, 0),
                (2, 3, 2, 2),
                (2, 3, 2, 4),
            ],
            variable_names=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z, w: (x**2 + 2 * y) - (2 * z - w),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f - g)"
        assert result.variable_names == ["x", "y", "z", "w"]

    def test_multiply(self, f, g):
        """Test f * g for non-aligned domains."""
        result = f * g
        expected_domain = Domain(
            [
                (0, 1, 1, 0),
                (0, 1, 2, 2),
                (0, 1, 2, 4),
                (1, 2, 1, 0),
                (1, 2, 2, 2),
                (1, 2, 2, 4),
                (2, 3, 1, 0),
                (2, 3, 2, 2),
                (2, 3, 2, 4),
            ],
            variable_names=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z, w: (x**2 + 2 * y) * (2 * z - w),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f * g)"
        assert result.variable_names == ["x", "y", "z", "w"]

    def test_divide(self, f, g):
        """Test g / f for non-aligned domains."""
        result = g / f
        expected_domain = Domain(
            [
                (1, 0, 0, 1),
                (1, 0, 1, 2),
                (1, 0, 2, 3),
                (2, 2, 0, 1),
                (2, 2, 1, 2),
                (2, 2, 2, 3),
                (2, 4, 0, 1),
                (2, 4, 1, 2),
                (2, 4, 2, 3),
            ],
            variable_names=["z", "w", "x", "y"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, z, w, x, y: (2 * z - w) / (x**2 + 2 * y),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(g / f)"
        assert result.variable_names == ["z", "w", "x", "y"]

    def test_power(self, f, g):
        """Test f ** g for non-aligned domains."""
        result = f**g
        expected_domain = Domain(
            [
                (0, 1, 1, 0),
                (0, 1, 2, 2),
                (0, 1, 2, 4),
                (1, 2, 1, 0),
                (1, 2, 2, 2),
                (1, 2, 2, 4),
                (2, 3, 1, 0),
                (2, 3, 2, 2),
                (2, 3, 2, 4),
            ],
            variable_names=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain,
            mapping=lambda *, x, y, z, w: (x**2 + 2 * y) ** (2 * z - w),
            output_name="output",
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(f ** g)"
        assert result.variable_names == ["x", "y", "z", "w"]


class TestNegation:
    @pytest.fixture
    def D(self):
        return Domain([(0, 1), (1, 2), (2, 3)], variable_names=["x", "y"])

    @pytest.fixture
    def f(self, D):
        return MultivariateFunction(
            domain=D, mapping=lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    def test_negation(self, f, D):
        """Test -f."""
        result = -f
        expected_result = MultivariateFunction(
            domain=D, mapping=lambda *, x, y: -(x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.output_name == "output"
        assert result.name == "(-f)"
        assert result.domain.data.equals(D.data)


class TestArithmeticValidation:
    """Test error handling for invalid arithmetic operations."""

    @pytest.fixture
    def D(self):
        return Domain([(0, 1), (1, 2)], variable_names=["x", "y"])

    @pytest.fixture
    def f(self, D):
        return MultivariateFunction(domain=D, mapping=lambda *, x, y: x + y)

    @pytest.fixture
    def g(self, D):
        return MultivariateFunction(domain=D, name="g", mapping=lambda *, x, y: x - y)

    @pytest.fixture
    def h(self, D):
        return MultivariateFunction(domain=D, name="h", mapping=lambda *, x, y: x * y)

    def test_invalid_operand_type_raises(self, f):
        """Test that operations with invalid types raise TypeError."""
        with pytest.raises(
            TypeError,
            match=r"Unsupported operand type\(s\) for \+: 'MultivariateFunction' and 'str'",
        ):
            f + "invalid"


class TestAlgebraicProperties:
    @pytest.fixture
    def D(self):
        return Domain([(0, 1), (1, 2), (2, 3)], variable_names=["x", "y"])

    @pytest.fixture
    def f(self, D):
        return MultivariateFunction(
            domain=D, mapping=lambda *, x, y: x**2 + y, output_name="output"
        )

    @pytest.fixture
    def g(self, D):
        return MultivariateFunction(
            domain=D,
            name="g",
            mapping=lambda *, x, y: 2 * x + y**2,
            output_name="output",
        )

    @pytest.fixture
    def h(self, D):
        return MultivariateFunction(
            domain=D, name="h", mapping=lambda *, x, y: x + 2 * y, output_name="output"
        )

    def test_addition_commutative(self, f, g):
        """Test that addition is commutative: f + g = g + f."""
        assert f + g == g + f

    def test_multiplication_commutative(self, f, g):
        """Test that multiplication is commutative: f * g = g * f."""
        assert f * g == g * f

    def test_addition_associative(self, f, g, h):
        """Test that addition is associative: (f + g) + h = f + (g + h)."""
        assert (f + g) + h == f + (g + h)

    def test_multiplication_associative(self, f, g, h):
        """Test that multiplication is associative: (f * g) * h = f * (g * h)."""
        assert (f * g) * h == f * (g * h)

    def test_additive_identity(self, f):
        """Test additive identity: f + 0 = f and 0 + f = f."""
        assert f + 0 == f
        assert 0 + f == f

    def test_multiplicative_identity(self, f):
        """Test multiplicative identity: f * 1 = f and 1 * f = f."""
        assert f * 1 == f
        assert 1 * f == f

    def test_absorption_multiplication_by_zero(self, f):
        """Test absorption: f * 0 = 0 and 0 * f = 0."""
        assert f * 0 == 0
        assert 0 * f == 0

    def test_additive_inverse(self, f):
        """Test additive inverse: f + (-f) = 0 and (-f) + f = 0."""
        assert f + (-f) == 0
        assert (-f) + f == 0

    def test_left_distributive(self, f, g, h):
        """Test left distributivity: f * (g + h) = f * g + f * h."""
        assert f * (g + h) == f * g + f * h

    def test_right_distributive(self, f, g, h):
        """Test right distributivity: (f + g) * h = f * h + g * h."""
        assert (f + g) * h == f * h + g * h

    def test_scalar_multiplication_commutative(self, f):
        """Test that scalar multiplication is commutative: c * f = f * c."""
        assert 2 * f == f * 2

    def test_scalar_multiplication_associative(self, f):
        """Test that scalar multiplication is associative: c * (d * f) = (c * d) * f."""
        assert 2 * (3 * f) == (2 * 3) * f

    def test_scalar_distributive_over_addition(self, f, g):
        """Test scalar distributivity over addition: c * (f + g) = c * f + c * g."""
        assert 2 * (f + g) == 2 * f + 2 * g

    def test_double_negation(self, f):
        """Test double negation: -(-f) = f."""
        assert f == -(-f)  # noqa: B002

    def test_negation_distributive_over_addition(self, f, g):
        """Test negation distributes over addition: -(f + g) = -f + -g."""
        assert -(f + g) == (-f) + (-g)

    def test_negation_distributive_over_multiplication_left(self, f, g):
        """Test negation distributes over multiplication (left): -(f * g) = (-f) * g."""
        assert -(f * g) == (-f) * g

    def test_negation_distributive_over_multiplication_right(self, f, g):
        """Test negation distributes over multiplication (right): -(f * g) = f * (-g)."""
        assert -(f * g) == f * (-g)

    def test_power_identity_exponent(self, f):
        """Test identity exponent: f ** 1 = f."""
        assert f**1 == f

    def test_power_zero_exponent(self, f):
        """Test zero exponent: f ** 0 = 1 (constant function with value 1)."""
        assert f**0 == 1
