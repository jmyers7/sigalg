import inspect

import pandas as pd
import pytest

from sigalg.core import Domain, MultivariateFunction, SampleSpace, SigmaAlgebra

# --------------------- test constructors --------------------- #


class TestBaseConstructor:
    def test_constructor_no_parameters(self):
        """Test base constructor with no parameters."""
        f = MultivariateFunction()

        assert f.name == "f"
        assert f.data is None
        assert f.function is None
        assert f.argument_names is None
        assert f.domain is None

    def test_constructor_with_all_parameters(self):
        """Test base constructor with all parameters."""
        domain = Domain().from_list([(0, 1), (1, 2)], data_name=["x", "y"])
        g = MultivariateFunction(domain=domain, name="g")

        assert g.name == "g"
        assert g.data is None
        assert g.function is None
        assert g.argument_names is None
        assert g.domain is domain


class TestFromCallable:
    def test_with_two_variables(self):
        """Test from_callable with two variables."""

        def function(*, x, y):
            return x**2 + y

        f = MultivariateFunction().from_callable(function)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in ["x", "y"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function is function
        assert f.num_arguments == 2
        assert f.argument_names == ["x", "y"]
        assert f.signature == expected_signature

    def test_with_one_variable(self):
        """Test from_callable with one variable."""

        def function(*, x):
            return x**2

        f = MultivariateFunction().from_callable(function)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in ["x"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function is function
        assert f.num_arguments == 1
        assert f.argument_names == ["x"]
        assert f.signature == expected_signature

    def test_with_bivariate_lambda(self):
        """Test from_callable with a bivariate lambda function."""
        f = MultivariateFunction().from_callable(lambda *, x, y: x + y)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in ["x", "y"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function(x=1, y=1) == 2
        assert f.function(x=0, y=1) == 1
        assert f.function(x=1, y=0) == 1
        assert f.num_arguments == 2
        assert f.argument_names == ["x", "y"]
        assert f.signature == expected_signature

    def test_with_univariate_lambda(self):
        """Test from_callable with a univariate lambda function."""
        f = MultivariateFunction().from_callable(lambda *, x: x**2)
        expected_parameters = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in ["x"]
        ]
        expected_signature = inspect.Signature(expected_parameters)

        assert f.function(x=1) == 1
        assert f.function(x=2) == 4
        assert f.function(x=4) == 16
        assert f.num_arguments == 1
        assert f.argument_names == ["x"]
        assert f.signature == expected_signature


class TestFromPandas:
    def test_with_two_variables(self):
        """Test from_pandas method with two variables."""
        data = pd.Series(
            [0, 1],
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["x", "y"]),
        )
        f = MultivariateFunction().from_pandas(data)

        assert f.data is data
        assert f.num_arguments == 2
        assert f.argument_names == ["x", "y"]
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
        assert f.num_arguments == 1
        assert f.argument_names == ["x"]
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
        f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: x**2 + y)
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
    def test_on_bivariate_function_with_explicit_arguments(self):
        """Test call method on bivariate function constructed with from_callable and explicit arguments."""
        f = MultivariateFunction().from_callable(lambda *, x, y: x + y)

        assert f(x=1, y=2) == 3

    def test_on_univariate_function_with_explicit_arguments(self):
        """Test call method on univariate function constructed with from_callable and explicit arguments."""
        f = MultivariateFunction().from_callable(lambda *, x: x**2)

        assert f(x=2) == 4

    def test_on_bivariate_function_with_pandas_data(self):
        """Test call method on bivariate function constructed with from_pandas."""
        data = pd.Series(
            [0, 1],
            index=pd.MultiIndex.from_tuples([(1, 2), (2, 3)], names=["x", "y"]),
        )
        f = MultivariateFunction().from_pandas(data)

        assert f(x=1, y=2) == 0
        assert f(x=2, y=3) == 1

    def test_on_univariate_function_with_pandas_data(self):
        """Test call method on univariate function constructed with from_pandas."""
        data = pd.Series(
            [0, 1],
            index=pd.Index([1, 2], name="x"),
        )
        f = MultivariateFunction().from_pandas(data)

        assert f(x=1) == 0
        assert f(x=2) == 1


# --------------------- test arithmetic methods --------------------- #


class TestArithmeticWithScalars:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def D(self):
        return Domain().from_list([(0, 1), (1, 2), (2, 3)], data_name=["x", "y"])

    @pytest.fixture
    def f(self, D, F):
        return MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    def test_add_scalar_right(self, f, D, F):
        """Test f + scalar."""
        result = f + 1
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) + 1, output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f + 1)"

    def test_add_scalar_left(self, f, D, F):
        """Test scalar + f."""
        result = 1 + f
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: 1 + (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(1 + f)"

    def test_subtract_scalar_right(self, f, D, F):
        """Test f - scalar."""
        result = f - 1
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) - 1, output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f - 1)"

    def test_subtract_scalar_left(self, f, D, F):
        """Test scalar - f."""
        result = 1 - f
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: 1 - (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(1 - f)"

    def test_multiply_scalar_right(self, f, D, F):
        """Test f * scalar."""
        result = f * 2
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) * 2, output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f * 2)"

    def test_multiply_scalar_left(self, f, D, F):
        """Test scalar * f."""
        result = 2 * f
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: 2 * (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(2 * f)"

    def test_divide_scalar_right(self, f, D, F):
        """Test f / scalar."""
        result = f / 2
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) / 2, output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f / 2)"

    def test_divide_scalar_left(self, f, D, F):
        """Test scalar / f."""
        result = 2 / f
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: 2 / (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(2 / f)"

    def test_power_scalar_right(self, f, D, F):
        """Test f ** scalar."""
        result = f**2
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) ** 2, output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f ** 2)"

    def test_power_scalar_left(self, f, D, F):
        """Test scalar ** f."""
        result = 2**f
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: 2 ** (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(2 ** f)"


class TestArithmeticFullyAlignedDomains:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def D(self):
        return Domain().from_list([(0, 1), (1, 2), (2, 3)], data_name=["x", "y"])

    @pytest.fixture
    def f(self, D, F):
        return MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    @pytest.fixture
    def g(self, D, F):
        return MultivariateFunction(domain=D, sig_alg=F, name="g").from_callable(
            lambda *, x, y: 2 * x - y, output_name="output"
        )

    def test_add(self, f, g, D, F):
        """Test f + g for fully aligned domains."""
        result = f + g
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) + (2 * x - y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f + g)"

    def test_subtract(self, f, g, D, F):
        """Test f - g for fully aligned domains."""
        result = f - g
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) - (2 * x - y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f - g)"

    def test_multiply(self, f, g, D, F):
        """Test f * g for fully aligned domains."""
        result = f * g
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) * (2 * x - y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f * g)"

    def test_divide(self, f, g, D, F):
        """Test g / f for fully aligned domains."""
        result = g / f
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (2 * x - y) / (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(g / f)"

    def test_power(self, f, g, D, F):
        """Test f ** g for fully aligned domains."""
        result = f**g
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: (x**2 + 2 * y) ** (2 * x - y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f ** g)"


class TestArithmeticPartiallyAlignedDomains:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def D_f(self):
        return Domain(name="D_f").from_list(
            [(2, -1), (0, 1), (1, 2), (2, 3)], data_name=["x", "y"]
        )

    @pytest.fixture
    def D_g(self):
        return Domain(name="D_g").from_list(
            [(1, 0), (2, 2), (2, 4), (3, -1), (4, 5)], data_name=["y", "z"]
        )

    @pytest.fixture
    def f(self, D_f, F):
        return MultivariateFunction(domain=D_f, sig_alg=F).from_callable(
            lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    @pytest.fixture
    def g(self, D_g, F):
        return MultivariateFunction(domain=D_g, sig_alg=F, name="g").from_callable(
            lambda *, y, z: 2 * y - z, output_name="output"
        )

    def test_add(self, f, g, F):
        """Test f + g for partially aligned domains."""
        result = f + g
        expected_domain = Domain().from_list(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)], data_name=["x", "y", "z"]
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z: (x**2 + 2 * y) + (2 * y - z), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f + g)"
        assert result.argument_names == ["x", "y", "z"]

    def test_subtract(self, f, g, F):
        """Test f - g for partially aligned domains."""
        result = f - g
        expected_domain = Domain().from_list(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)], data_name=["x", "y", "z"]
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z: (x**2 + 2 * y) - (2 * y - z), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f - g)"
        assert result.argument_names == ["x", "y", "z"]

    def test_multiply(self, f, g, F):
        """Test f * g for partially aligned domains."""
        result = f * g
        expected_domain = Domain().from_list(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)], data_name=["x", "y", "z"]
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z: (x**2 + 2 * y) * (2 * y - z), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f * g)"
        assert result.argument_names == ["x", "y", "z"]

    def test_divide(self, f, g, F):
        """Test g / f for partially aligned domains."""
        result = g / f
        expected_domain = Domain().from_list(
            [(1, 0, 0), (2, 2, 1), (2, 4, 1), (3, -1, 2)], data_name=["y", "z", "x"]
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, y, z, x: (2 * y - z) / (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(g / f)"
        assert result.argument_names == ["y", "z", "x"]

    def test_power(self, f, g, F):
        """Test f ** g for partially aligned domains."""
        result = f**g
        expected_domain = Domain().from_list(
            [(0, 1, 0), (1, 2, 2), (1, 2, 4), (2, 3, -1)], data_name=["x", "y", "z"]
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z: (x**2 + 2 * y) ** (2 * y - z), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f ** g)"
        assert result.argument_names == ["x", "y", "z"]


class TestArithmeticNonAlignedDomains:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def D_f(self):
        return Domain(name="D_f").from_list(
            [(0, 1), (1, 2), (2, 3)], data_name=["x", "y"]
        )

    @pytest.fixture
    def D_g(self):
        return Domain(name="D_g").from_list(
            [(1, 0), (2, 2), (2, 4)], data_name=["z", "w"]
        )

    @pytest.fixture
    def f(self, D_f, F):
        return MultivariateFunction(domain=D_f, sig_alg=F).from_callable(
            lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    @pytest.fixture
    def g(self, D_g, F):
        return MultivariateFunction(domain=D_g, sig_alg=F, name="g").from_callable(
            lambda *, z, w: 2 * z - w, output_name="output"
        )

    def test_add(self, f, g, F):
        """Test f + g for non-aligned domains (cross product)."""
        result = f + g
        expected_domain = Domain().from_list(
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
            data_name=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z, w: (x**2 + 2 * y) + (2 * z - w), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f + g)"
        assert result.argument_names == ["x", "y", "z", "w"]
        assert len(result.domain.data) == 9

    def test_subtract(self, f, g, F):
        """Test f - g for non-aligned domains."""
        result = f - g
        expected_domain = Domain().from_list(
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
            data_name=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z, w: (x**2 + 2 * y) - (2 * z - w), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f - g)"
        assert result.argument_names == ["x", "y", "z", "w"]

    def test_multiply(self, f, g, F):
        """Test f * g for non-aligned domains."""
        result = f * g
        expected_domain = Domain().from_list(
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
            data_name=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z, w: (x**2 + 2 * y) * (2 * z - w), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f * g)"
        assert result.argument_names == ["x", "y", "z", "w"]

    def test_divide(self, f, g, F):
        """Test g / f for non-aligned domains."""
        result = g / f
        expected_domain = Domain().from_list(
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
            data_name=["z", "w", "x", "y"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, z, w, x, y: (2 * z - w) / (x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(g / f)"
        assert result.argument_names == ["z", "w", "x", "y"]

    def test_power(self, f, g, F):
        """Test f ** g for non-aligned domains."""
        result = f**g
        expected_domain = Domain().from_list(
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
            data_name=["x", "y", "z", "w"],
        )
        expected_result = MultivariateFunction(
            domain=expected_domain, sig_alg=F
        ).from_callable(
            lambda *, x, y, z, w: (x**2 + 2 * y) ** (2 * z - w), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(f ** g)"
        assert result.argument_names == ["x", "y", "z", "w"]


class TestNegation:
    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def D(self):
        return Domain().from_list([(0, 1), (1, 2), (2, 3)], data_name=["x", "y"])

    @pytest.fixture
    def f(self, D, F):
        return MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: x**2 + 2 * y, output_name="output"
        )

    def test_negation(self, f, D, F):
        """Test -f."""
        result = -f
        expected_result = MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: -(x**2 + 2 * y), output_name="output"
        )

        assert result == expected_result
        assert result.sig_alg is F
        assert result.output_name == "output"
        assert result.name == "(-f)"
        assert result.domain.data.equals(D.data)


class TestArithmeticValidation:
    """Test error handling for invalid arithmetic operations."""

    @pytest.fixture
    def Omega(self):
        return SampleSpace().from_sequence(size=2)

    @pytest.fixture
    def F(self, Omega):
        return SigmaAlgebra.power_set(Omega)

    @pytest.fixture
    def G(self, Omega):
        return SigmaAlgebra(sample_space=Omega).from_dict({0: 0, 1: 0})

    @pytest.fixture
    def D(self):
        return Domain().from_list([(0, 1), (1, 2)], data_name=["x", "y"])

    @pytest.fixture
    def f(self, D, F):
        return MultivariateFunction(domain=D, sig_alg=F).from_callable(
            lambda *, x, y: x + y
        )

    @pytest.fixture
    def g(self, D, G):
        return MultivariateFunction(domain=D, sig_alg=G, name="g").from_callable(
            lambda *, x, y: x - y
        )

    @pytest.fixture
    def h(self, D):
        return MultivariateFunction(domain=D, name="h").from_callable(
            lambda *, x, y: x * y
        )

    def test_mismatched_sigma_algebras_raises(self, f, g):
        """Test that operations between functions with different sigma-algebras raise ValueError."""
        with pytest.raises(
            ValueError,
            match="Cannot perform operations on functions with different sigma-algebras",
        ):
            f + g

    def test_mixed_sigma_algebra_presence_raises(self, f, h):
        """Test that operations between function with sig_alg and without raise ValueError."""
        with pytest.raises(
            ValueError,
            match="Cannot perform operations on functions when one has a sigma-algebra",
        ):
            f + h

    def test_invalid_operand_type_raises(self, f):
        """Test that operations with invalid types raise TypeError."""
        with pytest.raises(
            TypeError,
            match=r"Unsupported operand type\(s\) for \+: 'MultivariateFunction' and 'str'",
        ):
            f + "invalid"
