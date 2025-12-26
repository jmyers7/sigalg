import pandas as pd
import pytest

from sigalg.core import RandomVariable, SampleSpace


class TestConstructor:

    @pytest.mark.parametrize(
        "domain_indices, outputs, name",
        [
            pytest.param(
                ["omega0", "omega1", "omega2"],
                {"omega0": 1, "omega1": 2, "omega2": 3},
                "Y",
                id="basic_construction",
            ),
            pytest.param(
                [0, 1, 2, 3],
                {0: 1, 1: 3, 2: 5, 3: 7},
                "default_name_flag",
                id="default_name",
            ),
            pytest.param(
                [0, 1, 2],
                {0: 100, 1: 200, 2: 300},
                None,
                id="none_name",
            ),
            pytest.param(
                ["a", "b", "c"],
                {"a": (0.1, 0.2, 0.3), "b": (0.4, 0.5, 0.6), "c": (0.7, 0.8, 0.9)},
                42,
                id="non_string_name",
            ),
        ],
    )
    def test_constructor(self, domain_indices, outputs, name):
        """Test RandomVariable constructor with various outputs and domain indices."""
        domain = SampleSpace(indices=domain_indices, name="Omega")
        if name == "default_name_flag":
            rv = RandomVariable(outputs=outputs, domain=domain)
            name = "X"
        else:
            rv = RandomVariable(outputs=outputs, domain=domain, name=name)
        expected_data = pd.Series(data=outputs, index=domain.data, name=name)
        expected_data.index.name = domain.data.name

        pd.testing.assert_series_equal(rv.data, expected_data)
        assert rv.outputs == outputs
        assert rv.domain == domain
        assert rv.name == name


class TestFromPandas:

    @pytest.mark.parametrize(
        "data, index, name",
        [
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                pd.Index(["a", "b", "c"], name="letters"),
                "Z",
                id="custom_indices",
            ),
            pytest.param(
                [(1, 2), (3, 4), (5, 6)],
                None,
                "X",
                id="default_indices",
            ),
        ],
    )
    def test_from_pandas(self, data, index, name):
        """Test RandomVariable.from_pandas method."""
        data = pd.Series(data=data, index=index)
        rv = RandomVariable.from_pandas(data=data, name=name)
        expected_domain = SampleSpace(
            indices=list(data.index), name="Omega", data_name=data.index.name
        )

        pd.testing.assert_series_equal(rv.data, data)
        assert rv.domain == expected_domain
        assert rv.name == name


class TestRange:

    @pytest.mark.parametrize(
        "outputs, name, domain_indices, expected_range_outputs, expected_range_name, expected_range_domain_indices",
        [
            pytest.param(
                {"omega0": 1, "omega1": 2, "omega2": 2},
                "X",
                ["omega0", "omega1", "omega2"],
                {"x0": 2, "x1": 1},
                "range(X)",
                ["x0", "x1"],
                id="variable_with_str_name",
            ),
            pytest.param(
                {"omega0": 1, "omega1": 2, "omega2": 2},
                42,
                ["omega0", "omega1", "omega2"],
                {0: 2, 1: 1},
                None,
                [0, 1],
                id="variable_with_int_name",
            ),
            pytest.param(
                {"omega0": 1, "omega1": 2, "omega2": 2},
                None,
                ["omega0", "omega1", "omega2"],
                {0: 2, 1: 1},
                None,
                [0, 1],
                id="variable_with_none_name",
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
    ):
        """Test range property of RandomVariable."""
        domain = SampleSpace(indices=domain_indices, name="Omega")
        rv = RandomVariable(outputs=outputs, domain=domain, name=name)
        expected_range_domain = SampleSpace(
            indices=expected_range_domain_indices,
            name=expected_range_name,
            data_name="output",
        )
        expected_range_data = pd.Series(
            data=expected_range_outputs,
            index=pd.Index(expected_range_domain_indices, name="output"),
            name=name,
        )
        expected_range_data.index.name = "output"

        pd.testing.assert_series_equal(rv.range.data, expected_range_data)
        assert rv.range.domain == expected_range_domain
        assert rv.range.name == expected_range_name


class TestRangeCounts:

    def test_range_counts(self):
        """Test range_counts property of RandomVariable."""
        outputs = {"omega0": "a", "omega1": "a", "omega2": "b"}
        domain = SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega")
        X = RandomVariable(outputs=outputs, domain=domain, name="X")
        expected_range = pd.Series(
            data={"x0": "a", "x1": "b"},
            index=pd.Index(["x0", "x1"], name="output"),
            name="X",
        )
        expected_counts = pd.Series(
            data=[2, 1],
            index=pd.Index(["x0", "x1"], name="output"),
            name="count",
        )

        pd.testing.assert_series_equal(X.range_counts, expected_counts)
        pd.testing.assert_series_equal(X.range.data, expected_range)


class TestCallMethod:

    def test_call_method_on_sample_index(self):
        """Test calling RandomVariable on a single sample index."""
        outputs = {"s0": 1, "s1": 2}
        domain = SampleSpace(indices=["s0", "s1"], name="Omega")
        X = RandomVariable(outputs=outputs, domain=domain, name="X")

        assert X("s0") == 1

    def test_call_method_on_sample_indices(self):
        """Test calling RandomVariable on a list of sample indices."""
        outputs = {"s0": 1, "s1": 3, "s2": 5}
        domain = SampleSpace(indices=["s0", "s1", "s2"], name="Omega")
        X = RandomVariable(outputs=outputs, domain=domain, name="X")
        expected_rv = RandomVariable.from_pandas(
            data=pd.Series(
                [1, 5],
                index=pd.Index(["s0", "s2"], name="sample"),
                name="X",
            ),
            name="X|event",
        )
        rv_subset = X(["s0", "s2"])

        pd.testing.assert_series_equal(rv_subset.data, expected_rv.data)
        assert rv_subset.name == "X|event"

    def test_call_method_on_event(self):
        """Test calling RandomVariable on an Event."""
        outputs = {"s0": 1, "s1": 3, "s2": 5}
        domain = SampleSpace(indices=["s0", "s1", "s2"], name="Omega")
        X = RandomVariable(outputs=outputs, domain=domain, name="X")
        B = domain.get_event(["s0", "s2"], name="B")
        expected_rv = RandomVariable.from_pandas(
            data=pd.Series(
                [1, 5],
                index=pd.Index(["s0", "s2"], name="sample"),
                name="X",
            ),
            name="X|B",
        )
        restricted_rv = X(B)

        pd.testing.assert_series_equal(restricted_rv.data, expected_rv.data)
        assert restricted_rv.name == "X|B"

    def test_invalid_input_raises(self):
        """Test that invalid inputs raise appropriate exceptions."""
        outputs = {"s0": 1, "s1": 3, "s2": 5}
        domain = SampleSpace(indices=["s0", "s1", "s2"], name="Omega")
        X = RandomVariable(outputs=outputs, domain=domain, name="X")

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


class TestArithmetic:

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
            name="10-(X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "10-(X)"

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
            name="100/(X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "100/(X)"

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
            name="2**(X)",
        )
        pd.testing.assert_series_equal(Z.data, expected_values)
        assert Z.name == "2**(X)"

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
            assert "RandomVariable or scalar" in str(e)
