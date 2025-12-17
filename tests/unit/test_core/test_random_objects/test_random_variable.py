import pandas as pd

from sigalg.core import RandomVariable, SampleSpace


class TestConstructor:

    def test_construction_from_outputs(self):
        """Test constructing RandomVariable from outputs."""
        outputs = {"omega0": 1, "omega1": 3, "omega2": 5}
        Omega = SampleSpace.generate_default(size=3, values_name="observation")
        Y = RandomVariable(outputs=outputs, domain=Omega, name="Y")
        expected_values = pd.Series(
            [1, 3, 5],
            index=pd.Index(["omega0", "omega1", "omega2"], name="observation"),
            name="Y",
        )
        pd.testing.assert_series_equal(Y.values, expected_values)
        assert Y.outputs == outputs
        assert Y.domain == Omega
        assert Y.name == "Y"

    def test_construction_from_values_basic(self):
        """Test constructing RandomVariable from pd.Series with default indices."""
        values = pd.Series([1, 3, 5], name="X")
        X = RandomVariable.from_values(values=values)
        expected_outputs = {0: 1, 1: 3, 2: 5}
        expected_domain = SampleSpace(indices=[0, 1, 2], name="Omega", values_name=None)
        pd.testing.assert_series_equal(X.values, values)
        assert X.outputs == expected_outputs
        assert X.domain == expected_domain
        assert X.name == "X"

    def test_construction_from_values_with_indices(self):
        """Test constructing RandomVariable from pd.Series with custom indices."""
        values = pd.Series(
            [1, 3, 5],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="Z",
        )
        Z = RandomVariable.from_values(values=values, name="Z")
        expected_outputs = {"a": 1, "b": 3, "c": 5}
        expected_domain = SampleSpace(
            indices=["a", "b", "c"], name="Omega", values_name="letters"
        )
        pd.testing.assert_series_equal(Z.values, values)
        assert Z.outputs == expected_outputs
        assert Z.domain == expected_domain
        assert Z.name == "Z"


class TestRange:

    def test_range_constructed_from_outputs(self):
        """Test range property of RandomVariable constructed from outputs."""
        Omega = SampleSpace.generate_default(size=3)
        outputs = {"omega0": 1, "omega1": 3, "omega2": 3}
        X = RandomVariable(outputs=outputs, domain=Omega, name="X")
        range_rv = X.range
        expected_series = pd.Series(
            data=[3, 1],
            index=pd.Index(["x0", "x1"], name="output"),
            name="range(X)",
        )
        expected_counts = pd.Series(
            data=[2, 1], index=expected_series.index, name="count"
        )
        pd.testing.assert_series_equal(range_rv.values, expected_series)
        pd.testing.assert_series_equal(X.range_counts, expected_counts)
        assert range_rv.name == "range(X)"

    def test_range_constructed_from_values_basic(self):
        """Test range property of RandomVariable constructed from values with default indices."""
        values = pd.Series([1, 3, 3], name="X")
        X = RandomVariable.from_values(values=values)
        range_rv = X.range
        expected_series = pd.Series(
            data=[3, 1],
            index=pd.Index(["x0", "x1"], name="output"),
            name="range(X)",
        )
        expected_counts = pd.Series(
            data=[2, 1], index=expected_series.index, name="count"
        )
        pd.testing.assert_series_equal(range_rv.values, expected_series)
        pd.testing.assert_series_equal(X.range_counts, expected_counts)
        assert range_rv.name == "range(X)"

    def test_range_from_values(self):
        """Test range property of RandomVariable constructed from values with custom indices."""
        values = pd.Series(
            [1, 3, 3],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="Y",
        )
        Y = RandomVariable.from_values(values=values, name="Y")
        range_rv = Y.range
        expected_series = pd.Series(
            data=[3, 1],
            index=pd.Index(["y0", "y1"], name="output"),
            name="range(Y)",
        )
        expected_counts = pd.Series(
            data=[2, 1], index=expected_series.index, name="count"
        )
        pd.testing.assert_series_equal(range_rv.values, expected_series)
        pd.testing.assert_series_equal(Y.range_counts, expected_counts)
        assert range_rv.name == "range(Y)"


class TestRangeCounts:

    def test_range_counts_constructed_from_outputs(self):
        """Test range_counts property of RandomVariable constructed from outputs."""
        Omega = SampleSpace.generate_default(size=3)
        outputs = {"omega0": 1, "omega1": 3, "omega2": 3}
        X = RandomVariable(outputs=outputs, domain=Omega, name="X")
        expected_counts = pd.Series(
            data=[2, 1], index=pd.Index(["x0", "x1"], name="output"), name="count"
        )
        pd.testing.assert_series_equal(X.range_counts, expected_counts)

    def test_range_counts_constructed_from_values_basic(self):
        """Test range_counts property of RandomVariable constructed from values with default indices."""
        values = pd.Series([1, 3, 3], name="X")
        X = RandomVariable.from_values(values=values)
        expected_counts = pd.Series(
            data=[2, 1], index=pd.Index(["x0", "x1"], name="output"), name="count"
        )
        pd.testing.assert_series_equal(X.range_counts, expected_counts)

    def test_range_counts_from_values(self):
        """Test range_counts property of RandomVariable constructed from values with custom indices."""
        values = pd.Series(
            [1, 3, 3],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="Y",
        )
        Y = RandomVariable.from_values(values=values, name="Y")
        expected_counts = pd.Series(
            data=[2, 1], index=pd.Index(["y0", "y1"], name="output"), name="count"
        )
        pd.testing.assert_series_equal(Y.range_counts, expected_counts)


class TestCallMethod:

    def test_call_method_on_sample_index(self):
        """Test calling RandomVariable on a single sample index."""
        values = pd.Series(
            [1, 3, 5],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="Y",
        )
        Y = RandomVariable.from_values(values=values, name="Y")

        assert Y("a") == 1

    def test_call_method_on_sample_indices(self):
        """Test calling RandomVariable on a list of sample indices."""
        Omega = SampleSpace.generate_default(size=3)
        outputs = {"omega0": 1, "omega1": 3, "omega2": 5}
        X = RandomVariable(outputs=outputs, domain=Omega, name="X")
        expected_rv = RandomVariable.from_values(
            values=pd.Series(
                [1, 5],
                index=pd.Index(["omega0", "omega2"], name="sample"),
                name="X|event",
            ),
            name="X|event",
        )
        result = X(["omega0", "omega2"])
        pd.testing.assert_series_equal(result.values, expected_rv.values)
        assert result.name == "X|event"

    def test_call_method_on_event(self):
        """Test calling RandomVariable on an Event."""
        Omega = SampleSpace.generate_default(size=3)
        outputs = {"omega0": 1, "omega1": 3, "omega2": 5}
        X = RandomVariable(outputs=outputs, domain=Omega, name="X")
        event = Omega.get_event(["omega0", "omega2"])
        expected_rv = RandomVariable.from_values(
            values=pd.Series(
                [1, 5],
                index=pd.Index(["omega0", "omega2"], name="sample"),
                name="X|A",
            ),
            name="X|A",
        )
        result = X(event)
        pd.testing.assert_series_equal(result.values, expected_rv.values)
        assert result.name == "X|A"


class TestGetItem:

    def test_getitem_on_int(self):
        """Test indexing RandomVariable with an integer."""
        Omega = SampleSpace.generate_default(size=3)
        outputs = {"omega0": 1, "omega1": 3, "omega2": 5}
        X = RandomVariable(outputs=outputs, domain=Omega, name="X")
        assert X[0] == 1

    def test_getitem_on_slice(self):
        """Test slicing RandomVariable with a slice object."""
        values = pd.Series(
            [1, 3, 5],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="Y",
        )
        Y = RandomVariable.from_values(values=values, name="Y")
        expected_rv = RandomVariable.from_values(
            values=pd.Series(
                [1, 3],
                index=pd.Index(["a", "b"], name="letters"),
                name="Y|event",
            ),
            name="Y|event",
        )
        result = Y[:2]
        pd.testing.assert_series_equal(result.values, expected_rv.values)
        assert result.name == "Y|event"

    def test_getitem_on_list_of_ints(self):
        """Test indexing RandomVariable with a list of integers."""
        values = pd.Series(
            [1, 3, 5],
            index=pd.Index(["a", "b", "c"], name="letters"),
            name="Y",
        )
        Y = RandomVariable.from_values(values=values, name="Y")
        expected_rv = RandomVariable.from_values(
            values=pd.Series(
                [1, 5],
                index=pd.Index(["a", "c"], name="letters"),
                name="Y|event",
            ),
            name="Y|event",
        )
        result = Y[[0, 2]]
        pd.testing.assert_series_equal(result.values, expected_rv.values)
        assert result.name == "Y|event"


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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
        pd.testing.assert_series_equal(Z.values, expected_values)
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
