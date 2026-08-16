from numbers import Real

import pandas as pd
import sigalg as sa


class TestIntegration:
    def test_with_2D_parameter_and_measurable_domains(self):
        """Test creating an instance from a 2D parameter domain and 2D measurable domain. Test the __call__ method."""
        Theta = sa.Domain.cartesian_power(
            [0, 1], variable_names=["theta_0", "theta_1"], n=2, name="Theta"
        )
        X = sa.Domain.cartesian_power(
            ["a", "b"], n=2, variable_names=["x_0", "x_1"], name="X"
        )
        F = sa.SigmaAlgebra(
            domain=X,
            mapping={
                ("a", "a"): 0,
                ("a", "b"): 1,
                ("b", "a"): 1,
                ("b", "b"): 2,
            },
        )
        mu = sa.Measure(
            domain=F,
            mapping={
                0: 1,
                1: 4,
                2: 5,
            },
        )

        def mapping(*, theta_0, theta_1, x_0, x_1):  # noqa: D103
            if (theta_0, theta_1) == (0, 0):
                if (x_0, x_1) == ("a", "a"):
                    return 1
                elif (x_0, x_1) == ("a", "b"):
                    return 2
                elif (x_0, x_1) == ("b", "a"):
                    return 2
                elif (x_0, x_1) == ("b", "b"):
                    return -1
            elif (theta_0, theta_1) == (0, 1):
                if (x_0, x_1) == ("a", "a"):
                    return 0
                elif (x_0, x_1) == ("a", "b"):
                    return -3
                elif (x_0, x_1) == ("b", "a"):
                    return -3
                elif (x_0, x_1) == ("b", "b"):
                    return -1
            elif (theta_0, theta_1) == (1, 0):
                if (x_0, x_1) == ("a", "a"):
                    return 4
                elif (x_0, x_1) == ("a", "b"):
                    return 0
                elif (x_0, x_1) == ("b", "a"):
                    return 0
                elif (x_0, x_1) == ("b", "b"):
                    return 0
            elif (theta_0, theta_1) == (1, 1):
                if (x_0, x_1) == ("a", "a"):
                    return 5
                elif (x_0, x_1) == ("a", "b"):
                    return 1
                elif (x_0, x_1) == ("b", "a"):
                    return 1
                elif (x_0, x_1) == ("b", "b"):
                    return -1

        f = sa.ParametrizedMeasurableFunction.from_domains(
            measurable_domain=X,
            parameter_domain=Theta,
            sig_alg=F,
            measure=mu,
            mapping=mapping,
        )
        expected_domain = sa.Domain.cartesian_product([Theta, X])
        expected_data = pd.Series(
            [1, 2, 2, -1, 0, -3, -3, -1, 4, 0, 0, 0, 5, 1, 1, -1],
            index=expected_domain.data,
            name="f",
        )
        assert f.measurable_domain is X
        assert f.parameter_domain is Theta
        assert f.sig_alg is F
        assert f.measure is mu
        assert f.variable_names == ["theta_0", "theta_1", "x_0", "x_1"]
        assert f.parameter_names == ["theta_0", "theta_1"]
        assert f.measurable_names == ["x_0", "x_1"]
        pd.testing.assert_index_equal(f.domain.data, expected_domain.data)
        pd.testing.assert_series_equal(f.data, expected_data)

        complete_eval = f(theta_0=0, theta_1=1, x_0="a", x_1="b")
        assert isinstance(complete_eval, Real)
        assert complete_eval == -3

        g = f(theta_0=0, theta_1=0)
        expected_data = pd.Series([1, 2, 2, -1], index=X.data, name=g.name)
        assert isinstance(g, sa.MeasurableFunction)
        assert g.name == "f(theta_0=0, theta_1=0)"
        assert g.domain is X
        assert g.sig_alg is F
        assert g.measure is mu
        pd.testing.assert_series_equal(g.data, expected_data)

        g = f(x_0="a", x_1="b")
        expected_domain_data = pd.MultiIndex.from_product(
            [[0, 1], [0, 1]], names=["theta_0", "theta_1"]
        )
        expected_data = pd.Series(
            [2, -3, 0, 1], index=expected_domain_data, name=g.name
        )
        assert isinstance(g, sa.Function)
        assert g.name == "f(x_0=a, x_1=b)"
        assert g.domain == Theta
        pd.testing.assert_series_equal(g.data, expected_data)

        g = f(theta_0=0)
        expected_parameter_domain = sa.Domain(
            [0, 1], variable_names=["theta_1"], name="Theta|{theta_0=0}"
        )
        expected_domain = sa.Domain.cartesian_product([expected_parameter_domain, X])
        expected_data = pd.Series(
            [1, 2, 2, -1, 0, -3, -3, -1], index=expected_domain.data, name=g.name
        )
        assert g.measurable_domain is X
        assert g.sig_alg is F
        assert g.measure is mu
        pd.testing.assert_index_equal(g.domain.data, expected_domain.data)
        pd.testing.assert_series_equal(g.data, expected_data)

    def test_with_1D_parameter_and_measurable_domains(self):
        """Test creating an instance from a 1D parameter domain and 1D measurable domain. Test the __call__ method."""

        Theta = sa.Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        X = sa.Domain.from_sequence(size=3, variable_name="x")
        F = sa.SigmaAlgebra(
            domain=X,
            mapping={
                0: 0,
                1: 1,
                2: 1,
            },
        )
        mu = sa.Measure(
            domain=F,
            mapping={
                0: 1,
                1: 4,
            },
        )

        def mapping(*, theta, x):  # noqa: D103
            if theta == 0:
                if x == 0:
                    return 1
                elif x == 1:
                    return 2
                elif x == 2:
                    return 2
            elif theta == 1:
                if x == 0:
                    return 0
                elif x == 1:
                    return -3
                elif x == 2:
                    return -3

        f = sa.ParametrizedMeasurableFunction.from_domains(
            measurable_domain=X,
            parameter_domain=Theta,
            sig_alg=F,
            measure=mu,
            mapping=mapping,
        )
        expected_domain = sa.Domain.cartesian_product([Theta, X])
        expected_data = pd.Series(
            [1, 2, 2, 0, -3, -3],
            index=expected_domain.data,
            name="f",
        )
        assert f.measurable_domain is X
        assert f.parameter_domain is Theta
        assert f.sig_alg is F
        assert f.measure is mu
        assert f.variable_names == ["theta", "x"]
        assert f.parameter_names == ["theta"]
        assert f.measurable_names == ["x"]
        pd.testing.assert_index_equal(f.domain.data, expected_domain.data)
        pd.testing.assert_series_equal(f.data, expected_data)

        complete_eval = f(theta=0, x=1)
        assert isinstance(complete_eval, Real)
        assert complete_eval == 2

        g = f(theta=0)
        expected_data = pd.Series([1, 2, 2], index=X.data, name=g.name)
        assert isinstance(g, sa.MeasurableFunction)
        assert g.name == "f(theta=0)"
        assert g.domain is X
        assert g.sig_alg is F
        assert g.measure is mu
        pd.testing.assert_series_equal(g.data, expected_data)

        g = f(x=2)
        expected_data = pd.Series([2, -3], index=Theta.data, name=g.name)
        assert isinstance(g, sa.Function)
        assert g.name == "f(x=2)"
        assert g.domain == Theta
        assert g.variable_names == ["theta"]
        pd.testing.assert_series_equal(g.data, expected_data)

    def test_creation_with_prob_measure(self):
        """Test that creation with a probability measure returns an instance of ParametrizedRandomVariable."""

        Theta = sa.Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        X = sa.Domain.from_sequence(size=3, variable_name="x")
        F = sa.SigmaAlgebra(
            domain=X,
            mapping={
                0: 0,
                1: 1,
                2: 1,
            },
        )
        P = sa.ProbabilityMeasure(
            domain=F,
            mapping={
                0: 0.2,
                1: 0.8,
            },
        )

        def mapping(*, theta, x):  # noqa: D103
            if theta == 0:
                if x == 0:
                    return 1
                elif x == 1:
                    return 2
                elif x == 2:
                    return 2
            elif theta == 1:
                if x == 0:
                    return 0
                elif x == 1:
                    return -3
                elif x == 2:
                    return -3

        f = sa.ParametrizedMeasurableFunction.from_domains(
            measurable_domain=X,
            parameter_domain=Theta,
            sig_alg=F,
            measure=P,
            mapping=mapping,
        )

        assert isinstance(f, sa.ParametrizedRandomVariable)
        assert isinstance(f.measure_space, sa.ProbabilitySpace)
        assert isinstance(f(theta=0), sa.RandomVariable)
        assert isinstance(f(theta=0).measure_space, sa.ProbabilitySpace)
